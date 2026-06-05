"""Speech-timed section planning + clip assembly for video-maker-3.

Pipeline position:
    align words -> sentences_from_words() -> sections
                -> matcher.build_worklist()      # offline shortlist: vision + transcript + title
                -> [Claude picks the best clip(s) per section from the worklist]
                -> plan_from_worklist()          -> to_shots() -> render

Assembly rules (verbatim from the brief):
  * BEST-CLIP-FIRST CONCAT: lay the best-matching clip; if it is shorter than the section,
    append the next-best matching clip, and so on, trimming the last to land on the section
    duration. A clip finishes, the next starts immediately.
  * NO-REPEAT: a clip is used AT MOST TWICE in the whole video, and NEVER consecutively
    (the same clip can't run back-to-back, within a section or across a section boundary).
  * NO FREEZE: a short clip is followed by another clip — never a held/paused/last frame.

Pure / offline / dependency-free (no network, no API keys) so the assembly logic is fully
unit-testable. The only Claude touchpoint is the worklist->decisions handoff (matcher.py).
"""
from __future__ import annotations

import os
import re


# --------------------------------------------------------------------------- #
# 1. sentence segmentation, timed by real speech (from the TTS word alignment) #
# --------------------------------------------------------------------------- #
_SENT_END_WORD = re.compile(r"[.!?]+[\"')\]]*$")  # a word that ends a sentence


def sentences_from_words(words):
    """Split the aligned narration into SENTENCES timed by real speech.

    words: [{"w": str, "start": float, "end": float}] in order (absolute times across the
    whole video). A sentence closes on a word ending in . ! or ? Each sentence carries its
    true spoken duration. Returns
    [{"start","end","dur","text","i0","i1","words":[{"idx","w","start","end"}]}].
    """
    sents, cur = [], []
    for k, w in enumerate(words):
        cur.append((k, w))
        if _SENT_END_WORD.search(w.get("w", "")):
            sents.append(_mk_span(cur))
            cur = []
    if cur:
        sents.append(_mk_span(cur))
    return sents


def _mk_span(items):
    i0, i1 = items[0][0], items[-1][0]
    start, end = items[0][1]["start"], items[-1][1]["end"]
    return {
        "start": round(start, 3), "end": round(end, 3), "dur": round(end - start, 3),
        "text": " ".join(it[1]["w"] for it in items).strip(), "i0": i0, "i1": i1,
        "words": [{"idx": k, "w": w.get("w", ""), "start": w["start"], "end": w["end"]}
                  for k, w in items],
    }


def words_per_second(words):
    """Observed speech rate (words/sec) across the aligned narration — informational."""
    if not words:
        return 0.0
    span = (words[-1]["end"] - words[0]["start"]) or 1.0
    return round(len(words) / span, 2)


def sections_from_sentences(sentences, max_sec=None):
    """Sections the matcher/assembler consume: one per sentence (optionally splitting a very
    long sentence on commas so a single section is never longer than max_sec seconds, which
    keeps each Claude query focused). Returns [{"index","start","end","dur","text"}]."""
    max_sec = float(max_sec if max_sec is not None
                    else os.environ.get("VM_MAX_SECTION_SEC", "9.0"))
    out = []
    for s in sentences:
        if s["dur"] <= max_sec or len(s.get("words", [])) < 8:
            out.append({"start": s["start"], "end": s["end"], "dur": s["dur"], "text": s["text"]})
            continue
        # split a long sentence into roughly-equal contiguous word chunks on comma boundaries
        ws = s["words"]
        n_chunks = max(2, int(round(s["dur"] / max_sec + 0.49)))
        size = max(1, len(ws) // n_chunks)
        i = 0
        while i < len(ws):
            chunk = ws[i:i + size]
            # extend to the next comma/clause end if close, so cuts fall on phrase boundaries
            j = i + len(chunk)
            while j < len(ws) and j - i < size + 5 and not chunk[-1]["w"].endswith((",", ";", ":")):
                chunk = ws[i:j + 1]; j += 1
            out.append({"start": round(chunk[0]["start"], 3), "end": round(chunk[-1]["end"], 3),
                        "dur": round(chunk[-1]["end"] - chunk[0]["start"], 3),
                        "text": " ".join(x["w"] for x in chunk).strip()})
            i += len(chunk)
    for k, sec in enumerate(out):
        sec["index"] = k
    return out


# --------------------------------------------------------------------------- #
# 2. corpus moments + stable identities                                        #
# --------------------------------------------------------------------------- #
def build_moments(library_index):
    """Flatten the shared-library index into usable B-roll moments (skip any flagged)."""
    out = []
    for url, v in (library_index or {}).items():
        for seg in v.get("segments", []):
            if seg.get("talking_head"):
                continue
            out.append({"url": url, "id": v.get("id"), "channel": v.get("channel", ""),
                        "title": v.get("title", ""), "source": v.get("source", "shared-library"),
                        "niche": v.get("niche", ""), "seg": seg})
    return out


def moment_key(m):
    """No-repeat identity: a window is (source url, rounded start). Two clips with the same
    key are 'the same clip' for the at-most-twice / never-consecutive rules."""
    return (m["url"], round(float(m["seg"].get("start", 0)), 2))


def cand_id(m):
    """Stable id used in the Claude worklist + decisions (human-pasteable)."""
    return f"{m.get('id') or m['url']}@{round(float(m['seg'].get('start', 0)), 2)}"


def index_by_cand(moments):
    return {cand_id(m): m for m in moments}


# --------------------------------------------------------------------------- #
# 3. assembly: cover each section best-first, <=2 uses, never consecutive      #
# --------------------------------------------------------------------------- #
def _lay(section, ordered, clips, filled, use_count, state, max_uses, min_clip, materialize=None):
    """Append clips from `ordered` (best-first) onto `clips` until the section is covered.

    Enforces, GLOBALLY across the whole video via the shared `use_count` + `state`:
      * a moment used at most `max_uses` times,
      * never two consecutive shots from the same moment (state['last_key']),
    and trims the last clip to land on the section duration.

    When a `materialize(moment, take, shot_seq) -> (final_path, actual_dur) | None` callback is
    given, a moment is only PLACED if it materializes (downloads + passes the light QC); a
    moment that won't materialize is skipped to the next-best, exactly like an editor reaching
    for the next clip when one doesn't pan out. Pure (no I/O) when materialize is None.
    Returns (clips, filled)."""
    target = float(section.get("dur") or (section.get("end", 0) - section.get("start", 0)))
    for m in ordered:
        if filled >= target - 0.4:
            break
        k = moment_key(m)
        if use_count.get(k, 0) >= max_uses:
            continue
        if state.get("last_key") == k:
            continue  # never back-to-back (within a section or across the boundary)
        seg = m["seg"]
        avail = float(seg.get("end", 0)) - float(seg.get("start", 0))
        if avail < min_clip:
            continue
        need = target - filled
        take = min(avail, need)
        if 0 < need - take < min_clip and take < avail:   # avoid a tiny leftover next clip
            take = min(avail, need + min_clip)
        cs = float(seg.get("start", 0))
        final_path = None
        if materialize is not None:
            res = materialize(m, round(take, 3), state.get("shot_seq", 0))
            if not res:
                continue  # clip unavailable / failed light QC -> reach for the next-best
            final_path, take = res[0], float(res[1])
            if take < min_clip:
                continue
        clips.append({
            "url": m["url"], "id": m["id"], "channel": m["channel"], "source": m["source"],
            "title": m.get("title", ""),
            "clip_start": round(cs, 3), "clip_end": round(cs + take, 3), "dur": round(take, 3),
            "match": round(float(m.get("match", 0.0)), 3),
            "scene": section.get("text", ""), "desc": seg.get("embed_text", seg.get("desc", "")),
            "transcript": seg.get("transcript", ""), "label": seg.get("label", ""),
            "final_clip_path": final_path,
            "rank": "best" if not clips else "runner-up",
        })
        use_count[k] = use_count.get(k, 0) + 1
        state["last_key"] = k
        state["shot_seq"] = state.get("shot_seq", 0) + 1
        filled += take
    return clips, round(filled, 3)


def plan_from_worklist(worklist, decisions, library_index, materialize=None, progress_cb=None):
    """Assemble every section's clips from the Claude decisions, honouring the no-repeat /
    concat-fill / no-freeze rules with a GLOBAL use-count + last-clip state.

    worklist:  [{"index","start","end","dur","text","candidates":[{"cand_id",...}]}] (offline order)
    decisions: {section_index(str|int): [cand_id, ...]}  — Claude's ordered picks per section.
               Missing/empty for a section -> fall back to the offline candidate order.
    materialize: optional callback (moment, take, shot_seq) -> (final_path, actual_dur) | None,
                 so only clips that actually download + pass the light QC are placed.
    Returns the worklist items enriched with `clips` + `covered`. Uncovered tails (rare) are
    filled with the globally least-used unused window — a real moving clip, never a freeze.
    """
    moments = build_moments(library_index)
    by_cand = index_by_cand(moments)
    max_uses = int(os.environ.get("VM_MAX_CLIP_USES", "2"))
    min_clip = float(os.environ.get("VM_MIN_CLIP_SEC", "1.0"))
    decisions = decisions or {}
    use_count, state = {}, {"last_key": None, "shot_seq": 0}
    out = []
    for item in worklist:
        idx = item.get("index")
        ranked = [by_cand[c["cand_id"]] for c in item.get("candidates", [])
                  if c.get("cand_id") in by_cand]
        picked_ids = decisions.get(str(idx))
        if picked_ids is None:
            picked_ids = decisions.get(idx, [])
        picks = [by_cand[c] for c in (picked_ids or []) if c in by_cand]
        # ordered = Claude's picks first, then the offline-ranked remainder (dedup)
        ordered, seen = [], set()
        for m in picks + ranked:
            cid = cand_id(m)
            if cid not in seen:
                seen.add(cid); ordered.append(m)
        clips, covered = _lay(item, ordered, [], 0.0, use_count, state, max_uses, min_clip, materialize)
        target = float(item.get("dur") or (item.get("end", 0) - item.get("start", 0)))
        if covered < target - 0.6:   # last-resort coverage: least-used unused moment, no freeze
            fallback = sorted(
                moments,
                key=lambda m: (use_count.get(moment_key(m), 0),
                               -(float(m["seg"].get("end", 0)) - float(m["seg"].get("start", 0)))))
            clips, covered = _lay(item, fallback, clips, covered, use_count, state,
                                  max_uses, min_clip, materialize)
        out.append({**item, "clips": clips, "covered": covered})
        if progress_cb:
            progress_cb(f"  section {idx} {item.get('start')}-{item.get('end')}s "
                        f"({target:.1f}s): {len(clips)} clip(s), {covered:.1f}s covered")
    return out


# --------------------------------------------------------------------------- #
# 4. flatten to sequential shots the renderer consumes                         #
# --------------------------------------------------------------------------- #
def to_shots(planned):
    """Flatten planned sections into back-to-back shots with absolute start times. Each shot
    carries the source window [verified_start, verified_end] to materialize + render."""
    shots, idx = [], 0
    for s in planned:
        t = float(s.get("start", 0))
        for c in s.get("clips", []):
            shots.append({
                "shot_idx": idx, "start_sec": round(t, 3), "duration": c["dur"],
                "final_clip_path": c.get("final_clip_path"),
                "scene_description": c["scene"], "search_query": c["scene"],
                "candidate": {
                    "id": c["id"], "url": c["url"], "channel": c["channel"],
                    "title": c.get("title", ""), "source": c["source"], "verified": True,
                    "verified_start": c["clip_start"], "verified_end": c["clip_end"],
                    "match": c["match"], "scene": c["scene"], "rank": c.get("rank", ""),
                    "verify_reason": "section:" + (c.get("desc", "")[:80]),
                },
            })
            t += c["dur"]
            idx += 1
    return shots


def assembly_report(planned):
    """A human/agent-readable audit of the no-repeat rules (for the run report + fact-check):
    per-clip use counts and any adjacency. Returns {uses, max_uses_seen, consecutive_violations}."""
    seq = []
    for s in planned:
        for c in s.get("clips", []):
            seq.append((c["url"], round(float(c["clip_start"]), 2)))
    uses = {}
    for k in seq:
        uses[k] = uses.get(k, 0) + 1
    consec = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    return {"n_shots": len(seq), "n_distinct": len(uses),
            "max_uses_seen": max(uses.values()) if uses else 0,
            "consecutive_violations": consec}

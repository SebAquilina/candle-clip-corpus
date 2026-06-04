"""v2.2: seed the editor-style TOPIC LIBRARY from the shared, pre-vetted clip database.

The shared clip DB (built by the YT-Clip-Classifier pipeline and synced via Google
Drive) is a per-niche library of YouTube SOURCE videos already classified into
labelled, timestamped B-roll moments, with talking-head / on-screen-text /
non-action windows FLAGGED. That is *exactly* a topic-library index — already
built, already vetted, and FREE (no Gemini "watch the video" calls). This module
loads it into the same `{url: {channel, id, source, segments:[...]}}` shape that
``library_match.match`` consumes, so vetted on-topic moments are the PRIMARY
source for every shot (Rule v25.0), ahead of Gemini discovery and Pexels.

Because the store already flags talking heads and on-screen text, every moment we
expose as usable is consistent with Rules v22.3 (no overlay text) and v22.4 (no
talking heads): flagged windows are emitted with ``talking_head: true`` so
``library_match`` skips them.

Data source: a local copy of the store directory (the `records/` folder of
per-video JSON records, pulled down from Drive by the orchestrator). Point
``YTA_SHARED_DB`` at that directory. Pure and offline — no network, no app deps,
no dependency on the ytclip package — so it imports and runs anywhere.
"""
from __future__ import annotations

import glob
import json
import os
import re

# A window is unusable as B-roll (and so emitted as talking_head=True, which
# library_match skips) if it is talking-head, has on-screen text, is blank, or is
# not a hands-on action step.
_NON_FOOTAGE_LABELS = {
    "talking_head", "intro_titlecard", "outro_cta", "transition",
    "other_unclear", "blank",
}


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _niche_matches(a_text, b_text):
    """Loose token overlap with prefix matching (candle ~ candles, make ~ making)."""
    if not b_text:
        return False
    if not a_text:
        return True
    at, bt = _tokens(a_text), _tokens(b_text)
    for a in at:
        for b in bt:
            if a == b:
                return True
            if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
                return True
    return False


def _store_dir(explicit=None):
    """Find the shared store dir (one containing a `records/` folder)."""
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("YTA_SHARED_DB"):
        cands.append(os.environ["YTA_SHARED_DB"])
    for base in (os.environ.get("WS", ""), os.getcwd(), os.path.expanduser("~")):
        if base:
            cands += [os.path.join(base, "shared_db_v2"),
                      os.path.join(base, "outputs", "shared_db_v2"),
                      os.path.join(base, "shared_db"),
                      os.path.join(base, "shared_clip_db"),
                      os.path.join(base, "outputs", "shared_db")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "records")):
            return c
    return None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _objective_override(w):
    """Label-INDEPENDENT exclusion from objective signals (face / text / caption).

    Fires only when the signal is actually present (empty string / missing -> no-op),
    so it is a harmless no-op on a corpus whose detectors never ran, and airtight once
    the signals exist. Catches a talking-head / on-screen-text window that Claude
    MISLABELLED as an action (e.g. labelled ``pour_wax`` but face_score 0.9). Returns a
    short reason or None. Kept byte-for-byte in sync with
    ``corpus_builder._objective_override`` (the ingest-time guard)."""
    fs, ts = _num(w.get("face_score")), _num(w.get("text_score"))
    # Zero-tolerance face mode (YTA_NO_FACES=1, default): ANY face signal at all excludes
    # the window, regardless of label/is_step. Legacy mode keeps the YTA_FACE_MAX threshold.
    no_faces = os.environ.get("YTA_NO_FACES", "1").strip().lower() not in ("0", "false", "no", "")
    if fs is not None and (fs > 0.0 if no_faces else fs >= float(os.environ.get("YTA_FACE_MAX", "0.40"))):
        return "face_score"
    if ts is not None and ts >= float(os.environ.get("YTA_TEXT_MAX", "0.50")):
        return "text_score"
    if int(w.get("has_caption") or 0) and os.environ.get("YTA_DROP_CAPTIONED", "0") == "1":
        return "has_caption"
    return None


def _window_is_flagged(w, flagged_idx):
    if w.get("window_index") in flagged_idx:
        return True
    if w.get("filter_reason"):
        return True
    if not int(w.get("is_step") or 0):
        return True
    if w.get("action_label") in _NON_FOOTAGE_LABELS:
        return True
    # objective overrides (only fire when the signal exists; empty -> no-op) so a
    # mislabelled talking head / on-screen-text window can never reach a render.
    if _objective_override(w):
        return True
    return False


# Boilerplate the upstream classifier appends to description_detailed:
#   "<phase> phase, ~N% in, M:SS-M:SS." (en-dash U+2013, em-dash, or hyphen; trailing . optional).
# It's on ~92% of the corpus and pollutes embedding text with no visual signal — strip it.
_BOILERPLATE = re.compile(
    r"\s*[A-Za-z][A-Za-z ]* phase,\s*~\d{1,3}% in,\s*\d+:\d\d\s*[–—-]\s*\d+:\d\d\.?\s*$")


def _humanize_label(label):
    return (label or "").replace("_", " ").strip()


def _embed_text(w):
    """Clean, action-led text for EMBEDDING (not for display): strip the timestamp/phase
    boilerplate and PREPEND the humanized action_label so 'pour' vs 'melt' vs 'measure'
    can't collapse on the shared noun 'wax'. The displayed `desc` is kept untouched."""
    base = (w.get("description_detailed") or w.get("description") or "").strip()
    base = _BOILERPLATE.sub("", base).strip()
    verb = _humanize_label(w.get("action_label", ""))
    if verb and verb not in base.lower():
        base = f"{verb}. {base}" if base else verb
    return base[:240]


def _record_to_entry(rec):
    """One per-video record -> (url, {id, channel, source, niche, segments})."""
    vid = rec.get("video_id")
    url = rec.get("video_url") or (
        f"https://www.youtube.com/watch?v={vid}" if vid else None)
    if not url:
        return None

    # v2 per-second corpus (corpus-revamp): every window already passed the second-by-second
    # face+OCR purge AND the description face backstop, so it is a CLEAN survivor by
    # construction — none of the v1 is_step/flagged/face_score logic applies (rejects live in
    # a separate rejected/ store). The window-level embed_text (vision) and transcript (speech)
    # are precomputed and kept SEPARATE so the matcher weighs them independently.
    if rec.get("schema") == "per_second_v1":
        segs = []
        for w in rec.get("windows", []):
            start = float(w.get("start_s", 0.0) or 0.0)
            end = float(w.get("end_s", start) or start)
            if end <= start:
                end = start + 5.0
            et = (w.get("embed_text") or "").strip()
            label = w.get("action_label", "")
            segs.append({
                "start": start, "end": end,
                "desc": (et or label)[:160],
                "embed_text": (et or label)[:400],        # VISION: what is on screen
                "transcript": (w.get("transcript") or "").strip()[:400],  # SEPARATE: speech
                "talking_head": False,                    # clean by construction
                "label": label,
            })
        if not segs:
            return None
        return url, {"id": vid, "channel": (rec.get("channel") or "").strip(),
                     "title": (rec.get("video_title") or "").strip(),  # source title: context for the matcher
                     "source": "shared-library-v2", "niche": rec.get("niche", ""),
                     "segments": segs}

    flagged_idx = {f.get("window_index") for f in rec.get("flagged", [])}
    segs = []
    for w in rec.get("windows", []):
        flagged = _window_is_flagged(w, flagged_idx)
        start = float(w.get("start_s", 0.0) or 0.0)
        end = float(w.get("end_s", start) or start)
        if end <= start:
            end = start + 5.0
        segs.append({
            "start": start,
            "end": end,
            "desc": (w.get("description_detailed") or w.get("description")
                     or w.get("action_label") or "")[:160],
            "embed_text": _embed_text(w),
            # flagged (talking-head / on-screen-text / non-action) -> skipped by match
            "talking_head": bool(flagged),
            "label": w.get("action_label", ""),
        })
    if not any(not s["talking_head"] for s in segs):
        return None  # no usable footage in this source
    return url, {
        "id": vid,
        "channel": (rec.get("channel") or "").strip(),
        "source": "shared-library",
        "niche": rec.get("niche", ""),
        "segments": segs,
    }


def load_index(topic="", store_dir=None, progress_cb=None):
    """Load the shared clip DB into the topic-library index shape.

    Filtering: if ``YTA_SHARED_NICHE`` is set, keep records whose niche matches it;
    else if a ``topic`` is given, keep records whose niche overlaps the topic;
    else keep everything. Returns ``{}`` (caller falls back) when no store is found.
    """
    d = _store_dir(store_dir)
    if not d:
        if progress_cb:
            progress_cb("shared-library: no store found (set YTA_SHARED_DB to the store dir)")
        return {}
    require_niche = os.environ.get("YTA_SHARED_NICHE")
    index, matched, total = {}, 0, 0
    for p in sorted(glob.glob(os.path.join(d, "records", "*.json"))):
        if os.path.basename(p) == "manifest.json":   # drive_sync checksum file, not a record
            continue
        try:
            rec = json.load(open(p))
        except (ValueError, OSError):
            continue
        if not (isinstance(rec, dict) and rec.get("video_id")):
            continue
        total += 1
        niche = rec.get("niche", "")
        if require_niche:
            if require_niche.lower() != (niche or "").lower() and not _niche_matches(require_niche, niche):
                continue
        elif topic and niche and not _niche_matches(topic, niche):
            continue
        entry = _record_to_entry(rec)
        if entry:
            index[entry[0]] = entry[1]
            matched += 1
    if progress_cb:
        usable = sum(sum(1 for s in v["segments"] if not s["talking_head"]) for v in index.values())
        progress_cb(f"shared-library: {matched}/{total} vetted source videos, "
                    f"{usable} usable moments from {d}")
    return index

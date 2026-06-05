"""make.py — video-maker-3 driver.

Streamlined pipeline, corpus-only sourcing, Claude in the matching loop:

  plan   TTS(narration) -> align words -> speech-timed sections
         -> matcher.build_worklist (vision+transcript shortlist) -> match_worklist.json   [STOP]
  ---->  [Claude reads match_worklist.json and writes match_decisions.json: the best clip(s)
          per section, judged on BOTH vision and transcript in the context of the title]
  build  plan_from_worklist(decisions, materialize=download+fit+lightQC)   # <=2 uses, never
         consecutive, best-clip-first concat, no freeze -> to_shots -> render -> final mp4
         (then run validate_render.py — the non-skippable safety gate)

Usage:
  python make.py plan  --topic top10 --script script.md --title "Top 10 Candle Making Tricks"
  python make.py build --topic top10                  # uses match_decisions.json if present
  python make.py all   --topic top10 --script ...      # plan + build with the OFFLINE order
  python make.py --selftest
"""
import os, sys
# --- portable header (auto-generated; do not reorder) ---
_WS = os.environ.get('WS') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = _WS
os.environ.setdefault('STORAGE_DIR', os.path.join(_WS, 'state'))
_VENV_BIN = os.path.join(_WS, '.venv', 'bin')
if os.path.isdir(_VENV_BIN):
    os.environ['PATH'] = _VENV_BIN + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so 'app' + '_autobootstrap' resolve
try:
    import _autobootstrap
    _autobootstrap.auto_bootstrap_if_needed()
except Exception:
    pass
# --- end header ---
import json, argparse, asyncio, subprocess, re
from pathlib import Path

BASE = Path(os.path.join(WS, 'state', 'runs'))
PAD = float(os.environ.get('VM_DOWNLOAD_PAD', '1.5'))         # tail padding so a clip is never short
# The skill is NICHE-AGNOSTIC: the corpus (an external, niche-specific GitHub repo supplied at
# runtime) defines the niche. VM_NICHE overrides; otherwise it's derived from the corpus records.
NICHE = os.environ.get('VM_NICHE', '')


def _derive_niche(index):
    """Most common niche across loaded corpus records, so candle/soap/etc. all 'just work'."""
    import collections
    c = collections.Counter((v.get("niche") or "").strip() for v in (index or {}).values())
    c.pop("", None)
    return c.most_common(1)[0][0] if c else ""


# --------------------------------------------------------------------------- #
# script -> paragraphs                                                         #
# --------------------------------------------------------------------------- #
_META_RE = re.compile(
    r"(^(persona|voice|narration script|script|title|author|notes?)\s*[:\-])"
    r"|(one idea per (paragraph|sentence))|(b-?roll window)|(\bwpm\b)|(~?\d+\s*minutes?\b)",
    re.I)


def read_script(path):
    """Narration paragraphs from a .md/.txt. Drops markdown headings and '---' rules at the
    line level, then — crucially — filters metadata at the PARAGRAPH level (after joining
    wrapped lines), so a multi-line front-matter block like
        Persona: ... One idea per paragraph so the video-maker
        can size a B-roll window per sentence. ~150 wpm -> ~10 minutes.
    is dropped whole, not just its first line. Splits the rest on blank lines."""
    text = Path(path).read_text(encoding="utf-8")
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#") or (set(s) == {"-"} and len(s) >= 3):   # heading or '---' rule
            continue
        lines.append(ln)
    paras = []
    for b in re.split(r"\n\s*\n", "\n".join(lines)):
        p = " ".join(x.strip() for x in b.splitlines() if x.strip()).strip()
        if len(p.split()) < 4:
            continue
        if _META_RE.search(p[:120]):   # front-matter / production notes, not narration
            continue
        paras.append(p)
    return paras


# --------------------------------------------------------------------------- #
# TTS + alignment -> absolute word timeline -> sections                        #
# --------------------------------------------------------------------------- #
def _probe(p):
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return 0.0


async def _tts(paras, audio_dir):
    from app.services import tts_provider as tp
    print(f"[tts] provider: {tp.describe()}")
    return await tp.synthesize_all(paras, audio_dir)


def build_sections(paras, audio_dir):
    from app.services.v2 import align
    from app.services.v3 import section_planner as sp
    words, offset = [], 0.0
    for i, para in enumerate(paras):
        aud = Path(audio_dir) / f"para_{i:03d}.mp3"
        dur = _probe(aud)
        if dur <= 0:
            continue
        ww = align.transcribe_words(aud) or align.synthesize_word_timings(para, dur)
        if not ww:
            offset += dur; continue
        for w in align.align([para], ww):
            s = (w.start if w.start is not None else 0.0) + offset
            e = (w.end if w.end is not None else s) + offset
            words.append({"w": w.word_punct, "start": round(s, 3), "end": round(max(e, s), 3)})
        offset += dur
    sentences = sp.sentences_from_words(words)
    return sp.sections_from_sentences(sentences), round(offset, 2)


# --------------------------------------------------------------------------- #
# PLAN: shortlist candidates per section (Claude then picks)                   #
# --------------------------------------------------------------------------- #
def cmd_plan(args):
    from app.services.v3 import shared_library as lib
    from app.services.v3 import embeddings as emb
    from app.services.v3 import matcher
    run = BASE / args.topic
    audio_dir = run / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    paras = read_script(args.script)
    print(f"[plan] {len(paras)} narration paragraphs")
    asyncio.run(_tts(paras, audio_dir))

    sections, audio_dur = build_sections(paras, audio_dir)
    (run / "sections.json").write_text(json.dumps(sections, indent=2))
    print(f"[plan] {len(sections)} speech-timed sections, narration {audio_dur:.0f}s")

    os.environ.setdefault("YTA_SHARED_NICHE", "")  # don't over-filter; corpus is single-niche
    index = lib.load_index(topic=NICHE, progress_cb=print)
    if not index:
        print("[plan] FATAL: no corpus found. Point YTA_SHARED_DB at a corpus's"
              " outputs/shared_db_v2 (e.g. `bash scripts/get_corpus.sh <repo-url>`)."); sys.exit(2)
    niche = NICHE or _derive_niche(index)          # niche comes from the corpus when unset
    if niche:
        print(f"[plan] niche (from corpus): {niche!r}")

    title = args.title or args.topic
    wl = matcher.build_worklist(sections, index, emb.embed_many, emb.cosine,
                                project_title=title, niche=niche, progress_cb=print)
    wpath = run / "match_worklist.json"
    matcher.write_worklist(wl, wpath, project_title=title, niche=niche)
    print(f"\n[plan] wrote {wpath}")

    # ----- PARALLEL PICKS: emit N slice files so a TEAM of agents picks in parallel -----
    # The operator spawns one agent per slice, each agent picks clips for ITS sections,
    # writes decisions_<i>.json. cmd_build merges them safely by cand_id (so even if an
    # agent mis-keys with positional indices, the merge auto-recovers).
    n_slices = max(1, int(getattr(args, "slices", 0) or os.environ.get("VM_PICK_SLICES", "4")))
    sl_dir = run / "worklist_slices"; sl_dir.mkdir(parents=True, exist_ok=True)
    # clear any older slices from a prior plan
    for old in sl_dir.glob("slice_*.json"):
        old.unlink()
    secs = wl
    if n_slices > 1 and len(secs) >= 2:
        chunk = (len(secs) + n_slices - 1) // n_slices
        for i in range(n_slices):
            ch = secs[i * chunk:(i + 1) * chunk]
            if not ch:
                continue
            json.dump({"_instructions": (
                "Pick the best 1-4 cand_id per section, BEST FIRST. Key your output by each "
                "section's `index` FIELD (NOT slice/list position), AS A STRING. Every cand_id "
                "MUST come from THAT section's `candidates`. Prefer clips whose vision AND "
                "(when spoken) transcript convey the narration in the context of the title. "
                f"Write your output to state/runs/{args.topic}/decisions_{i}.json"),
                "project_title": title, "niche": niche, "slice_index": i,
                "section_index_range": [ch[0]["index"], ch[-1]["index"]],
                "sections": ch}, open(sl_dir / f"slice_{i:02d}.json", "w"))
        print(f"[plan] wrote {n_slices} slices under {sl_dir} (sections per slice: ~{chunk})")
        print(f"[plan] NEXT — Claude-in-the-loop (PARALLEL): spawn {n_slices} agents in parallel,")
        print(f"       one per slice. Each agent reads worklist_slices/slice_<i>.json and writes")
        print(f"       decisions_<i>.json. Then: python make.py build --topic {args.topic}")
    else:
        print(f"[plan] NEXT (Claude-in-the-loop): read {wpath}; pick clips per section; write")
        print(f"       {run/'match_decisions.json'} = {{\"<section_index>\": [cand_id,...]}}.")
        print(f"       Then: python make.py build --topic {args.topic}")


# --------------------------------------------------------------------------- #
# BUILD: assemble (materialize-on-place) -> render                            #
# --------------------------------------------------------------------------- #
def _make_materializer(run):
    from app.services import youtube
    from app.services.v3 import duration_ladder as dl
    from app.services.v3 import section_planner as sp
    import render as rnd
    raw_dir = run / "raw"; seg_dir = run / "segs"
    raw_dir.mkdir(parents=True, exist_ok=True); seg_dir.mkdir(parents=True, exist_ok=True)

    # Source credit: every clip gets a "via <channel> on YouTube" overlay (REQUIRED attribution).
    # Channel comes from the corpus record if present, else it's resolved once via yt-dlp metadata
    # and CACHED to disk (channels.json) so re-runs/restarts never re-resolve. Resolution is ON by
    # default (VM_RESOLVE_CHANNELS=0 forces the bare "via YouTube" fallback, e.g. fully offline).
    chan_path = run / "channels.json"
    try:
        chan_cache = json.loads(chan_path.read_text())
    except Exception:
        chan_cache = {}
    resolve = os.environ.get("VM_RESOLVE_CHANNELS", "1").strip().lower() not in ("0", "false", "no")

    def credit(moment):
        ch = (moment.get("channel") or "").strip()
        url = moment.get("url", "")
        if not ch and url in chan_cache:
            ch = chan_cache[url]
        if not ch and resolve and url:
            try:
                ch = youtube.fetch_channel(url) or ""
            except Exception:
                ch = ""
            chan_cache[url] = ch
            try: chan_path.write_text(json.dumps(chan_cache))
            except Exception: pass
        return f"via {ch} on YouTube" if ch else "via YouTube"
    # Per-clip QC: black/validity is always on (cheap, no cv2). A per-clip FACE/TEXT recheck
    # (VM_LIGHT_QC_FACES=1, ON by default) catches the few corpus windows whose face flickers
    # between the corpus purge's per-second samples — so the build rejects them during assembly
    # and reaches for the next-best face-free clip, instead of only failing at the final gate.
    # It runs the SAME detector at the SAME density as the gate (clip_checks, GATE_FPS sampling)
    # inside a SUBPROCESS with a timeout, so a cv2/YuNet C-level hang can never wedge the build.
    face_qc = os.environ.get("VM_LIGHT_QC_FACES", "1").strip().lower() in ("1", "true", "yes")
    raw_status = {}  # window_key -> raw Path or None

    # Build QC is FACE-ONLY by default: the corpus is already OCR-text-purged and the final
    # gate still runs the full text scan, so re-running the slow 3-pass OCR per clip here is
    # redundant and was the main hang/latency source. Faces are the real corpus miss, so we
    # scan those. Set VM_QC_TEXT=1 to also text-scan per clip.
    qc_text = os.environ.get("VM_QC_TEXT", "0").strip().lower() in ("1", "true", "yes")

    def _face_text_ok(seg_path):
        if not face_qc:
            return True
        # gate-aligned face scan (same detector + density as validate_render). Reject on
        # >=1 face frame (zero-tolerance), so a clip passing here also passes the gate's face check.
        textscan = "th=cc.scan_video_text(%r);" % str(seg_path) if qc_text else "th=[];"
        code = ("import sys,os,json;sys.path.insert(0,'scripts');os.environ['WS']=%r;"
                "import clip_checks as cc;"
                "fh=cc.scan_video_talking_head(%r); " + textscan +
                "print('RESULT'+json.dumps({'face':len(fh),'text':len(th),"
                "'gf':cc.GATE_FACE_HITS,'gt':cc.GATE_TEXT_HITS}))") % (
                    os.path.abspath('.'), str(seg_path))
        try:
            out = subprocess.check_output([sys.executable, "-c", code],
                                          timeout=int(os.environ.get("VM_LIGHT_QC_TIMEOUT", "120")),
                                          stderr=subprocess.DEVNULL).decode()
            v = json.loads([l for l in out.splitlines() if l.startswith("RESULT")][-1].partition("RESULT")[2])
            return v["face"] < v["gf"] and v["text"] < v["gt"]
        except Exception:
            # FAIL-CLOSED (zero-tolerance): a window we can't verify face-free is unusable,
            # so the assembler reaches for the next-best instead of risking a face slipping in.
            return False

    # VM_CACHED_ONLY: never hit the network; place only windows already downloaded under
    # raw/. Useful when YouTube is rate-limiting (429/503) — the build assembles from the
    # cached pool instead of stalling on doomed downloads.
    cached_only = os.environ.get("VM_CACHED_ONLY", "0").strip().lower() in ("1", "true", "yes")

    # face/text QC verdict per window, PERSISTED to disk so the (expensive) cv2 scans survive
    # a restart or a process death — the build resumes the QC instead of re-scanning.
    qc_path = run / "qc_cache.json"
    try:
        face_status = {tuple(k.rsplit("@", 1)[:1]) + (float(k.rsplit("@", 1)[1]),): v
                       for k, v in json.loads(qc_path.read_text()).items()}
    except Exception:
        face_status = {}

    def _save_qc():
        try:
            qc_path.write_text(json.dumps({f"{k[0]}@{k[1]}": v for k, v in face_status.items()}))
        except Exception:
            pass

    # If a CORPUS-WIDE clip cache exists at outputs/clip_cache/<vid>_<start>.mp4 (the layout
    # the corpus-builder's enrich_v2 phase writes), use it directly. Massive speedup:
    # builds skip every YouTube download. Env override: VM_CORPUS_CLIP_CACHE.
    corpus_cache_env = os.environ.get("VM_CORPUS_CLIP_CACHE", "")
    if corpus_cache_env:
        corpus_cache = Path(corpus_cache_env)
    else:
        sdb = os.environ.get("YTA_SHARED_DB", "")
        corpus_cache = (Path(sdb).parent / "clip_cache") if sdb else None
    if corpus_cache and not corpus_cache.is_dir():
        corpus_cache = None

    def materialize(moment, take, shot_seq):
        seg = moment["seg"]; url = moment["url"]; vid = moment.get("id") or "x"
        start = float(seg.get("start", 0)); end = float(seg.get("end", start))
        wkey = (vid, round(start, 2))
        raw = raw_dir / f"raw_{vid}_{start:.2f}.mp4"
        if wkey not in raw_status:
            ok = raw.exists() and raw.stat().st_size > 50_000
            # 1) prefer the corpus-wide cache (pre-downloaded by the corpus-builder)
            if not ok and corpus_cache is not None:
                ccp = corpus_cache / f"{vid}_{start:.2f}.mp4"
                if ccp.exists() and ccp.stat().st_size > 50_000:
                    try:
                        import shutil
                        shutil.copy(str(ccp), str(raw)); ok = True
                    except Exception:
                        pass
            # 2) fall back to YouTube download
            if not ok and not cached_only:
                ok = youtube.download_segment(url, start, end + PAD, raw)
            raw_status[wkey] = raw if (ok and raw.exists()) else None
        raw = raw_status[wkey]
        if raw is None:
            return None
        # face/text QC once per window (cached on disk, fail-closed): scan the CROPPED full
        # window = exactly the framing the final gate sees (the gate flags the cropped render,
        # which can show a face the uncropped raw doesn't). A window the corpus purge missed or
        # that trips the detector is unusable -> the assembler reaches for the next-best clip.
        if wkey not in face_status:
            qc_seg = seg_dir / f"qc_{vid}_{start:.2f}.mp4"
            qres = dl.fit_clip(raw, qc_seg, max(1.0, end - start), credit(moment))
            face_status[wkey] = (bool(qres.get("ok")) and qc_seg.exists()
                                 and not rnd.mostly_black(qc_seg) and _face_text_ok(qc_seg))
            try: qc_seg.unlink()
            except Exception: pass
            _save_qc()
        if not face_status[wkey]:
            return None
        seg_path = seg_dir / f"seg_{shot_seq:04d}.mp4"
        res = dl.fit_clip(raw, seg_path, float(take), credit(moment))
        if not res.get("ok") or not seg_path.exists() or rnd.mostly_black(seg_path):
            try: seg_path.unlink()
            except Exception: pass
            return None
        return (str(seg_path), res.get("out_dur") or take)

    return materialize


def _merge_parallel_decisions(run, worklist):
    """Merge any decisions_*.json from parallel pick agents into match_decisions.json,
    safely. If an agent (mis)used SLICE-LOCAL section indices instead of the global
    `index` field, we recover by looking up each pick's cand_id in the worklist —
    every cand_id is unique to its section's shortlist, so the picks identify the
    correct section unambiguously even with positional keys. Idempotent.
    """
    import glob
    decs = sorted(glob.glob(str(run / "decisions_*.json")))
    if not decs:
        return None
    cand2sec = {}
    for s in worklist:
        for c in s.get("candidates", []):
            cand2sec.setdefault(c["cand_id"], set()).add(s["index"])
    merged = {}; remapped = 0; ambiguous = 0
    for f in decs:
        try: d = json.load(open(f))
        except Exception: continue
        if isinstance(d, dict) and "decisions" in d:
            d = d["decisions"]
        for k, picks in (d.items() if isinstance(d, dict) else []):
            if not picks: continue
            if isinstance(picks, str): picks = [picks]
            # find the section index that ALL picks share (intersection)
            share = None
            for cid in picks:
                secs = cand2sec.get(cid, set())
                share = secs if share is None else (share & secs)
                if share is not None and not share: break
            if not share: ambiguous += 1; continue
            gidx = sorted(share)[0]
            if str(gidx) != str(k): remapped += 1
            merged.setdefault(str(gidx), [str(c) for c in picks])
    out_path = run / "match_decisions.json"
    json.dump(merged, open(out_path, "w"), indent=2)
    if remapped or ambiguous:
        print(f"[build] merged {len(decs)} decisions_*.json -> {len(merged)} sections "
              f"(positional remaps: {remapped}, ambiguous skipped: {ambiguous})")
    else:
        print(f"[build] merged {len(decs)} decisions_*.json -> {len(merged)} sections")
    return out_path


def cmd_build(args):
    from app.services.v3 import shared_library as lib
    from app.services.v3 import section_planner as sp
    from app.services.v3 import matcher
    import render as rnd
    run = BASE / args.topic
    wpath = run / "match_worklist.json"
    if not wpath.exists():
        print(f"[build] no worklist at {wpath}; run plan first"); sys.exit(2)
    worklist = matcher.load_worklist(wpath)
    # auto-merge parallel decisions_*.json if any exist (and match_decisions.json doesn't
    # already exist with the same effective content). Safe vs positional-key confusion.
    if not (run / "match_decisions.json").exists():
        _merge_parallel_decisions(run, worklist)
    elif list(run.glob("decisions_*.json")):
        _merge_parallel_decisions(run, worklist)
    dpath = Path(args.decisions) if args.decisions else (run / "match_decisions.json")
    decisions = matcher.load_decisions(dpath)
    print(f"[build] {len(worklist)} sections; {len(decisions)} have Claude decisions"
          f"{' (none -> offline order)' if not decisions else ''}")

    index = lib.load_index(topic=NICHE, progress_cb=print)
    materialize = None if args.dry_run else _make_materializer(run)
    planned = sp.plan_from_worklist(worklist, decisions, index,
                                    materialize=materialize, progress_cb=print)
    report = sp.assembly_report(planned)
    print(f"[build] assembly: {report}")
    assert report["max_uses_seen"] <= int(os.environ.get("VM_MAX_CLIP_USES", "2")), "cap violated"
    assert report["consecutive_violations"] == 0, "consecutive-repeat violated"

    shots = sp.to_shots(planned)
    (run / "shots.json").write_text(json.dumps(shots, indent=2))
    placed = [s for s in shots if s.get("final_clip_path")]
    print(f"[build] {len(shots)} shots planned, {len(placed)} materialized")
    if args.dry_run:
        print("[build] --dry-run: planned only, no download/render"); return

    out_dir = Path(os.environ.get("VIDEO_OUTPUT_DIR", run))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.topic}.mp4"
    summary = rnd.render(shots, run / "audio", out_path, run / "work", progress_cb=print)
    print(f"[build] {summary}")
    print(f"[build] NEXT: python validate_render.py {out_path}   # non-skippable safety gate")


# --------------------------------------------------------------------------- #
# selftest (offline): deps + assembly rules on a synthetic library            #
# --------------------------------------------------------------------------- #
def cmd_selftest():
    from app.services.v3 import section_planner as sp
    ok = True
    # 1. assembly rules: <=2 uses, never consecutive, concat-fill, no freeze
    lib_index = {f"https://y/{i}": {"id": f"v{i}", "channel": "", "title": "", "niche": "n",
                 "segments": [{"start": 0.0, "end": 3.0, "embed_text": f"clip {i}",
                               "transcript": "", "talking_head": False, "label": "pour_wax"}]}
                 for i in range(6)}
    worklist = []
    for k in range(4):
        cands = [{"cand_id": f"v{i}@0.0"} for i in range(6)]
        worklist.append({"index": k, "start": k * 8.0, "end": k * 8.0 + 8.0, "dur": 8.0,
                         "text": "pour the wax", "candidates": cands})
    planned = sp.plan_from_worklist(worklist, {}, lib_index)   # no materialize -> pure
    rep = sp.assembly_report(planned)
    print("  assembly_report:", rep)
    if rep["max_uses_seen"] > 2:
        print("  FAIL: a clip used > 2 times"); ok = False
    if rep["consecutive_violations"] != 0:
        print("  FAIL: consecutive repeat"); ok = False
    covered = all(p["covered"] >= p["dur"] - 0.6 for p in planned)
    print(f"  coverage ok: {covered}")
    ok = ok and covered
    # 2. deps present
    for mod in ("numpy", "edge_tts"):
        try:
            __import__(mod); print(f"  dep {mod}: ok")
        except Exception as e:
            print(f"  dep {mod}: MISSING ({e})")
    try:
        from app.services.v3 import embeddings as emb
        v = emb.embed_many(["pour the wax", "measuring fragrance"])
        print(f"  embedder: {emb.active_backend()} dim={len(v[0]) if v else 0}")
    except Exception as e:
        print(f"  embedder: FAIL {e}"); ok = False
    print("SELFTEST:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_review(args):
    """Emit N spot-check slices for parallel cut-review agents (a second-pass QC layer)."""
    run = BASE / args.topic
    mp4 = run / f"{args.topic}.mp4"
    if not mp4.exists():
        print(f"[review] no video at {mp4}; build first"); sys.exit(2)
    n_slices = max(1, int(getattr(args, "slices", 0) or os.environ.get("VM_REVIEW_SLICES", "4")))
    shots = json.loads((run / "shots.json").read_text())
    chunk = (len(shots) + n_slices - 1) // n_slices
    sl_dir = run / "review_slices"; sl_dir.mkdir(parents=True, exist_ok=True)
    for old in sl_dir.glob("slice_*.json"):
        old.unlink()
    for i in range(n_slices):
        ch = shots[i * chunk:(i + 1) * chunk]
        if not ch: continue
        json.dump({"_instructions": (
            "Spot-check the listed shots in the rendered video for issues a final-gate "
            "automated scan can miss: an off-topic clip the matcher picked, a clip whose "
            "content contradicts the narration, jarring back-to-back cuts, etc. For each "
            "shot, judge ok / warn / fail with a one-line reason. Write your output to "
            f"state/runs/{args.topic}/review_{i}.json as a list of "
            "{shot_idx,start_sec,verdict,reason}."),
            "video": str(mp4), "slice_index": i,
            "shots": [{"shot_idx": s["shot_idx"], "start_sec": s["start_sec"],
                       "duration": s["duration"],
                       "scene_description": s.get("scene_description", ""),
                       "candidate": {k: s["candidate"].get(k) for k in
                                     ("id", "url", "source_title", "match", "scene")}}
                      for s in ch]}, open(sl_dir / f"slice_{i:02d}.json", "w"))
    print(f"[review] wrote {n_slices} review slices under {sl_dir} ({chunk} shots each)")
    print(f"[review] NEXT — spawn {n_slices} agents in parallel, one per review slice, each")
    print(f"         spot-checking the video at the given timestamps. Aggregate review_*.json.")


def main():
    if "--selftest" in sys.argv:
        return cmd_selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["plan", "build", "review", "all"])
    ap.add_argument("--topic", required=True)
    ap.add_argument("--script")
    ap.add_argument("--title", default="")
    ap.add_argument("--decisions", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--slices", type=int, default=0,
                    help="N parallel agent slices for plan (default $VM_PICK_SLICES=4) or review ($VM_REVIEW_SLICES=4)")
    args = ap.parse_args()
    if args.phase in ("plan", "all"):
        if not args.script:
            print("plan needs --script"); sys.exit(1)
        cmd_plan(args)
    if args.phase in ("build", "all"):
        cmd_build(args)
    if args.phase == "review":
        cmd_review(args)


if __name__ == "__main__":
    main()

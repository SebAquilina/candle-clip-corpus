"""Unified corpus-revamp driver — one pass per video produces the new per-second corpus.

Per video (resumable, crash-safe, download-failure-tolerant):
  download once (≤720p) -> for each usable window:
     1. PURGE: second-by-second YuNet+OCR; any face/text -> reject whole window (pixel gate)
     2. DESCRIBE survivors second-by-second with a local VLM (scene-change deduped)
        + align the speaker's transcript per second (YouTube captions / whisper)
     3. STAGE 4b: if any second's caption implies a face/person -> reject the window too
  write a new per-second record (survivors only) + a quarantine file (rejects) -> discard dl.

Outputs (a NEW corpus, leaving the old one untouched until you bless it):
  outputs/shared_db_v2/records/<vid>.json    survivors, each with seconds[] + a window-level
                                              embed_text/transcript so the existing matcher
                                              can consume it as-is
  outputs/shared_db_v2/rejected/<vid>.json    every dropped window + reason (face|text|desc_face)
  outputs/shared_db_v2/by_label/<label>.jsonl one line per kept window (browse/match index)

Full grind (where YouTube works):     python reclassify.py --max-seconds 0
Validate on local clips (no network): python reclassify.py --local-clips <dir> [--limit N]
Reindex by_label from records:        python reclassify.py --reindex
Status:                               python reclassify.py --status
"""
from __future__ import annotations
import json, os, sys, time, glob, tempfile, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RECORDS = Path(os.environ.get("REVAMP_RECORDS", REPO / "outputs/shared_db/records"))
V2 = Path(os.environ.get("REVAMP_V2", REPO / "outputs/shared_db_v2"))
STATE_DIR = HERE / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
ATTEMPTS_FILE = STATE_DIR / "attempts.json"
REC_OUT = V2 / "records"; REJ_OUT = V2 / "rejected"; LBL_OUT = V2 / "by_label"
for d in (REC_OUT, REJ_OUT, LBL_OUT):
    d.mkdir(parents=True, exist_ok=True)

import detectors      # noqa: E402  (configures + loads the strong detector)
import describe as DESC  # noqa: E402


def usable_windows(rec: dict) -> list[dict]:
    return [w for w in rec.get("windows", []) if w.get("is_step", 0) != 0]


def _atomic(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _clear_attempt(vid: str) -> None:
    """Drop a video's attempt-marker once it produced a RESULT (record or rejected) — only
    a video that hung (no result at all) keeps its marker and is quarantined on resume."""
    try:
        a = json.load(open(ATTEMPTS_FILE))
    except Exception:
        return
    if a.pop(vid, None) is not None:
        _atomic(ATTEMPTS_FILE, a)


def _window_rollup(seconds: list[dict]) -> tuple[str, str]:
    """Window-level embed_text (unique vision phrases) + transcript (joined) so the
    existing matcher can use the v2 corpus without changes."""
    seen, descs = set(), []
    for s in seconds:
        d = (s.get("vision_desc") or "").strip()
        if d and d.lower() not in seen:
            seen.add(d.lower()); descs.append(d)
    # join per-second transcript, collapsing consecutive duplicate tokens (a word that
    # straddles a second boundary legitimately appears in both seconds' slices).
    toks = " ".join(s.get("transcript_text", "") for s in seconds).split()
    dedup = []
    for t in toks:
        if not dedup or dedup[-1].lower() != t.lower():
            dedup.append(t)
    return " | ".join(descs), " ".join(dedup)


def reclassify_video(rec: dict, video_path: str, words: list) -> tuple[dict, dict]:
    """Returns (v2_record, rejected_record) for one downloaded video."""
    vid = rec["video_id"]
    survivors, rejects = [], []
    for w in usable_windows(rec):
        base = {"window_index": w["window_index"], "start_s": w["start_s"],
                "end_s": w["end_s"], "action_label": w.get("action_label", ""),
                "phase": w.get("phase", "")}
        # 1. pixel purge
        v = detectors.scan_window(video_path, w["start_s"], w["end_s"])
        if not v["clean"]:
            rejects.append({**base, "reason": v["reason"], "hit_t": v["hit_t"]})
            continue
        # 2. describe survivors + transcript
        d = DESC.describe_window(video_path, w["start_s"], w["end_s"], words)
        # 3. stage 4b: caption implies a face -> drop whole window
        if d["desc_face"]:
            rejects.append({**base, "reason": "desc_face", "hit_t": None,
                            "detail": d["desc_face_terms"]})
            continue
        embed_text, transcript = _window_rollup(d["seconds"])
        survivors.append({**base, "seconds": d["seconds"],
                          "embed_text": embed_text, "transcript": transcript,
                          "old_description": w.get("description", "")})
    v2 = {
        "video_id": vid, "video_title": rec.get("video_title", ""),
        "video_url": rec.get("video_url", ""), "video_duration_s": rec.get("video_duration_s"),
        "niche": rec.get("niche", ""), "channel": rec.get("channel", ""),
        "schema": "per_second_v1", "reclassified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vlm": DESC.get_captioner().name, "detector": detectors.availability().get("face_backend"),
        "n_windows_in": len(usable_windows(rec)), "n_windows_kept": len(survivors),
        "n_windows_rejected": len(rejects), "windows": survivors,
    }
    rej = {"video_id": vid, "n_rejected": len(rejects), "rejected": rejects}
    return v2, rej


def _reindex_one(v2: dict) -> None:
    """Append each kept window to its by_label/<label>.jsonl (idempotent rewrite per video
    is overkill; we just append — reindex --rebuild clears + rebuilds if needed)."""
    for w in v2.get("windows", []):
        lbl = (w.get("action_label") or "unlabeled").replace("/", "_")
        line = {"video_id": v2["video_id"], "video_url": v2["video_url"],
                "window_index": w["window_index"], "start_s": w["start_s"], "end_s": w["end_s"],
                "embed_text": w["embed_text"], "transcript": w["transcript"]}
        with open(LBL_OUT / f"{lbl}.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")


def run_grind(max_videos=0, max_seconds=0):
    import fetch, transcript as TR
    a = detectors.availability()
    if not (a["face_available"] and a["text_available"]):
        sys.exit(f"BLOCKER: detectors not available: {a}")
    print(f"detector={a['face_backend']} vlm={DESC.get_captioner().name}")
    recs = sorted(glob.glob(str(RECORDS / "*.json")))
    t0, done = time.time(), 0
    for rf in recs:
        try:
            rec = json.load(open(rf))
        except Exception:
            continue
        vid = rec.get("video_id")
        if not vid or (REC_OUT / f"{vid}.json").exists() or (REJ_OUT / f"{vid}.json").exists():
            continue
        if not usable_windows(rec):
            _atomic(REC_OUT / f"{vid}.json", {"video_id": vid, "schema": "per_second_v1",
                    "n_windows_in": 0, "n_windows_kept": 0, "windows": []})
            continue
        # Stall guard: a video can hang INSIDE a C call (cv2/torch), which a Python signal
        # can't interrupt — so an external watchdog (grind_supervisor.sh) kills a stalled
        # grind, and this attempt-marker makes the resumed run SKIP the culprit instead of
        # re-hanging on it. We record an attempt BEFORE the heavy work; if we ever see a
        # video that was already attempted (and has no record), it hung -> quarantine it.
        attempts = {}
        try:
            attempts = json.load(open(ATTEMPTS_FILE))
        except Exception:
            pass
        if attempts.get(vid, 0) >= 1:
            _atomic(REJ_OUT / f"{vid}.json",
                    {"video_id": vid, "error": f"skipped after stall (attempts={attempts[vid]})", "rejected": []})
            print(f"[skip] {vid}: previously stalled, quarantined")
            continue
        attempts[vid] = attempts.get(vid, 0) + 1
        _atomic(ATTEMPTS_FILE, attempts)
        dl = fetch.download(vid, rec.get("video_url", ""))
        if not dl["ok"]:
            _atomic(REJ_OUT / f"{vid}.json", {"video_id": vid, "error": dl["err"], "rejected": []})
            _clear_attempt(vid); continue
        # capture the source channel/title from yt-dlp metadata for clip attribution
        if dl.get("channel") and not (rec.get("channel") or "").strip():
            rec["channel"] = dl["channel"]
        if dl.get("title") and not (rec.get("video_title") or "").strip():
            rec["video_title"] = dl["title"]
        words = TR.get_transcript(vid, dl["path"], rec.get("video_url", ""))
        v2, rej = reclassify_video(rec, dl["path"], words)
        _atomic(REC_OUT / f"{vid}.json", v2)
        _atomic(REJ_OUT / f"{vid}.json", rej)
        _clear_attempt(vid)
        _reindex_one(v2)
        fetch.discard(vid)
        done += 1
        print(f"[{done}] {vid}: kept {v2['n_windows_kept']}/{v2['n_windows_in']} windows "
              f"({rej['n_rejected']} dropped)")
        if max_videos and done >= max_videos:
            break
        if max_seconds and time.time() - t0 >= max_seconds:
            print(f"time budget reached; exiting cleanly (resume next call)"); break
    print(f"done this pass: {done}")


def status():
    recs = glob.glob(str(REC_OUT / "*.json")); rejs = glob.glob(str(REJ_OUT / "*.json"))
    kept = win_in = dropped = errs = 0; reasons = {}
    for f in recs:
        d = json.load(open(f)); kept += d.get("n_windows_kept", 0); win_in += d.get("n_windows_in", 0)
    for f in rejs:
        d = json.load(open(f))
        if d.get("error"):
            errs += 1
        for r in d.get("rejected", []):
            dropped += 1; reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    total = len(glob.glob(str(RECORDS / "*.json")))
    print(f"videos reclassified: {len(recs)}/{total}  (download-errors: {errs})")
    print(f"windows kept: {kept}/{win_in}   dropped: {dropped}   reasons: {reasons}")


def local_clips(clip_dir: str, limit=0):
    """Validate the FULL pipeline on existing clips (each clip = one window). No network."""
    clips = sorted(glob.glob(os.path.join(clip_dir, "*.mp4")))
    if limit:
        clips = clips[:limit]
    print(f"detector={detectors.availability()['face_backend']} vlm={DESC.get_captioner().name}")
    print(f"reclassifying {len(clips)} local clips (no transcript)\n")
    kept = []; rej = []
    for c in clips:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", c]).decode().strip())
        v = detectors.scan_window(c, 0.0, dur)
        if not v["clean"]:
            rej.append((os.path.basename(c), v["reason"]));
            print(f"  REJECT[{v['reason']}@{v['hit_t']}s]  {os.path.basename(c)}"); continue
        d = DESC.describe_window(c, 0.0, dur, [])
        if d["desc_face"]:
            rej.append((os.path.basename(c), "desc_face"))
            print(f"  REJECT[desc_face:{d['desc_face_terms']}]  {os.path.basename(c)}"); continue
        et, _ = _window_rollup(d["seconds"])
        kept.append((os.path.basename(c), et))
        print(f"  KEEP  {os.path.basename(c):14s} :: {et[:70]}")
    print(f"\nSUMMARY: kept {len(kept)}/{len(clips)}, rejected {len(rej)}  {dict((r,sum(1 for _,x in rej if x==r)) for _,r in rej)}")


# keyword -> action_label for NEW videos (which have no upstream classification labels)
_LABEL_KW = [
    ("measure_wax", ["scale", "weigh", "weighing", "grams", "ounces"]),
    ("melt_wax", ["melt", "melting", "double boiler", "heating wax", "pouring pitcher", "hot wax"]),
    ("add_fragrance", ["fragrance", "scent", "essential oil", "perfume", "adding oil"]),
    ("add_dye_color", ["dye", "colour", "coloring", "colouring", "pigment", "tint", "colored wax"]),
    ("set_wick", ["wick bar", "centering", "wick sticker", "securing wick", "placing wick"]),
    ("trim_wick", ["trim", "scissors", "wick trimmer", "cutting wick"]),
    ("pour_wax", ["pour", "pouring", "pours", "filling"]),
    ("prepare_container", ["jar", "container", "tin", "vessel", "glass jar", "empty jar"]),
    ("cure_cool", ["cooling", "cure", "curing", "setting up", "hardening", "left to set"]),
    ("decorate_finish", ["decorate", "flower", "rose", "label", "ribbon", "dried", "topping", "garnish"]),
    ("reveal_result", ["lit", "burning", "flame", "glowing", "finished candle", "display", "shown"]),
    ("gather_materials", ["materials", "supplies", "ingredients", "tools", "kit", "laid out"]),
]


def _infer_label(text: str) -> str:
    t = (text or "").lower()
    for label, kws in _LABEL_KW:
        if any(k in t for k in kws):
            return label
    return ""


def ingest_list(ids, niche="candle_making", source="watch-later", window_s=8.0):
    """Put BRAND-NEW videos (not in the old corpus) through the v3 pipeline. They have no
    upstream window segmentation, so we auto-segment [0,dur] into ~window_s windows, run the
    exact same purge+describe+4b as the grind, then infer an action_label from the vision
    text. Resumable + attempt-marker stall guard, same as run_grind."""
    import fetch, transcript as TR
    a = detectors.availability()
    if not (a["face_available"] and a["text_available"]):
        sys.exit(f"BLOCKER: detectors unavailable: {a}")
    print(f"ingest: {len(ids)} videos | detector={a['face_backend']} vlm={DESC.get_captioner().name}")
    done = 0
    for vid in ids:
        if (REC_OUT / f"{vid}.json").exists():
            print(f"[have] {vid} already in corpus"); continue
        try:
            attempts = json.load(open(ATTEMPTS_FILE))
        except Exception:
            attempts = {}
        if attempts.get(vid, 0) >= 1 and not (REJ_OUT / f"{vid}.json").exists():
            _atomic(REJ_OUT / f"{vid}.json", {"video_id": vid, "error": "skipped after stall", "rejected": []})
            print(f"[skip] {vid}: previously stalled"); continue
        attempts[vid] = attempts.get(vid, 0) + 1
        _atomic(ATTEMPTS_FILE, attempts)
        url = f"https://www.youtube.com/watch?v={vid}"
        dl = fetch.download(vid, url)
        if not dl["ok"]:
            _atomic(REJ_OUT / f"{vid}.json", {"video_id": vid, "error": dl["err"], "rejected": []})
            _clear_attempt(vid); print(f"[dlfail] {vid}: {dl['err'][:70]}"); continue
        try:
            dur = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", dl["path"]]).decode().strip())
        except Exception:
            dur = 0.0
        if dur <= 4:
            _atomic(REJ_OUT / f"{vid}.json", {"video_id": vid, "error": "no/short duration", "rejected": []})
            fetch.discard(vid); continue
        wins, i, t = [], 0, 0.0
        while t + 4.0 <= dur:
            wins.append({"window_index": i, "start_s": round(t, 2),
                         "end_s": round(min(t + window_s, dur), 2), "is_step": 1, "action_label": ""})
            t += window_s; i += 1
        rec = {"video_id": vid, "video_url": url, "video_duration_s": dur,
               "niche": niche, "channel": dl.get("channel", ""),
               "video_title": dl.get("title", ""), "windows": wins}
        words = TR.get_transcript(vid, dl["path"], url)
        v2, rej = reclassify_video(rec, dl["path"], words)
        for w in v2["windows"]:
            w["action_label"] = _infer_label(w.get("embed_text", ""))
        v2["source"] = source
        _atomic(REC_OUT / f"{vid}.json", v2)
        _atomic(REJ_OUT / f"{vid}.json", rej)
        _clear_attempt(vid)
        _reindex_one(v2)
        fetch.discard(vid)
        done += 1
        print(f"[{done}] {vid}: kept {v2['n_windows_kept']}/{v2['n_windows_in']} ({rej['n_rejected']} dropped)")
    print(f"ingested {done} new videos")


def selftest():
    """Preflight: confirm this machine is ready BEFORE committing to the full grind —
    strong face detector, OCR, local VLM, and a real test download with the cookies."""
    ok = True
    import detectors as D
    a = D.availability()
    print("== corpus-revamp preflight ==")
    print(f"  face detector : {a['face_backend']}  ({'OK' if a['face_available'] else 'MISSING'})")
    print(f"  OCR (tesseract): {'OK' if a['text_available'] else 'MISSING'}")
    if a["face_backend"] != "yunet":
        print("  !! weak face backend (want yunet) — check assets/face_detection_yunet_2023mar.onnx"); ok = False
    if not (a["face_available"] and a["text_available"]):
        ok = False
    try:
        cap = DESC.get_captioner(); print(f"  VLM           : {cap.name}  ({'OK' if cap.name!='stub' else 'STUB — install torch+transformers'})")
        if cap.name == "stub":
            ok = False
    except Exception as e:
        print(f"  VLM           : FAIL ({str(e)[:80]})"); ok = False
    # test download (shortest record), with whatever cookies are configured
    try:
        import fetch, glob as _g, json as _j
        rows = []
        for rf in _g.glob(str(RECORDS / "*.json")):
            try:
                d = _j.load(open(rf))
                if d.get("video_id"):
                    rows.append((d.get("video_duration_s", 1e9) or 1e9, d["video_id"], d.get("video_url", "")))
            except Exception:
                pass
        rows.sort()
        if rows:
            _, vid, url = rows[0]
            print(f"  test download : {vid} ...")
            dl = fetch.download(vid, url)
            if dl["ok"]:
                print(f"                  OK ({dl.get('height')}p)"); fetch.discard(vid)
            else:
                print(f"                  FAIL: {dl['err'][:140]}")
                print("                  -> need: yt-dlp-ejs installed, deno on PATH, FRESH full-auth cookies (REVAMP_COOKIES)")
                ok = False
    except Exception as e:
        print(f"  test download : ERROR {str(e)[:80]}"); ok = False
    print("== PREFLIGHT", "PASS — ready to grind ==" if ok else "FAIL — fix the above first ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--status" in sys.argv:
        status()
    elif "--reindex" in sys.argv:
        import shutil
        if LBL_OUT.exists():
            shutil.rmtree(LBL_OUT); LBL_OUT.mkdir(parents=True)
        for f in glob.glob(str(REC_OUT / "*.json")):
            _reindex_one(json.load(open(f)))
        print("reindexed by_label/")
    elif "--local-clips" in sys.argv:
        i = sys.argv.index("--local-clips"); d = sys.argv[i + 1]
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
        local_clips(d, lim)
    elif "--ingest" in sys.argv:
        ids = json.load(open(sys.argv[sys.argv.index("--ingest") + 1]))
        ingest_list(ids)
    else:
        mv = int(sys.argv[sys.argv.index("--max-videos") + 1]) if "--max-videos" in sys.argv else 0
        ms = int(sys.argv[sys.argv.index("--max-seconds") + 1]) if "--max-seconds" in sys.argv else 0
        run_grind(mv, ms)

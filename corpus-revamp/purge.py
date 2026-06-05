"""Stage 1-3: per-second, zero-tolerance face+OCR PURGE of the corpus.

For each video: download once (≤720p, scratch), scan every usable window second-by-second
with the strong detector (YuNet+SSD+Haar faces, Tesseract OCR). A window is REJECTED whole
the instant any sampled second shows a face OR confident text (Rule: drop-the-whole-window,
zero-tolerance, fail-closed). Survivors pass to the describe stage.

Resumable + crash-safe: one result file per video (atomic rename); a finished video is
skipped on re-run. Download failures are recorded, not fatal, so the grind continues.

Run (full grind, where YouTube works):   python purge.py
Validate on local clips (no network):    python purge.py --validate-local <dir>
Status:                                   python purge.py --status
"""
from __future__ import annotations
import json, os, sys, time, glob, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RECORDS = Path(os.environ.get("REVAMP_RECORDS", REPO / "outputs/shared_db/records"))
STATE = Path(os.environ.get("REVAMP_STATE", HERE / "state"))
PURGE_OUT = STATE / "purge"
PURGE_OUT.mkdir(parents=True, exist_ok=True)

import detectors  # configures + loads the vendored detector


def usable_windows(rec: dict) -> list[dict]:
    """The matchable set = windows with is_step != 0 (flagged ones already excluded)."""
    return [w for w in rec.get("windows", []) if w.get("is_step", 0) != 0]


def _atomic_write(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def purge_video(video_id: str, windows: list[dict], video_path: str) -> dict:
    """Scan every usable window of one already-downloaded video. Returns a result dict."""
    survivors, rejects = [], []
    for w in windows:
        v = detectors.scan_window(video_path, w["start_s"], w["end_s"])
        entry = {"window_index": w["window_index"], "start_s": w["start_s"],
                 "end_s": w["end_s"], "action_label": w.get("action_label", ""),
                 **v}
        if v["clean"]:
            survivors.append(entry)
        else:
            rejects.append(entry)
    return {
        "video_id": video_id, "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_windows": len(windows), "n_survivors": len(survivors), "n_rejects": len(rejects),
        "reject_reasons": _count([r["reason"] for r in rejects]),
        "survivors": survivors, "rejects": rejects,
    }


def _count(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def run_grind(max_videos=0, max_seconds=0):
    """Full grind: download each video, purge, discard. Resumable. Needs working YouTube."""
    import fetch
    avail = detectors.availability()
    if not (avail["face_available"] and avail["text_available"] and avail["face_backend"] == "yunet"):
        sys.exit(f"BLOCKER: detector not strong enough: {avail}")
    recs = sorted(glob.glob(str(RECORDS / "*.json")))
    t0 = time.time(); done = 0
    for rf in recs:
        try:
            rec = json.load(open(rf))
        except Exception:
            continue
        vid = rec.get("video_id")
        if not vid:
            continue
        out = PURGE_OUT / f"{vid}.json"
        if out.exists():
            continue
        wins = usable_windows(rec)
        if not wins:
            _atomic_write(out, {"video_id": vid, "n_windows": 0, "n_survivors": 0,
                                "n_rejects": 0, "survivors": [], "rejects": [], "note": "no usable windows"})
            continue
        dl = fetch.download(vid, rec.get("video_url", ""))
        if not dl["ok"]:
            _atomic_write(out, {"video_id": vid, "error": dl["err"], "n_windows": len(wins),
                                "n_survivors": 0, "n_rejects": 0, "survivors": [], "rejects": []})
            continue
        res = purge_video(vid, wins, dl["path"])
        _atomic_write(out, res)
        fetch.discard(vid)
        done += 1
        print(f"[{done}] {vid}: {res['n_survivors']}/{res['n_windows']} survive  {res['reject_reasons']}")
        if max_videos and done >= max_videos:
            break
        if max_seconds and (time.time() - t0) >= max_seconds:
            print(f"time budget {max_seconds}s reached; exiting cleanly (resume next call)")
            break
    print(f"done this pass: {done} videos")


def status():
    files = glob.glob(str(PURGE_OUT / "*.json"))
    tot_w = tot_s = tot_r = 0; errs = 0; reasons = {}
    for f in files:
        d = json.load(open(f))
        if d.get("error"):
            errs += 1; continue
        tot_w += d.get("n_windows", 0); tot_s += d.get("n_survivors", 0); tot_r += d.get("n_rejects", 0)
        for k, v in d.get("reject_reasons", {}).items():
            reasons[k] = reasons.get(k, 0) + v
    total_recs = len(glob.glob(str(RECORDS / "*.json")))
    print(f"videos purged: {len(files)}/{total_recs}  (download-errors: {errs})")
    print(f"windows: {tot_w}  survive: {tot_s}  reject: {tot_r}  "
          f"({(100.0*tot_r/tot_w if tot_w else 0):.1f}% rejected)")
    print(f"reject reasons: {reasons}")


def validate_local(clip_dir: str, limit=0):
    """Run the detector on existing local clips (each clip = one window [0,dur]). No network."""
    import subprocess
    clips = sorted(glob.glob(os.path.join(clip_dir, "*.mp4")))
    if limit:
        clips = clips[:limit]
    print(f"detector: {detectors.availability()}")
    print(f"scanning {len(clips)} local clips from {clip_dir}\n")
    res = []
    for c in clips:
        try:
            dur = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", c]).decode().strip())
        except Exception:
            dur = 8.0
        v = detectors.scan_window(c, 0.0, dur)
        res.append((os.path.basename(c), dur, v))
        flag = "CLEAN " if v["clean"] else f"REJECT[{v['reason']}@{v['hit_t']}s]"
        print(f"  {flag:24s} {os.path.basename(c):42s} dur={dur:4.1f}s frames={v['frames_read']}")
    n_rej = sum(1 for _, _, v in res if not v["clean"])
    print(f"\nSUMMARY: {n_rej}/{len(res)} clips rejected "
          f"(face={sum(1 for _,_,v in res if v['reason']=='face')}, "
          f"text={sum(1 for _,_,v in res if v['reason']=='text')})")
    return res


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--validate-local" in sys.argv:
        i = sys.argv.index("--validate-local")
        d = sys.argv[i + 1]
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
        validate_local(d, lim)
    else:
        mv = int(sys.argv[sys.argv.index("--max-videos") + 1]) if "--max-videos" in sys.argv else 0
        ms = int(sys.argv[sys.argv.index("--max-seconds") + 1]) if "--max-seconds" in sys.argv else 0
        run_grind(mv, ms)

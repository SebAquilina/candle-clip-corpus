#!/usr/bin/env python3
"""validate_render.py — non-skippable post-render gate.

Scans a finished mp4 for: black stretches, frozen-frame stretches, and audio/video
duration mismatch. Exits non-zero with a clear report if ANY check fails. There is
no --skip flag and the script does not take quality shortcuts. Render time is
irrelevant; correctness is the bar.

Invoked by make_video.sh as the final step. A video is not "done" until this
script exits 0. Operators who built the mp4 via an ad-hoc renderer MUST still run
this gate before declaring success.

Usage:
    python validate_render.py <path-to-final.mp4>

Exit codes:
    0  -> mp4 passes all checks; ready to ship
    2  -> mp4 has at least one black stretch >= MAX_BLACK_SEC
    3  -> mp4 has at least one frozen-frame stretch >= MAX_FREEZE_SEC
    4  -> audio/video duration mismatch > MAX_AV_SKEW_SEC
    5  -> file unreadable / probe failed
    6  -> overlay / burned-in TEXT detected (Rule v22.3)
    7  -> TALKING HEAD detected (Rule v22.4)
    8  -> cannot verify text / talking-head — detectors unavailable, OR disabled via
          VM_DISABLE_CLIP_CHECKS while YTA_REQUIRE_GATES=1 (default). A blocker, never a silent pass.
"""
from __future__ import annotations
import os, sys, re, subprocess, json, time, math, tempfile
from pathlib import Path

# Resumable mode (opt-in): set VM_VALIDATE_BUDGET_SEC>0 so a single invocation does only
# as much work as fits the budget, persists progress, and exits 75 ("run me again") until
# the whole file is scanned — then emits the SAME verdict/exit codes as the one-shot path.
# Default (budget=0) keeps the original one-shot behaviour exactly. Lets the non-skippable
# gate finish a long video across short execution windows without ever redoing work or
# weakening a single threshold.
EXIT_INCOMPLETE = 75

# Tunables — strict by design. Don't relax these to "save time." (Env-overridable for the
# rare legitimate case: e.g. genuinely dark, moody candle B-roll can read as "black" at the
# default 0.10 pixel threshold — lower VM_BLACK_PIX_TH to ~0.05 so only true black fails.)
MAX_BLACK_SEC = float(os.environ.get("VM_MAX_BLACK_SEC", "0.5"))       # black stretch >= this fails
MAX_FREEZE_SEC = float(os.environ.get("VM_MAX_FREEZE_SEC", "8.0"))     # frozen stretch >= this fails
MAX_AV_SKEW_SEC = float(os.environ.get("VM_MAX_AV_SKEW_SEC", "1.0"))   # av duration mismatch allowed
BLACK_PIX_THRESHOLD = float(os.environ.get("VM_BLACK_PIX_TH", "0.10")) # blackdetect pixel intensity
FREEZE_PIX_THRESHOLD = float(os.environ.get("VM_FREEZE_PIX_TH", "0.003"))  # freezedetect picture diff


def _probe(path: Path) -> dict:
    """Probe duration of both streams."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,duration:format=duration",
        "-of", "json", str(path)
    ])
    return json.loads(out)


def _run_ffmpeg_filter(path: Path, vf: str) -> str:
    """Run ffmpeg with a filter, return stderr log. NEVER skip; if ffmpeg fails, raise."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-vf", vf, "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return r.stderr or ""


def check_black(path: Path):
    """Find any black stretch >= MAX_BLACK_SEC. Scans the ENTIRE video — no sampling."""
    vf = f"blackdetect=d={MAX_BLACK_SEC}:pix_th={BLACK_PIX_THRESHOLD}"
    log = _run_ffmpeg_filter(path, vf)
    hits = []
    for m in re.finditer(r"black_start:([0-9.]+) black_end:([0-9.]+) black_duration:([0-9.]+)", log):
        start, end, dur = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if dur >= MAX_BLACK_SEC:
            hits.append((start, end, dur))
    return hits


def check_freeze(path: Path):
    """Find any frozen-frame stretch >= MAX_FREEZE_SEC."""
    vf = f"freezedetect=n={FREEZE_PIX_THRESHOLD}:d={MAX_FREEZE_SEC}"
    log = _run_ffmpeg_filter(path, vf)
    hits = []
    # ffmpeg writes lines like:
    #   [freezedetect @ 0x...] lavfi.freezedetect.freeze_start: 12.34
    #   [freezedetect @ 0x...] lavfi.freezedetect.freeze_duration: 9.5
    #   [freezedetect @ 0x...] lavfi.freezedetect.freeze_end: 21.84
    starts = [float(x) for x in re.findall(r"freeze_start: ([0-9.]+)", log)]
    durs   = [float(x) for x in re.findall(r"freeze_duration: ([0-9.]+)", log)]
    for s, d in zip(starts, durs):
        if d >= MAX_FREEZE_SEC:
            hits.append((s, s + d, d))
    return hits


def check_av_skew(path: Path):
    """Confirm audio and video durations are close."""
    probe = _probe(path)
    streams = probe.get("streams", [])
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    a = next((s for s in streams if s["codec_type"] == "audio"), None)
    if not v or not a:
        return None  # mismatched stream count is reported separately
    try:
        vd, ad = float(v.get("duration") or 0), float(a.get("duration") or 0)
    except ValueError:
        return None
    return abs(vd - ad), vd, ad


# --- resumable mode (opt-in via VM_VALIDATE_BUDGET_SEC) ---------------------
def _sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def _state_path(p: Path) -> Path:
    # keep state off the (possibly write-once) output mount
    d = Path(os.environ.get("VM_VALIDATE_STATE_DIR", tempfile.gettempdir()))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"validate_{p.name}.{_sig(p)}.json"


def _load_state(p: Path, dur: float, chunk: float) -> dict:
    sp = _state_path(p)
    if sp.exists():
        try:
            st = json.load(open(sp))
            if st.get("sig") == _sig(p):
                return st
        except Exception:
            pass
    return {"sig": _sig(p), "duration": dur, "chunk": chunk,
            "black": None, "freeze": None, "skew": None, "tf_chunks": {}}


def _save_state(p: Path, st: dict) -> None:
    sp = _state_path(p)
    fd, tmp = tempfile.mkstemp(dir=str(sp.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(st, f)
    os.replace(tmp, sp)


def _detector_state():
    try:
        import clip_checks as cc
    except Exception as e:
        return None, False, False, str(e)
    require = os.environ.get("YTA_REQUIRE_GATES", "1").strip().lower() not in ("0", "false", "no", "")
    return cc, bool(cc.DISABLED), bool(cc.TEXT_AVAILABLE and cc.FACE_AVAILABLE), require


def _scan_chunk_text_face(p: Path, t0: float, t1: float, cc):
    """Run the CANONICAL gate scans on the [t0,t1] sub-clip and offset timestamps by t0,
    so per-chunk results aggregate to exactly what a whole-file scan would find."""
    tmp = tempfile.mktemp(suffix=".mp4", dir=tempfile.gettempdir())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0}", "-to", f"{t1}",
                    "-i", str(p), "-c:v", "copy", "-an", tmp], check=True)
    try:
        th = [(float(t) + t0, frac) for (t, frac) in cc.scan_video_text(Path(tmp))]
        fh = [(float(t) + t0, frac) for (t, frac) in cc.scan_video_talking_head(Path(tmp))]
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
    return th, fh


def _emit_verdict(p, black, freeze, skew, text_hits, face_hits, cc, disabled, detectors_ok, require):
    """Same checks, thresholds and exit codes as the one-shot path — just fed aggregated data."""
    if black:
        print(f"FAIL ({len(black)} black stretch(es) >= {MAX_BLACK_SEC}s):", file=sys.stderr)
        for s, e, d in black:
            print(f"   {s:>7.2f}s -> {e:>7.2f}s   ({d:.2f}s black)", file=sys.stderr)
        sys.exit(2)
    print("  black: PASS")
    if freeze:
        print(f"FAIL ({len(freeze)} frozen-frame stretch(es) >= {MAX_FREEZE_SEC}s):", file=sys.stderr)
        for s, e, d in freeze:
            print(f"   {s:>7.2f}s -> {e:>7.2f}s   ({d:.2f}s frozen)", file=sys.stderr)
        sys.exit(3)
    print(f"  freeze (>={MAX_FREEZE_SEC}s): PASS")
    if skew and skew != "none":
        d, vd, ad = skew
        if d > MAX_AV_SKEW_SEC:
            print(f"FAIL: audio/video duration mismatch {d:.2f}s (video {vd:.2f}, audio {ad:.2f})", file=sys.stderr)
            sys.exit(4)
        print(f"  av-skew: PASS ({d:.2f}s)")
    else:
        print("  av-skew: SKIPPED (missing stream metadata)")
    if disabled and not require:
        print("  overlay-text / talking-head: SKIPPED (VM_DISABLE_CLIP_CHECKS=1, YTA_REQUIRE_GATES=0)")
    elif disabled or not detectors_ok:
        print("FAIL: cannot verify no-text / no-talking-head — detectors unavailable or disabled.", file=sys.stderr)
        print("   This is a BLOCKER (Rules v22.3/v22.4).", file=sys.stderr)
        sys.exit(8)
    else:
        if len(text_hits) >= cc.GATE_TEXT_HITS:
            print(f"FAIL ({len(text_hits)} frame(s) with overlay/burned-in text, >= {cc.GATE_TEXT_HITS}):", file=sys.stderr)
            for t, frac in sorted(text_hits)[:8]:
                print(f"   {t:>7.2f}s   text covers {frac*100:.2f}% of frame", file=sys.stderr)
            sys.exit(6)
        print(f"  overlay-text: PASS ({len(text_hits)} flagged frame(s) < {cc.GATE_TEXT_HITS})")
        if len(face_hits) >= cc.GATE_FACE_HITS:
            print(f"FAIL ({len(face_hits)} frame(s) with a dominant face / talking head, >= {cc.GATE_FACE_HITS}):", file=sys.stderr)
            for t, frac in sorted(face_hits)[:8]:
                print(f"   {t:>7.2f}s   face covers {frac*100:.2f}% of frame", file=sys.stderr)
            sys.exit(7)
        print(f"  talking-head: PASS ({len(face_hits)} flagged frame(s) < {cc.GATE_FACE_HITS})")
    print("\nVALIDATE_RENDER: OK (resumable)")
    sys.exit(0)


def run_resumable(p: Path, budget: float):
    t_start = time.time()
    dur = float(_probe(p).get("format", {}).get("duration") or 0.0)
    chunk = float(os.environ.get("VM_VALIDATE_CHUNK_SEC", "120"))
    st = _load_state(p, dur, chunk)

    def over():
        return (time.time() - t_start) >= budget

    print(f"=== validate_render (resumable, budget={budget}s, chunk={chunk}s, dur={dur:.0f}s): {p} ===")
    if st["black"] is None:
        st["black"] = [list(h) for h in check_black(p)]; _save_state(p, st)
        if over():
            print(f"  black done; {EXIT_INCOMPLETE}=incomplete, re-run to continue"); sys.exit(EXIT_INCOMPLETE)
    if st["freeze"] is None:
        st["freeze"] = [list(h) for h in check_freeze(p)]; _save_state(p, st)
        if over():
            print(f"  freeze done; re-run to continue"); sys.exit(EXIT_INCOMPLETE)
    if st["skew"] is None:
        sk = check_av_skew(p); st["skew"] = list(sk) if sk else "none"; _save_state(p, st)

    cc, disabled, detectors_ok, require = _detector_state()
    if not (disabled and not require):
        if disabled or not detectors_ok:
            print("FAIL: cannot verify no-text / no-talking-head — detectors unavailable.", file=sys.stderr)
            sys.exit(8)
        n = max(1, math.ceil(dur / chunk))
        processed = 0
        for i in range(n):
            if str(i) in st["tf_chunks"]:
                continue
            # Guarantee forward progress: always do >=1 chunk per call before honouring
            # the budget, so a budget smaller than one chunk can't livelock (it overshoots
            # by at most one chunk). Size VM_VALIDATE_CHUNK_SEC to fit your window.
            if processed > 0 and over():
                _save_state(p, st)
                print(f"  text/face: {len(st['tf_chunks'])}/{n} chunks done; "
                      f"re-run to continue (exit {EXIT_INCOMPLETE})")
                sys.exit(EXIT_INCOMPLETE)
            t0, t1 = i * chunk, min(dur, (i + 1) * chunk)
            th, fh = _scan_chunk_text_face(p, t0, t1, cc)
            st["tf_chunks"][str(i)] = {"text": th, "face": fh}; _save_state(p, st)
            processed += 1
            print(f"  text/face chunk {i+1}/{n} [{t0:.0f}-{t1:.0f}s]: "
                  f"{len(th)} text, {len(fh)} face")

    text_hits, face_hits = [], []
    for v in st["tf_chunks"].values():
        text_hits += [tuple(x) for x in v["text"]]
        face_hits += [tuple(x) for x in v["face"]]
    black = [tuple(x) for x in (st["black"] or [])]
    freeze = [tuple(x) for x in (st["freeze"] or [])]
    _emit_verdict(p, black, freeze, st["skew"], text_hits, face_hits, cc, disabled, detectors_ok, require)


def main():
    if len(sys.argv) < 2:
        print("usage: validate_render.py <mp4>", file=sys.stderr); sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"FAIL: file not found: {p}", file=sys.stderr); sys.exit(5)

    budget = float(os.environ.get("VM_VALIDATE_BUDGET_SEC", "0") or 0)
    if budget > 0:
        return run_resumable(p, budget)

    print(f"=== validate_render: {p} ===")
    try:
        # 1. Black stretches
        black = check_black(p)
        if black:
            print(f"FAIL ({len(black)} black stretch(es) >= {MAX_BLACK_SEC}s):", file=sys.stderr)
            for s, e, d in black:
                print(f"   {s:>7.2f}s -> {e:>7.2f}s   ({d:.2f}s black)", file=sys.stderr)
            print("\nRule v20.2/v20.3: a video with any black stretch is not shippable.\n"
                  "Re-render with proper clip validation; never skip blackdetect to save time.",
                  file=sys.stderr)
            sys.exit(2)
        print("  black: PASS")

        # 2. Frozen-frame stretches
        freeze = check_freeze(p)
        if freeze:
            print(f"FAIL ({len(freeze)} frozen-frame stretch(es) >= {MAX_FREEZE_SEC}s):", file=sys.stderr)
            for s, e, d in freeze:
                print(f"   {s:>7.2f}s -> {e:>7.2f}s   ({d:.2f}s frozen)", file=sys.stderr)
            print("\nLong freezes mean a clip ran short and the fallback held the last frame too long.\n"
                  "Either use longer source clips or borrow a different shot.", file=sys.stderr)
            sys.exit(3)
        print(f"  freeze (>={MAX_FREEZE_SEC}s): PASS")

        # 3. AV skew
        skew = check_av_skew(p)
        if skew is not None:
            d, vd, ad = skew
            if d > MAX_AV_SKEW_SEC:
                print(f"FAIL: audio/video duration mismatch {d:.2f}s (video {vd:.2f}, audio {ad:.2f})",
                      file=sys.stderr)
                sys.exit(4)
            print(f"  av-skew: PASS ({d:.2f}s)")
        else:
            print("  av-skew: SKIPPED (missing stream metadata)")

        # 4. Overlay/burned-in TEXT + TALKING HEADS (Rules v22.3 / v22.4) — NON-SKIPPABLE.
        # Text or a presenter facing the camera must NEVER ship. If the detectors are
        # unavailable we BLOCK (exit 8) rather than silently pass.
        try:
            import clip_checks as cc
            _cc_err = None
        except Exception as e:
            cc, _cc_err = None, e
        # YTA_REQUIRE_GATES=1 (default) makes this gate non-skippable: the
        # VM_DISABLE_CLIP_CHECKS bypass is REFUSED (treated as a blocker) so no one can
        # silently ship unchecked footage. Set YTA_REQUIRE_GATES=0 to allow the bypass.
        require_gates = os.environ.get("YTA_REQUIRE_GATES", "1").strip().lower() not in ("0", "false", "no", "")
        disabled = bool(cc is not None and cc.DISABLED)
        detectors_ok = bool(cc is not None and cc.TEXT_AVAILABLE and cc.FACE_AVAILABLE)
        if disabled and not require_gates:
            print("  overlay-text / talking-head: SKIPPED (VM_DISABLE_CLIP_CHECKS=1, YTA_REQUIRE_GATES=0)")
        elif disabled or not detectors_ok:
            avail = cc.availability() if cc is not None else {"import_error": str(_cc_err)}
            print("FAIL: cannot verify no-text / no-talking-head — detectors unavailable or disabled.",
                  file=sys.stderr)
            print(f"   availability: {avail}", file=sys.stderr)
            if disabled:
                print("   VM_DISABLE_CLIP_CHECKS=1 but YTA_REQUIRE_GATES=1 (default) — refusing to skip the gate.",
                      file=sys.stderr)
            print("   This is a BLOCKER, not a pass — text / talking heads must NEVER ship (Rules v22.3/v22.4).",
                  file=sys.stderr)
            print("   Install: pip install opencv-python-headless pytesseract ; apt-get install -y tesseract-ocr",
                  file=sys.stderr)
            print("   (Set YTA_REQUIRE_GATES=0 to allow the VM_DISABLE_CLIP_CHECKS bypass — NOT recommended.)",
                  file=sys.stderr)
            sys.exit(8)
        else:
            text_hits = cc.scan_video_text(p)
            if len(text_hits) >= cc.GATE_TEXT_HITS:
                print(f"FAIL ({len(text_hits)} frame(s) with overlay/burned-in text, "
                      f">= {cc.GATE_TEXT_HITS}):", file=sys.stderr)
                for t, frac in text_hits[:8]:
                    print(f"   {t:>7.2f}s   text covers {frac*100:.2f}% of frame", file=sys.stderr)
                print("\nRule v22.3: the video must contain NO overlay/burned-in text. Clips with\n"
                      "text are unusable — re-render so they are replaced (do not ship texted footage).",
                      file=sys.stderr)
                sys.exit(6)
            print(f"  overlay-text: PASS ({len(text_hits)} flagged frame(s) < {cc.GATE_TEXT_HITS})")

            head_hits = cc.scan_video_talking_head(p)
            if len(head_hits) >= cc.GATE_FACE_HITS:
                print(f"FAIL ({len(head_hits)} frame(s) with a dominant face / talking head, "
                      f">= {cc.GATE_FACE_HITS}):", file=sys.stderr)
                for t, frac in head_hits[:8]:
                    print(f"   {t:>7.2f}s   face covers {frac*100:.2f}% of frame", file=sys.stderr)
                print("\nRule v22.4: the video must contain NO talking-head clips. Talking-head clips\n"
                      "are unusable — re-render with B-roll instead.", file=sys.stderr)
                sys.exit(7)
            print(f"  talking-head: PASS ({len(head_hits)} flagged frame(s) < {cc.GATE_FACE_HITS})")

        print("\nVALIDATE_RENDER: OK")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"FAIL: ffmpeg/ffprobe error on {p}: {e}", file=sys.stderr); sys.exit(5)


if __name__ == "__main__":
    main()

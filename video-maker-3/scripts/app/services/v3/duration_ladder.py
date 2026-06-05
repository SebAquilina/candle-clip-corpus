"""Materialize a corpus window into a renderable 1920x1080 / 30fps clip + one credit.

video-maker-3 never freezes or holds a frame. A clip is either long enough (trim it) or it
is used at its natural length and the assembler appends the next-best clip to cover the rest
(best-clip-first concat). So the only strategies here are TRIM and NATURAL — the legacy
slow-mo / last-frame-hold / Ken-Burns paths are gone.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

# scale+crop to 1080p, 30fps. One credit overlay, bottom-left, applied here at materialize
# time only (the renderer must NOT burn a second one).
_BASE_VF = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30"


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ], stderr=subprocess.DEVNULL, timeout=30).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def _vf(watermark: str = "") -> str:
    vf = _BASE_VF
    if watermark:
        safe = watermark.replace("'", "").replace(":", " -").replace(",", "")[:60]
        vf += (f",drawtext=text='{safe}':fontcolor=white@0.95:fontsize=22:x=20:y=h-32:"
               f"box=1:boxcolor=black@0.6:boxborderw=8")
    return vf


def fit_clip(src: Path, out: Path, shot_dur: float, watermark: str = "") -> dict:
    """Normalize `src` to 1920x1080/30fps with one bottom-left credit.

    If the source is at least `shot_dur` long, trim the leading `shot_dur` seconds (the
    matched window starts at the intended moment, so trim from the START, not the centre).
    Otherwise emit the whole clip at its natural length (< shot_dur) — NEVER a frozen tail.
    Returns {"strategy", "ok", "out_dur"}.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    src_dur = probe_duration(src)
    if src_dur <= 0:
        return {"strategy": "none", "ok": False, "out_dur": 0.0}
    vf = _vf(watermark)

    if src_dur >= shot_dur > 0:
        try:
            subprocess.check_call([
                "ffmpeg", "-y", "-i", str(src), "-t", f"{shot_dur:.3f}",
                "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "22", "-pix_fmt", "yuv420p", str(out),
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=180)
            return {"strategy": "trim", "ok": True, "out_dur": probe_duration(out)}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {"strategy": "trim_failed", "ok": False, "out_dur": 0.0}

    # Shorter than the slot: keep the REAL clip at its natural length. The assembler/renderer
    # appends the next-best distinct clip to fill the remainder (no hold, no loop, no freeze).
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", str(src), "-vf", vf, "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", str(out),
        ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=180)
        return {"strategy": "natural", "ok": True, "out_dur": probe_duration(out)}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"strategy": "natural_failed", "ok": False, "out_dur": 0.0}

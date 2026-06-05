"""Lightweight bootstrap check for video-maker-3.

Unlike the old skill this does NOT probe Gemini, Pexels, yt-dlp cookies or run a YouTube
smoke test. It just makes the bundled venv usable and warns (once) if a core dependency is
missing, pointing at bootstrap.sh which does the real setup. Never raises on import.
"""
from __future__ import annotations
import os
import shutil
import sys

_DONE = False


def auto_bootstrap_if_needed():
    global _DONE
    if _DONE:
        return
    _DONE = True
    here = os.path.dirname(os.path.abspath(__file__))
    ws = os.environ.get("WS") or os.path.dirname(here)
    vbin = os.path.join(ws, ".venv", "bin")
    if os.path.isdir(vbin) and vbin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = vbin + os.pathsep + os.environ.get("PATH", "")
    missing = []
    for binname in ("ffmpeg", "ffprobe"):
        if not shutil.which(binname):
            missing.append(binname)
    for mod in ("numpy", "edge_tts"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        sys.stderr.write(
            f"[video-maker-3] missing: {', '.join(sorted(set(missing)))} — "
            f"run `bash bootstrap.sh` in the skill dir first.\n")

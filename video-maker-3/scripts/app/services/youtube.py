"""YouTube segment download — the ONLY live-YouTube touchpoint in video-maker-3.

Discovery is gone: the clean corpus (outputs/shared_db_v2) already holds every vetted
B-roll window as `[start_s, end_s]` of a known source URL. To turn a matched window into a
renderable file we download exactly that range — a small, deterministic fetch, not a search
or a pool. Needs the same unblockers as the corpus builder: yt-dlp-ejs (`--remote-components
ejs:github`) + deno + node on PATH, and fresh full-auth cookies (else YouTube returns only
storyboards). Cookies are read from $VM_COOKIES / $REVAMP_COOKIES / $YTA_COOKIES, or
<storage>/youtube_cookies.txt.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from functools import lru_cache


def _cookies_path() -> str | None:
    """First existing, non-trivial cookies file from env, then the storage dir."""
    for ev in ("VM_COOKIES", "REVAMP_COOKIES", "YTA_COOKIES"):
        p = os.environ.get(ev, "")
        if p and os.path.exists(p) and os.path.getsize(p) > 100:
            return p
    try:
        from app.config import settings
        p = settings.storage_path / "youtube_cookies.txt"
        if p.exists() and p.stat().st_size > 100:
            return str(p)
    except Exception:
        pass
    return None


@lru_cache(maxsize=1024)
def fetch_channel(url: str) -> str:
    """Resolve a video's channel/uploader for the on-clip credit. '' if unresolved."""
    if not url:
        return ""
    try:
        import yt_dlp
    except Exception:
        return ""
    opts = {"quiet": True, "skip_download": True, "noplaylist": True, "nocheckcertificate": True}
    cookies = _cookies_path()
    if cookies:
        opts["cookiefile"] = cookies
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return (info.get("channel") or info.get("uploader") or "").strip()
    except Exception:
        return ""


def download_segment(video_url: str, start: float, end: float, out_path: Path) -> bool:
    """Download the [start, end] second range of a YouTube video to out_path (H.264/mp4).

    H.264 is forced (avc1/mp4) because OpenCV — used by the final gate — cannot decode AV1.
    Tries cookies first, then alternative player clients, then plain, so it degrades like the
    corpus builder. Returns True iff a non-trivial file landed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s, e = max(0.0, float(start)), float(end)
    base = [
        "yt-dlp", "-q",
        "--download-sections", f"*{s:.2f}-{e:.2f}",
        "--force-keyframes-at-cuts",
        # solve YouTube's `n` challenge via deno (must be on PATH)
        "--remote-components", "ejs:github", "--js-runtimes", "node",
        # avc1 only: cv2 can't decode AV1, which would read 0 frames at the gate
        "-f", "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/b[height<=1080][vcodec^=avc1]/"
              "bv*[height<=1080][ext=mp4]+ba/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "--no-playlist", "--no-check-certificate",
        "-o", str(out_path),
    ]
    cookies = _cookies_path()
    strategies = []
    if cookies:
        strategies.append(base + ["--cookies", cookies, video_url])
    strategies.append(base + ["--extractor-args", "youtube:player_client=mweb,android,tv", video_url])
    strategies.append(base + [video_url])
    timeout = int(os.environ.get("VM_DOWNLOAD_TIMEOUT", "180"))
    for cmd in strategies:
        try:
            subprocess.run(cmd, timeout=timeout, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT, check=True)
            if out_path.exists() and out_path.stat().st_size > 50_000:
                return True
        except Exception:
            pass
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
    return False

"""Download a source video once (≤720p) into a deletable scratch dir, scan, delete.

Grouping by video_id (one download, many windows scanned by seeking) keeps it to ~257
downloads instead of one per window. Uses the yt_dlp module (no binary needed); node is
put on PATH for yt-dlp's JS challenge; cookies are used when available.
"""
from __future__ import annotations
import os
from pathlib import Path

# node for yt-dlp's JS runtime
for _nd in ("/opt/node22/bin", "/usr/bin", "/usr/local/bin"):
    if os.path.isdir(_nd) and _nd not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _nd + os.pathsep + os.environ.get("PATH", "")

SCRATCH = Path(os.environ.get("REVAMP_SCRATCH", "/tmp/revamp_dl"))
SCRATCH.mkdir(parents=True, exist_ok=True)
MAX_H = int(os.environ.get("REVAMP_MAX_HEIGHT", "720"))


def _cookies() -> str:
    c = os.environ.get("REVAMP_COOKIES", "")
    if c and os.path.exists(c):
        return c
    for cand in ("/tmp/yta_ws/state/youtube_cookies.txt",):
        if os.path.exists(cand):
            return cand
    return ""


def download(video_id: str, url: str = "") -> dict:
    """Download one video ≤MAX_H. Returns {ok, path, err, height, duration}."""
    import yt_dlp
    url = url or f"https://www.youtube.com/watch?v={video_id}"
    # reuse if already on disk
    for ext in ("mp4", "mkv", "webm"):
        p = SCRATCH / f"{video_id}.{ext}"
        if p.exists() and p.stat().st_size > 10000:
            return {"ok": True, "path": str(p), "err": "", "height": None, "duration": None}
    out_tmpl = str(SCRATCH / "%(id)s.%(ext)s")
    # Force H.264 (avc1): YouTube also serves AV1, which this platform's ffmpeg/cv2 CANNOT
    # decode (every frame read fails -> 0 frames -> the scan would falsely pass). avc1 is
    # available up to 1080p (e.g. fmt 136 at 720p). Fall back through mp4 then anything.
    fmt = (f"bv*[height<={MAX_H}][vcodec^=avc1]+ba[ext=m4a]/"
           f"bv*[height<={MAX_H}][vcodec^=avc1]+ba/"
           f"b[height<={MAX_H}][vcodec^=avc1]/"
           f"bv*[height<={MAX_H}][ext=mp4]+ba/b[height<={MAX_H}][ext=mp4]/b[height<={MAX_H}]/b")
    opts = {
        "format": fmt, "outtmpl": out_tmpl, "quiet": True, "no_warnings": True,
        "noprogress": True, "retries": 3, "fragment_retries": 3,
        "merge_output_format": "mp4", "concurrent_fragment_downloads": 4,
        # Socket timeout so a hung server doesn't freeze the grind forever (an earlier
        # run stalled for 2h on a single video). yt_dlp's own retry will kick in.
        "socket_timeout": int(os.environ.get("REVAMP_SOCKET_TIMEOUT", "30")),
        # Sandbox egress is a transparent TLS-intercepting proxy with a self-signed root
        # that yt_dlp's bundled certifi doesn't trust. Content is public video, so skip
        # cert verification here (the system CA bundle path doesn't reach yt_dlp).
        "nocheckcertificate": True,
        "extractor_args": {"youtube": {"player_client": ["mweb", "android", "tv", "web"]}},
    }
    ck = _cookies()
    if ck:
        opts["cookiefile"] = ck
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        for ext in ("mp4", "mkv", "webm"):
            p = SCRATCH / f"{video_id}.{ext}"
            if p.exists() and p.stat().st_size > 10000:
                return {"ok": True, "path": str(p), "err": "",
                        "height": info.get("height"), "duration": info.get("duration")}
        return {"ok": False, "path": "", "err": "downloaded but no file found",
                "height": None, "duration": None}
    except Exception as e:
        return {"ok": False, "path": "", "err": str(e)[:200], "height": None, "duration": None}


def discard(video_id: str) -> None:
    """Delete the scratch download for a video (scratch is on a deletable mount)."""
    for ext in ("mp4", "mkv", "webm", "part", "ytdl"):
        for p in SCRATCH.glob(f"{video_id}.{ext}*"):
            try:
                p.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import sys, json
    vid = sys.argv[1] if len(sys.argv) > 1 else "ni6Z78PCyVw"
    print(json.dumps(download(vid), indent=2))

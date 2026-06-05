"""Per-video transcript with word/segment timings, for per-second alignment in stage 4.

Two sources, in order:
  1. YouTube timed captions (yt_dlp, json3) — free, no compute, the speaker's actual words.
  2. faster-whisper on the downloaded media — fallback when a video has no captions.

Returns a flat list of (start_s, end_s, token) that describe.transcript_for_second() slices.
"""
from __future__ import annotations
import os, json, glob, tempfile

CACHE = os.environ.get("REVAMP_TRANSCRIPT_CACHE", "/tmp/revamp_transcripts")
os.makedirs(CACHE, exist_ok=True)


def _parse_json3(path: str) -> list:
    """Parse a YouTube json3 caption file into (start,end,token) word-ish tuples."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    out = []
    for ev in data.get("events", []):
        segs = ev.get("segs") or []
        t0 = ev.get("tStartMs", 0) / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        toks = text.split()
        if not toks:
            continue
        # distribute the event's duration across its tokens
        step = (dur / len(toks)) if dur > 0 else 0.4
        for i, tok in enumerate(toks):
            s = t0 + i * step
            out.append((round(s, 2), round(s + step, 2), tok))
    return out


def youtube_transcript(video_id: str, url: str = "") -> list:
    """Fetch auto/uploaded English captions via yt_dlp (json3). [] if none/unavailable."""
    import yt_dlp
    url = url or f"https://www.youtube.com/watch?v={video_id}"
    base = os.path.join(tempfile.mkdtemp(prefix="subs_", dir=CACHE), "%(id)s")
    opts = {
        "skip_download": True, "writeautomaticsub": True, "writesubtitles": True,
        "subtitleslangs": ["en", "en-US", "en-orig"], "subtitlesformat": "json3",
        "outtmpl": base, "quiet": True, "no_warnings": True, "nocheckcertificate": True,
        "extractor_args": {"youtube": {"player_client": ["mweb", "web", "android"]}},
    }
    ck = os.environ.get("REVAMP_COOKIES", "/tmp/yta_ws/state/youtube_cookies.txt")
    if os.path.exists(ck):
        opts["cookiefile"] = ck
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
    except Exception:
        pass
    cands = glob.glob(os.path.dirname(base) + "/*.json3") + glob.glob(os.path.dirname(base) + "/*.json")
    for c in sorted(cands):
        words = _parse_json3(c)
        if words:
            return words
    return []


def whisper_transcript(media_path: str, model_size: str = "tiny.en") -> list:
    """Fallback: faster-whisper word timings from the media's audio."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return []
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(media_path, word_timestamps=True, vad_filter=True, beam_size=1)
        out = []
        for seg in segments:
            for w in (seg.words or []):
                out.append((round(float(w.start), 2), round(float(w.end), 2), w.word.strip()))
        return out
    except Exception:
        return []


def get_transcript(video_id: str, media_path: str = "", url: str = "") -> list:
    """Cached: YouTube captions first, then whisper on the media. Returns [(s,e,tok)]."""
    cf = os.path.join(CACHE, f"{video_id}.json")
    if os.path.exists(cf):
        try:
            return [tuple(x) for x in json.load(open(cf))]
        except Exception:
            pass
    words = youtube_transcript(video_id, url)
    if not words and media_path and os.path.exists(media_path):
        words = whisper_transcript(media_path)
    try:
        json.dump(words, open(cf, "w"))
    except Exception:
        pass
    return words


if __name__ == "__main__":
    import sys
    if "--whisper" in sys.argv:
        p = sys.argv[sys.argv.index("--whisper") + 1]
        w = whisper_transcript(p)
        print(f"whisper words: {len(w)}")
        for x in w[:12]:
            print("  ", x)
    else:
        vid = sys.argv[1]
        w = get_transcript(vid)
        print(f"{vid}: {len(w)} tokens; first: {w[:8]}")

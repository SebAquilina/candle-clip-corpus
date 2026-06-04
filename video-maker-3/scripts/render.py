"""video-maker-3 renderer — concat pre-materialized segments + atomic mux.

The assembler (section_planner) already produced the exact back-to-back shot list honouring
best-clip-first concat + the no-repeat rules, and the driver already materialized each shot
into an exact-duration 1920x1080/30fps credited segment. So rendering is now just: concat
the segments to a silent track and mux the narration over it (loudnorm, +faststart), atomically.

No freeze, no last-frame hold, no from-scratch re-sourcing, no black panels — those paths
are gone. A segment that failed to materialize was already dropped/substituted upstream, so
every segment here is a real moving clip. The non-skippable final gate (validate_render.py)
is still run afterwards as the "just in case" safety check.
"""
from __future__ import annotations
import os
import re
import subprocess
from pathlib import Path


def probe_dur(p: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def mostly_black(p: Path) -> bool:
    """True if the clip is essentially black/blank over a short sample (corrupt/truncated
    downloads render to black). Used as the light per-segment check."""
    try:
        dur = probe_dur(p)
        if dur <= 0:
            return True
        sample = min(dur, 6.0)
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-t", f"{sample:.2f}", "-i", str(p),
             "-vf", "blackdetect=d=0.5:pix_th=0.10", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30)
        black = 0.0
        for m in re.finditer(r"black_start:([0-9.]+) black_end:([0-9.]+)", r.stderr or ""):
            black += float(m.group(2)) - float(m.group(1))
        return black > 0.6 * sample
    except Exception:
        return True


def _concat(parts, out: Path, reencode=False) -> bool:
    if not parts:
        return False
    lst = out.with_suffix(".concat.txt")
    lst.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in parts))
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-r", "30", "-an"]
    else:
        cmd += ["-c", "copy"]
    try:
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        ok = out.exists() and out.stat().st_size > 10_000
    except subprocess.CalledProcessError:
        ok = False
    try:
        lst.unlink()
    except Exception:
        pass
    return ok


def render(shots, audio_dir: Path, out_path: Path, work_dir: Path, progress_cb=print) -> str:
    """shots: ordered list with `final_clip_path` (exact-duration 1080p/30 segment) per shot.
    audio_dir: holds para_*.mp3. Concats segments + muxes the narration. Returns a summary."""
    audio_dir, out_path, work_dir = Path(audio_dir), Path(out_path), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. concat narration audio
    audio_files = sorted(audio_dir.glob("para_*.mp3"))
    if not audio_files:
        raise RuntimeError(f"no narration audio in {audio_dir}")
    audio_concat = work_dir / "narration.wav"
    alist = work_dir / "narration.txt"
    alist.write_text("\n".join(f"file '{f.resolve()}'" for f in audio_files))
    subprocess.check_call(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(alist),
                           "-ac", "2", "-ar", "48000", str(audio_concat)], stderr=subprocess.DEVNULL)
    audio_dur = probe_dur(audio_concat)

    # 2. collect the materialized segments in order (skip any missing/black — already rare)
    seg_paths, total_v = [], 0.0
    for sh in shots:
        fp = sh.get("final_clip_path")
        if not fp:
            continue
        p = Path(fp)
        if not p.exists() or p.stat().st_size < 20_000 or mostly_black(p):
            progress_cb(f"  shot {sh.get('shot_idx')}: segment unusable, skipping ({fp})")
            continue
        seg_paths.append(p)
        total_v += probe_dur(p)
    if not seg_paths:
        raise RuntimeError("no usable segments to render")
    progress_cb(f"  {len(seg_paths)} segments, video ~{total_v:.1f}s vs audio {audio_dur:.1f}s")

    # 3. concat segments -> silent video (try stream-copy; re-encode if params drifted)
    silent = work_dir / "silent.mp4"
    if not _concat(seg_paths, silent, reencode=False) or abs(probe_dur(silent) - total_v) > 2.0:
        _concat(seg_paths, silent, reencode=True)
    if not silent.exists():
        raise RuntimeError("segment concat failed")

    # 4. ATOMIC mux: encode to a temp file then os.replace, so a killed mux never leaves a
    #    corrupt final file. loudnorm for consistent narration level; +faststart for the web.
    tmp = out_path.with_suffix(".muxing.mp4")
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(silent), "-i", str(audio_concat),
        "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(tmp),
    ], stderr=subprocess.DEVNULL)
    os.replace(tmp, out_path)
    return f"rendered {len(seg_paths)} segments, video~{total_v:.1f}s audio={audio_dur:.1f}s -> {out_path}"

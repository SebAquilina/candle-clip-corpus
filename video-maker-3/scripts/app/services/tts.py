"""EdgeTTS chunked synthesis. Produces per-paragraph audio files."""
from __future__ import annotations
import asyncio
from pathlib import Path
import edge_tts

async def synthesize_paragraph(text: str, voice: str, out_path: Path) -> None:
    import os
    # Apply env prosody for more natural pacing (Rule v22.1); neutral by default.
    rate = os.environ.get("EDGETTS_RATE", "+0%")
    pitch = os.environ.get("EDGETTS_PITCH", "+0Hz")
    try:
        com = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await com.save(str(out_path))
    except Exception:
        # unavailable voice or transient error -> safe natural default
        com = edge_tts.Communicate(text, "en-US-AriaNeural", rate=rate, pitch=pitch)
        await com.save(str(out_path))

async def synthesize_all(paragraphs: list[str], voice: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, p in enumerate(paragraphs):
        f = out_dir / f"para_{i:03d}.mp3"
        if not (f.exists() and f.stat().st_size > 1000):
            await synthesize_paragraph(p, voice, f)
        paths.append(f)
    return paths

def split_into_sentences(paragraph: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p.strip() for p in parts if p.strip()]

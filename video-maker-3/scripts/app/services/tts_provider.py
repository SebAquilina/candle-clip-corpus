"""TTS provider dispatcher for the video-maker pipeline.

Chooses the TTS backend based on environment variables, so the pipeline code
doesn't need to know which provider is active.

Provider selection (in priority order):
  1. If LABS69_API_KEY and LABS69_VOICE_ID are set -> 69labs
  2. Otherwise -> EdgeTTS (free fallback, voice from EDGETTS_VOICE or default)

Env vars:
  LABS69_API_KEY        69labs key (vk_...). Triggers 69labs when present.
  LABS69_VOICE_ID       the voice code (ElevenLabs-style id, e.g. XH7KR8MDn5xIMYpbfUTx)
  LABS69_VOICE_PROVIDER elevenlabs | edgetts | minimax  (default elevenlabs)
  LABS69_MODEL_ID       optional model override (default eleven_multilingual_v2)
  LABS69_VOICE_CLONE_ID optional: use a cloned voice instead of a stock voice id
  LABS69_STABILITY / LABS69_SIMILARITY / LABS69_STYLE / LABS69_SPEAKER_BOOST
                        ElevenLabs voice-setting overrides for naturalness (Rule v22.1)
  EDGETTS_VOICE         EdgeTTS voice (default en-US-AvaMultilingualNeural — natural)
  EDGETTS_RATE / EDGETTS_PITCH   EdgeTTS prosody (default +0%, +0Hz)

All providers expose the SAME async interface so run_v13 can call it uniformly:
  await synthesize_all(paragraphs, out_dir) -> list[Path]   # writes para_NNN.mp3
"""
from __future__ import annotations
import os
import asyncio
from pathlib import Path


def _labs_voice_settings():
    """ElevenLabs-style voice settings tuned for natural, expressive narration
    (Rule v22.1). Only meaningful for the elevenlabs provider. All env-overridable:
      LABS69_STABILITY (0.45)  lower => more lifelike prosody variation, less monotone
      LABS69_SIMILARITY (0.80) closeness to the reference voice
      LABS69_STYLE (0.35)      expressiveness / delivery style
      LABS69_SPEAKER_BOOST (1) clarity boost (1/0)
    Returns None for non-elevenlabs providers (minimax/clone use their own params)."""
    if os.environ.get("LABS69_VOICE_PROVIDER", "elevenlabs") != "elevenlabs":
        return None

    def _f(name, default):
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return float(default)

    boost = os.environ.get("LABS69_SPEAKER_BOOST", "1").strip().lower() not in ("0", "false", "no", "")
    return {
        "stability": _f("LABS69_STABILITY", "0.45"),
        "similarity_boost": _f("LABS69_SIMILARITY", "0.80"),
        "style": _f("LABS69_STYLE", "0.35"),
        "use_speaker_boost": boost,
    }


def active_provider() -> str:
    if os.environ.get("LABS69_API_KEY") and (
        os.environ.get("LABS69_VOICE_ID") or os.environ.get("LABS69_VOICE_CLONE_ID")
    ):
        return "69labs"
    return "edgetts"


def describe() -> str:
    p = active_provider()
    if p == "69labs":
        if os.environ.get("LABS69_VOICE_CLONE_ID"):
            return f"69labs (clone {os.environ['LABS69_VOICE_CLONE_ID']})"
        return (f"69labs voice={os.environ.get('LABS69_VOICE_ID')} "
                f"provider={os.environ.get('LABS69_VOICE_PROVIDER', 'elevenlabs')}")
    return f"edgetts voice={os.environ.get('EDGETTS_VOICE', 'en-US-AvaMultilingualNeural')}"


async def synthesize_all(paragraphs: list[str], out_dir: Path) -> list[Path]:
    """Write para_000.mp3 ... per paragraph using the active provider. Resumable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = active_provider()
    paths: list[Path] = []

    if provider == "69labs":
        # tts_69labs lives in app/services/. Import it as a package member; fall back to a
        # top-level import for the legacy layout where scripts were flattened onto sys.path.
        try:
            from app.services import tts_69labs as labs
        except ImportError:
            import tts_69labs as labs
        api_key = os.environ["LABS69_API_KEY"]
        voice_id = os.environ.get("LABS69_VOICE_ID", "")
        clone_id = os.environ.get("LABS69_VOICE_CLONE_ID", "")
        voice_provider = os.environ.get("LABS69_VOICE_PROVIDER", "elevenlabs")
        # Default to ElevenLabs' most natural narration model unless overridden.
        model_id = os.environ.get("LABS69_MODEL_ID") or "eleven_multilingual_v2"
        voice_settings = _labs_voice_settings()  # tuned for naturalness (Rule v22.1)
        for i, p in enumerate(paragraphs):
            f = out_dir / f"para_{i:03d}.mp3"
            if f.exists() and f.stat().st_size > 1000:
                paths.append(f); continue
            if clone_id:
                res = await asyncio.to_thread(
                    labs.synthesize_with_clone, api_key, p, clone_id, f)
            else:
                res = await asyncio.to_thread(
                    labs.synthesize, api_key, p, voice_id, f,
                    voice_provider=voice_provider, model_id=model_id,
                    voice_settings=voice_settings)
            if not res.get("ok"):
                # CENSORED or otherwise — surface so caller can handle
                raise RuntimeError(
                    f"69labs TTS did not complete for paragraph {i}: "
                    f"status={res.get('status')} blocked={res.get('blocked_chunks')}")
            paths.append(f)
        return paths

    # EdgeTTS fallback. The multilingual neural voices are markedly more natural than the
    # older en-US-AriaNeural, so default to one — but fall back to Aria if it is unavailable
    # in this edge-tts build. Rate/pitch are env-tunable for pacing (Rule v22.1).
    import edge_tts
    voice = os.environ.get("EDGETTS_VOICE", "en-US-AvaMultilingualNeural")
    rate = os.environ.get("EDGETTS_RATE", "+0%")
    pitch = os.environ.get("EDGETTS_PITCH", "+0Hz")

    async def _edge_save(text, vc, dest):
        com = edge_tts.Communicate(text, vc, rate=rate, pitch=pitch)
        await com.save(str(dest))

    for i, p in enumerate(paragraphs):
        f = out_dir / f"para_{i:03d}.mp3"
        if not (f.exists() and f.stat().st_size > 1000):
            try:
                await _edge_save(p, voice, f)
            except Exception:
                # unknown/unavailable voice or transient error -> safe natural default
                await _edge_save(p, "en-US-AriaNeural", f)
        paths.append(f)
    return paths

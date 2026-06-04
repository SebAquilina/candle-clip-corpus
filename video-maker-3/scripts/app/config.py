"""Minimal settings for video-maker-3 (no pydantic dependency).

The streamlined skill needs almost nothing from config: a storage directory (where the
optional YouTube cookies file lives and where scratch/clip caches go) and an optional
Gemini key that ONLY the opt-in google embedder branch reads. Everything else the old
Settings carried (Pexels/Gemini discovery/CORS/ports) is gone with the modules that used it.
"""
from __future__ import annotations
import os
from pathlib import Path


class _Settings:
    # Only the google embedder (YTA_EMBEDDER=google, off by default) reads this.
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    STORAGE_DIR = os.environ.get("STORAGE_DIR", os.path.join(
        os.environ.get("WS", os.getcwd()), "state"))

    @property
    def storage_path(self) -> Path:
        p = Path(self.STORAGE_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        for sub in ("clips", "renders", "uploads"):
            (p / sub).mkdir(exist_ok=True)
        return p


settings = _Settings()

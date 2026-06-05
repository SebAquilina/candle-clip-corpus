"""Text embeddings for semantic shot<->moment matching (Rule v29).

BACKEND CHAIN (resolved once at import, memoized; preserves the public API
embed_many/embed_one/cosine):

  1. model2vec  (numpy-only, ~30 MB bundled, dim 256)   — PRIMARY (semantic, offline)
  2. fastembed  (BAAI/bge-small-en-v1.5, dim 384)        — opt-in (download at bootstrap)
  3. lexical    (hashed bag-of-words + char-n-gram, dim 2048) — terminal fallback
  4. google     (gemini-embedding-001, opt-in via YTA_EMBEDDER=google)

The lexical embedder is a LEXICAL similarity (word/substring overlap), not semantic — it
scored true paraphrases 0.10-0.18 and shared-noun false-friends 0.55-0.72, so "pouring
wax" was matching a "measure wax" clip purely because they share "wax/jar". model2vec
gives real semantic vectors offline; it falls back to lexical if the model is missing so
matching is never silently broken.

Vectors are returned as plain Python list[float] so the existing JSON caches
(pipeline_v3.py `embeds_file`, `shot_embeds.json`) keep working unchanged. The cache key
should include the backend name (`active_backend()`) to avoid mixing different-dim vectors
across runs — different backends use different dims, but cosine within one process always
compares vectors from the same backend.
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

_DIM = int(os.environ.get("YTA_EMBED_DIM", "2048"))
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "into", "onto", "and", "or", "for",
    "with", "is", "are", "be", "being", "it", "this", "that", "as", "at", "by",
    "from", "up", "out", "over", "then", "so", "we", "you", "your", "their", "his",
    "her", "its", "they", "them", "shows", "showing", "shot", "clip", "footage",
    "video", "scene", "screen",
}


# --------------------------------------------------------------------------- #
# backend chain resolution                                                    #
# --------------------------------------------------------------------------- #
def _model_dir(name="potion-base-8M"):
    """Locate a bundled model dir (assets/<name>) or honour YTA_EMBED_MODEL_DIR."""
    p = os.environ.get("YTA_EMBED_MODEL_DIR", "")
    if p and os.path.isdir(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "..", "..", "assets", name),    # skill_root/assets/<name>
              os.path.join(here, "..", "..", "..", "..", "assets", name),
              os.path.join(os.environ.get("WS", ""), "assets", name)):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return ""


@functools.lru_cache(maxsize=1)
def _resolve_backend():
    """Resolve once: try preferred backends in order; memoized for the whole process.
    Returns (name, encode_fn(texts)->list[list[float]], dim, default_min_sim, default_floor).
    Terminal fallback is lexical so embedding never crashes the pipeline."""
    pref = os.environ.get("YTA_EMBEDDER", "auto").strip().lower()
    if pref in ("google", "gemini"):
        return ("google", _gemini_embed_many, 3072, 0.55, 0.30)
    if pref == "lexical" or pref == "local":
        return ("lexical", _local_embed_many, _DIM, 0.55, 0.30)
    # auto chain (or explicit model2vec/fastembed)
    chain = ["model2vec", "fastembed", "lexical"] if pref == "auto" else [pref, "lexical"]
    for name in chain:
        try:
            if name == "model2vec":
                from model2vec import StaticModel
                path = _model_dir("potion-base-8M") or "minishlab/potion-base-8M"
                m = StaticModel.from_pretrained(path)
                def enc(texts, _m=m):
                    return [v.tolist() for v in _m.encode(list(texts))]
                # default_floor tuned to 0.18 against the real candle corpus: narration
                # sentences are long and corpus descriptions are short, so true matches
                # land low on model2vec cosine. 0.32 left most shots uncovered and dropped
                # them to per-shot keyword search; 0.18 keeps ~95% sourced from the corpus
                # (Rule v26.0 — try harder in the corpus before falling back). Env overrides
                # via YTA_LIBRARY_MIN_SIM_FLOOR still win.
                return ("model2vec", enc, 256, 0.50, 0.18)
            if name == "fastembed":
                from fastembed import TextEmbedding
                cache = _model_dir("fastembed") or os.environ.get("YTA_EMBED_MODEL_DIR")
                te = TextEmbedding(
                    model_name=os.environ.get("YTA_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"),
                    cache_dir=cache, local_files_only=bool(cache))
                def enc(texts, _t=te):
                    return [v.tolist() for v in _t.embed(list(texts))]
                return ("fastembed", enc, 384, 0.60, 0.40)
            if name == "lexical":
                return ("lexical", _local_embed_many, _DIM, 0.55, 0.30)
        except Exception:
            continue                                # degrade to next link
    return ("lexical", _local_embed_many, _DIM, 0.55, 0.30)


def active_backend():
    """The name of the resolved backend ('model2vec' | 'fastembed' | 'lexical' | 'google').
    Use this as part of a vector-cache key so a backend change invalidates stale cached vectors."""
    return _resolve_backend()[0]


def backend_thresholds():
    """(default_min_sim, default_floor) tuned per backend; callers can still override via env."""
    name, _enc, _dim, mn, fl = _resolve_backend()
    return mn, fl


# --------------------------------------------------------------------------- #
# lexical backend (terminal fallback, kept verbatim)                          #
# --------------------------------------------------------------------------- #
def _bucket(feat: str) -> int:
    return int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16) % _DIM


def _features(text: str):
    for w in _WORD.findall((text or "").lower()):
        if w in _STOP or len(w) < 2:
            continue
        yield "w:" + w, 2.0
        if len(w) >= 4:
            pad = "#" + w + "#"
            for n in (3, 4):
                for i in range(len(pad) - n + 1):
                    yield "g:" + pad[i:i + n], 1.0


def _local_embed_one(text: str):
    v = [0.0] * _DIM
    for feat, wt in _features(text):
        v[_bucket(feat)] += wt
    norm = math.sqrt(sum(x * x for x in v))
    if norm:
        v = [x / norm for x in v]
    return v


def _local_embed_many(texts):
    return [_local_embed_one(t) for t in texts]


# --------------------------------------------------------------------------- #
# google backend (opt-in: YTA_EMBEDDER=google)                                #
# --------------------------------------------------------------------------- #
_MODEL = "models/gemini-embedding-001"
_API = "https://generativelanguage.googleapis.com/v1beta"


def _gemini_embed_batch(texts):
    if not texts:
        return []
    from app.config import settings
    key = settings.GEMINI_API_KEY
    url = f"{_API}/{_MODEL}:batchEmbedContents?key={key}"
    body = {"requests": [{"model": _MODEL, "content": {"parts": [{"text": t[:8000] or " "}]}} for t in texts]}
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return [e["values"] for e in data.get("embeddings", [])]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    return []


def _gemini_embed_many(texts):
    out = []
    for i in range(0, len(texts), 10):
        out.extend(_gemini_embed_batch(texts[i:i + 10]))
    return out


# --------------------------------------------------------------------------- #
# public API (backend-dispatched, with per-call safety fallback to lexical)   #
# --------------------------------------------------------------------------- #
def embed_many(texts):
    if not texts:
        return []
    _name, enc, _dim, _, _ = _resolve_backend()
    try:
        return enc(texts)
    except Exception:
        return _local_embed_many(texts)


def embed_one(text):
    if not text:
        return []
    return embed_many([text])[0]


def cosine(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

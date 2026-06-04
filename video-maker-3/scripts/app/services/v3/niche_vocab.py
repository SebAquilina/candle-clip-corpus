"""v2.7: niche-specific vocabulary so the OFFLINE matcher understands jargon, and
descriptions include the right terms (Rule v28.0).

Candle-making (and any niche) has terminology — tunneling, borehole, sinkhole, wet
spots, frosting, mushrooming, memory ring — that a plain bag-of-words / n-gram embedder
can't relate to a plain-language clip description ("wax burned down the middle leaving a
ring on the sides"). This module loads a per-niche glossary (`niche_vocab.json`) and:

  * expand(text)   -> the text PLUS the synonyms and visual-cue phrases of any jargon it
                      contains, so after expansion "tunneling" and "a ring of unmelted wax
                      up the sides" embed near each other. Used on BOTH the moment
                      descriptions and the shot concepts before embedding, so matching
                      understands the terms. Bidirectional: a plain-language description
                      that matches a term's visual cue also gets the canonical term added.
  * terms_in(text) -> the canonical jargon terms detected (for descriptions / reports).

Pure / offline / stdlib-only (json, re). Niche is taken from `YTA_SHARED_NICHE` (or the
`niche=` arg); `_common` always applies. `YTA_NICHE_VOCAB=0` disables expansion.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

_PATH = os.path.join(os.path.dirname(__file__), "niche_vocab.json")


def enabled():
    return os.environ.get("YTA_NICHE_VOCAB", "1").strip().lower() not in ("0", "false", "no")


@lru_cache(maxsize=8)
def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _niche_key(niche):
    n = (niche or os.environ.get("YTA_SHARED_NICHE", "")).strip().lower()
    return n.replace(" ", "_").replace("-", "_")


@lru_cache(maxsize=64)
def _entries(niche_key, path):
    """Flattened entries for a niche + _common: [(canonical, aka_tuple, visual, phrases)]."""
    data = _load(path)
    out = []
    for key in (niche_key, "_common"):
        section = data.get(key) or {}
        if not isinstance(section, dict):
            continue
        for canonical, d in section.items():
            if canonical.startswith("_") or not isinstance(d, dict):
                continue
            aka = tuple(str(a) for a in (d.get("aka") or []))
            visual = str(d.get("visual") or "")
            # the phrases that, if present in plain text, imply this term (cheap cue match)
            cue = tuple(p for p in re.split(r"[;,]", visual) if len(p.strip()) >= 8)
            out.append((canonical, aka, visual, cue))
    return tuple(out)


def _surface_forms(canonical, aka):
    forms = {canonical.replace("_", " "), canonical.replace("_", "")}
    forms.update(a.lower() for a in aka)
    return [f for f in forms if f]


def _present(text_l, phrase):
    """Loose containment: the phrase's salient words mostly appear in the text."""
    pw = [w for w in re.findall(r"[a-z]+", phrase.lower()) if len(w) >= 4]
    if not pw:
        return False
    hit = sum(1 for w in pw if w in text_l)
    return hit >= max(2, int(0.6 * len(pw)))


def terms_in(text, niche=None, path=None):
    """Canonical jargon terms detected in `text` (by surface form OR visual cue)."""
    if not text:
        return []
    tl = " " + text.lower() + " "
    found = []
    for canonical, aka, visual, cue in _entries(_niche_key(niche), path or _PATH):
        forms = _surface_forms(canonical, aka)
        if any(re.search(r"\b" + re.escape(f) + r"\b", tl) for f in forms) \
                or any(_present(tl, c) for c in cue):
            found.append(canonical)
    return found


def expand(text, niche=None, path=None):
    """text + synonyms + visual-cue phrases (+ canonical term) for any jargon it contains.

    Symmetric so both sides of a match converge: a description that SAYS "tunneling" gains
    the visual cue; a description that DESCRIBES the look gains the word "tunneling".
    """
    if not text or not enabled():
        return text
    tl = " " + text.lower() + " "
    extra = []
    for canonical, aka, visual, cue in _entries(_niche_key(niche), path or _PATH):
        forms = _surface_forms(canonical, aka)
        by_word = any(re.search(r"\b" + re.escape(f) + r"\b", tl) for f in forms)
        by_cue = any(_present(tl, c) for c in cue)
        if by_word or by_cue:
            extra.append(canonical.replace("_", " "))
            extra.extend(aka)
            if visual:
                extra.append(visual)
    if not extra:
        return text
    return text + " . " + " . ".join(dict.fromkeys(extra))

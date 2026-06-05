"""Action-aware re-rank — layered on top of cosine to fix the false-friend matching bug.

Today the matcher picks a "measure_wax" clip for a "pouring wax" narration because they
share more surface nouns than the correct "pour_wax" clip — pure cosine over the lexical
embedder rewards shared words, and the embedded text has no notion of *action*. This
re-rank reduces both sides to an action GROUP (using the corpus action_label as the
authoritative signal on the clip side, and a small synonym map on the shot side) and
scores: final = w1*cosine + w2*(actions agree) - w3*(actions disagree). The penalty is
larger than the bonus so a clear conflict (pour vs melt) is pushed below a no-conflict
candidate at the same cosine — exactly the pour/melt/measure/dye separation.

Behaviour-preserving: when either side has no detectable action, both terms are zero and
ranking reduces to pure cosine. Works under any embedder (semantic or lexical).
"""
from __future__ import annotations
import os
import re

# Action groups: each corpus action_label maps to a group; shot-side narration is scanned
# for the group's synonyms. Groups contain VERBS / discriminating modifiers, NOT bare
# object nouns (e.g. "wick" alone is the object — a "trimming the wick" narration must
# resolve to TRIM, not WICK; we keep "wick" out of the WICK group and use action verbs
# like center/secure/attach instead). action_of returns the group with the MOST synonym
# hits so narrations that mention multiple things (e.g. "fragrance oil into the melted
# wax") pick the dominant action by count, not by dict-traversal order.
_ACTION_SYNONYMS = {
    "fragrance": {"fragrance", "fragrances", "scent", "scents", "scented", "essential",
                  "perfume"},
    "dye":       {"dye", "dyes", "dyeing", "color", "colour", "colouring", "coloring",
                  "pigment", "tint", "tinting"},
    "trim":      {"trim", "trimming", "trimmed", "snip", "snipping", "scissors"},
    "measure":   {"measure", "measuring", "measured", "weigh", "weighing", "weighed",
                  "scale", "grams"},
    "wick":      {"center", "centre", "centred", "centered", "centering", "centring",
                  "stick", "sticker", "sticking", "attach", "attaching",
                  "secure", "securing"},
    "pour":      {"pour", "pouring", "poured", "fill", "filling", "decant", "decanting"},
    "melt":      {"melt", "melting", "melted", "liquefy", "liquefying", "boiler",
                  "simmer", "simmering"},
    "cure":      {"cure", "curing", "cool", "cooling", "rest", "resting", "shelf",
                  "harden", "hardening"},
    "prepare":   {"prepare", "preparing", "prep", "clean", "wipe"},
    "decorate":  {"decorate", "decorating", "paint", "painting", "sprinkle", "sprinkles",
                  "embed", "embedding"},
    "monitor":   {"monitor", "monitoring", "temperature", "thermometer"},
}

# action_label -> group (the corpus labels we know about). action_labels not listed here
# fall back to scanning their tokens against _ACTION_SYNONYMS, which is fine for niches
# beyond candle-making (soap, ceramics, etc).
_LABEL_GROUP = {
    "pour_wax": "pour", "pour_mould": "pour", "melt_wax": "melt", "melt_oils": "melt",
    "measure_wax": "measure", "add_dye_color": "dye", "add_dye": "dye",
    "add_fragrance": "fragrance", "set_wick": "wick", "attach_wick": "wick",
    "trim_wick": "trim", "cut_bars": "trim", "cure_cool": "cure", "cure_soap": "cure",
    "prepare_container": "prepare", "gather_materials": "prepare",
    "decorate_finish": "decorate", "monitor_temperature": "monitor",
    "stir_wax": "pour",  # stirring molten wax shares the pour group's tools
}

_WORD = re.compile(r"[a-z]+")


def action_of(text="", label=""):
    """Map a side to its action group. Authoritative on the clip side via _LABEL_GROUP.
    On the shot side, count synonym hits per group and return the highest — so a narration
    that mentions multiple things ("fragrance oil into the MELTED wax") resolves by
    dominance, not by dict-traversal order. Ties broken by dict insertion order, which is
    "more discriminating groups first" (fragrance > melt, trim > wick). Returns '' when
    no group matched (then the rerank's bonus/penalty terms are both zero — safe no-op)."""
    if label and label in _LABEL_GROUP:
        return _LABEL_GROUP[label]
    toks = set(_WORD.findall((label + " " + (text or "")).lower()))
    if not toks:
        return ""
    best, best_n = "", 0
    for group, syns in _ACTION_SYNONYMS.items():
        n = len(toks & syns)
        if n > best_n:                # strict >: first group on a tie wins (insertion order)
            best, best_n = group, n
    return best


def rerank_enabled():
    return os.environ.get("YTA_ACTION_RERANK", "1").strip().lower() not in ("0", "false", "no")


def _w():
    return (float(os.environ.get("YTA_ACTION_RERANK_W1", "1.0")),
            float(os.environ.get("YTA_ACTION_RERANK_W2", "0.15")),
            float(os.environ.get("YTA_ACTION_RERANK_W3", "0.25")))


def rerank_score(cosine_sim, shot_text="", seg_text="", seg_label=""):
    """final = w1*cosine + w2*(action_match) - w3*(action_conflict).
    When either side's action group is undecidable, returns cosine_sim unchanged."""
    if not rerank_enabled():
        return float(cosine_sim)
    w1, w2, w3 = _w()
    a_shot = action_of(shot_text)
    a_seg = action_of(seg_text, seg_label)
    match = 1.0 if (a_shot and a_seg and a_shot == a_seg) else 0.0
    conflict = 1.0 if (a_shot and a_seg and a_shot != a_seg) else 0.0
    return w1 * float(cosine_sim) + w2 * match - w3 * conflict

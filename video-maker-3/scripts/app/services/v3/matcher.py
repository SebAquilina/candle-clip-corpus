"""Vision + transcript matcher with Claude in the loop (the brief's core requirement).

The corpus stores, per clean window, a VISION caption (what is on screen) and the SPEECH
TRANSCRIPT (what is said) SEPARATELY. Matching a narration section to a clip should weigh
BOTH — and, because a generic caption ("two candles on a table") often under-rates a window
whose speaker is literally narrating the same action, the transcript carries real signal.

Two phases:
  1. build_worklist()  — OFFLINE shortlist. For each section it ranks every corpus moment by
     a COMBINED score = w_vision*cos(narration, vision) + w_transcript*cos(narration,
     transcript), then an action-aware re-rank, and keeps the top-K. Writes match_worklist.json.
  2. [Claude]           — reads the worklist and, for each section, picks the best-ordered
     clip(s) USING BOTH the vision and the transcript IN THE CONTEXT OF THE VIDEO TITLE
     (the worklist hands Claude exactly that), and writes match_decisions.json. This is the
     "tell the coworker to actually go through these" step — not a blind embedder.

The offline pass only SHORTLISTS (turns 671 windows into ~12 real candidates per section);
Claude makes the call. If Claude skips a section, the offline order stands, so the skill
still produces a video. The hard no-repeat rules live downstream in section_planner.
"""
from __future__ import annotations

import json
import os

from app.services.v3 import section_planner as sp


def _weights():
    return (float(os.environ.get("VM_W_VISION", "0.55")),
            float(os.environ.get("VM_W_TRANSCRIPT", "0.45")))


def _niche_expander(niche):
    try:
        from app.services.v3 import niche_vocab as nv
        return lambda t: nv.expand(t, niche=niche)
    except Exception:
        return lambda t: t


def _cos_matrix(q_vecs, s_vecs):
    """(n_q x n_s) cosine via numpy with safe zero-norm handling; None if numpy absent."""
    try:
        import numpy as np
    except Exception:
        return None
    Q = np.asarray(q_vecs, dtype=np.float32)
    S = np.asarray(s_vecs, dtype=np.float32)
    if Q.ndim != 2 or S.ndim != 2 or Q.shape[1] != S.shape[1]:
        return None
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    return Q @ S.T


def build_worklist(sections, library_index, embed_many, cosine,
                   project_title="", niche="", top_k=None, progress_cb=None):
    """Shortlist the top-K corpus moments per section by combined vision+transcript score.

    Returns the worklist (also the caller persists it). Each section item:
      {"index","start","end","dur","text",
       "candidates":[{"cand_id","url","id","channel","source_title","start","end","dur",
                      "label","vision","transcript","score","cos_vision","cos_transcript"}]}
    """
    moments = sp.build_moments(library_index)
    if not moments or not sections:
        return []
    top_k = int(top_k or os.environ.get("VM_SHORTLIST_K", "12"))
    w_vis, w_txt = _weights()
    exp = _niche_expander(niche)

    vis_texts = [exp(m["seg"].get("embed_text") or m["seg"].get("desc", "")) for m in moments]
    txt_texts = [(m["seg"].get("transcript") or "") for m in moments]   # natural speech: not expanded
    q_texts = [exp(s.get("text", "")) for s in sections]

    vis_vecs = embed_many(vis_texts)
    txt_vecs = embed_many(txt_texts)
    q_vecs = embed_many(q_texts)

    COSV = _cos_matrix(q_vecs, vis_vecs)
    COST = _cos_matrix(q_vecs, txt_vecs)

    try:
        from app.services.v3 import action_rerank as ar
        rerank = ar.rerank_score
    except Exception:
        rerank = lambda s, **_: s  # noqa: E731

    worklist = []
    for qi, sec in enumerate(sections):
        scored = []
        for i, m in enumerate(moments):
            if COSV is not None:
                cv = float(COSV[qi][i]); ct = float(COST[qi][i])
            else:
                cv = float(cosine(q_vecs[qi], vis_vecs[i]))
                ct = float(cosine(q_vecs[qi], txt_vecs[i])) if txt_texts[i] else 0.0
            combined = w_vis * cv + w_txt * ct
            score = rerank(combined, shot_text=sec.get("text", ""),
                           seg_text=m["seg"].get("embed_text", ""),
                           seg_label=m["seg"].get("label", ""))
            scored.append((score, cv, ct, i))
        scored.sort(key=lambda t: -t[0])
        cands = []
        for score, cv, ct, i in scored[:top_k]:
            m = moments[i]
            seg = m["seg"]
            cands.append({
                "cand_id": sp.cand_id(m),
                "url": m["url"], "id": m["id"], "channel": m.get("channel", ""),
                "source_title": m.get("title", ""),
                "start": round(float(seg.get("start", 0)), 2),
                "end": round(float(seg.get("end", 0)), 2),
                "dur": round(float(seg.get("end", 0)) - float(seg.get("start", 0)), 2),
                "label": seg.get("label", ""),
                "vision": seg.get("embed_text", seg.get("desc", "")),
                "transcript": seg.get("transcript", ""),
                "score": round(float(score), 3),
                "cos_vision": round(cv, 3), "cos_transcript": round(ct, 3),
            })
        worklist.append({
            "index": sec.get("index", qi), "start": sec.get("start"), "end": sec.get("end"),
            "dur": sec.get("dur"), "text": sec.get("text", ""), "candidates": cands,
        })
        if progress_cb and (qi % 10 == 0 or qi == len(sections) - 1):
            progress_cb(f"  shortlisted section {qi + 1}/{len(sections)} "
                        f"({len(cands)} candidates)")
    return worklist


def write_worklist(worklist, path, project_title="", niche=""):
    """Persist the worklist with a header that tells Claude exactly what to do."""
    doc = {
        "_instructions": (
            "For each section, choose the clip(s) whose VISION (what is on screen) AND "
            "TRANSCRIPT (what is said) best convey the narration, in the context of the video "
            "title. Prefer a clip that both SHOWS and (when spoken) DESCRIBES the action. "
            "Return match_decisions.json mapping each section index to an ORDERED list of "
            "cand_id (best first); list a few so short clips can be concatenated to cover the "
            "section. A clip may be reused at most twice total and never back-to-back — the "
            "assembler enforces this, so just rank by fit. Omit a section to accept the "
            "offline order."),
        "project_title": project_title, "niche": niche,
        "n_sections": len(worklist), "sections": worklist,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return path


def load_worklist(path):
    doc = json.load(open(path))
    return doc.get("sections", doc) if isinstance(doc, dict) else doc


def load_decisions(path):
    """Parse match_decisions.json -> {section_index(str): [cand_id, ...]}.

    Accepts {"0": ["id@1.2", ...], ...} or {"decisions": {...}} or
    [{"index":0,"picks":[...]}, ...]. Tolerant so Claude can write the natural shape."""
    if not path or not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    if isinstance(raw, dict) and "decisions" in raw:
        raw = raw["decisions"]
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            picks = v.get("picks", v) if isinstance(v, dict) else v
            if isinstance(picks, str):
                picks = [picks]
            out[str(k)] = [str(c) for c in (picks or [])]
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            idx = item.get("index", item.get("section"))
            picks = item.get("picks", item.get("cand_ids", []))
            if idx is not None:
                out[str(idx)] = [str(c) for c in (picks or [])]
    return out

"""Stage 4: per-second description of SURVIVING windows + transcript alignment,
and Stage 4b: drop any second whose description itself implies a human face/person.

For each surviving (purge-passed) window we walk it second-by-second. For each second we
record (a) a vision caption of the representative frame from a LOCAL VLM, and (b) the
transcript text spoken during that second — kept SEPARATE so the video-maker can weigh
them independently at match time. To keep CPU cost sane on near-static B-roll, the caption
is only recomputed on a scene change and otherwise carried forward.

Stage 4b (semantic face backstop): the pixel detector in `detectors.py` is the first line,
but a frame's caption sometimes names a person the detector missed ("a woman pours wax").
After describing, if ANY second's caption clearly implies a face/person, the whole window
is dropped (consistent with the drop-the-whole-window rule). Conservative word-boundary
matching so "hand", "manufacture", etc. don't trip it.

VLM backend is pluggable (REVAMP_VLM=blip|stub). BLIP runs offline on CPU.
"""
from __future__ import annotations
import os, re, functools
import cv2
import numpy as np

# ---- Stage 4b: face/person mentions in a caption -----------------------------
# Word-boundary terms that clearly imply a human face/person on screen. Deliberately
# excludes hands/fingers/arms (those are allowed B-roll) and avoids substring traps.
_FACE_WORDS = [
    r"face", r"faces", r"person", r"people", r"man", r"men", r"woman", r"women",
    r"boy", r"girl", r"child", r"children", r"kid", r"baby", r"lady", r"guy",
    r"he", r"she", r"his", r"her", r"him", r"someone", r"somebody", r"human",
    r"presenter", r"host", r"speaker", r"selfie", r"portrait", r"smiling",
    r"smile", r"beard", r"glasses", r"hair", r"eyes", r"mouth", r"head",
    r"model", r"influencer", r"vlogger", r"youtuber", r"talking",
]
_FACE_RE = re.compile(r"\b(" + "|".join(_FACE_WORDS) + r")\b", re.IGNORECASE)


def description_implies_face(text: str) -> bool:
    """True if a caption clearly implies a human face/person is on screen."""
    return bool(_FACE_RE.search(text or ""))


def face_terms_in(text: str) -> list[str]:
    return sorted({m.group(1).lower() for m in _FACE_RE.finditer(text or "")})


# ---- VLM captioner (pluggable, offline) --------------------------------------
class _StubCaptioner:
    name = "stub"
    def caption(self, frame_bgr) -> str:
        return ""


class _BlipCaptioner:
    name = "blip"
    def __init__(self):
        os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch  # noqa: F401
        m = os.environ.get("REVAMP_BLIP_MODEL", "Salesforce/blip-image-captioning-base")
        self.proc = BlipProcessor.from_pretrained(m)
        self.model = BlipForConditionalGeneration.from_pretrained(m)
        self.model.eval()
        from PIL import Image  # noqa: F401
        self._Image = Image

    def caption(self, frame_bgr) -> str:
        import torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._Image.fromarray(rgb)
        inputs = self.proc(img, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30, num_beams=1)
        return self.proc.decode(out[0], skip_special_tokens=True).strip()


@functools.lru_cache(maxsize=1)
def get_captioner():
    backend = os.environ.get("REVAMP_VLM", "blip").lower()
    if backend == "stub":
        return _StubCaptioner()
    try:
        return _BlipCaptioner()
    except Exception as e:
        print(f"[describe] VLM '{backend}' unavailable ({str(e)[:120]}); using stub")
        return _StubCaptioner()


# ---- scene-change detection (cheap, so we don't caption every static second) --
def _gray_small(frame, w=64):
    h = max(1, int(frame.shape[0] * w / max(1, frame.shape[1])))
    g = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def _scene_changed(prev, cur, thresh=None) -> bool:
    if prev is None:
        return True
    thr = float(os.environ.get("REVAMP_SCENE_DIFF", "10.0")) if thresh is None else thresh
    return float(np.mean(np.abs(cur - prev))) >= thr


# ---- transcript per second ---------------------------------------------------
def transcript_for_second(words: list, abs_t: float) -> str:
    """words: [(start,end,token), ...]. Return tokens overlapping [abs_t, abs_t+1)."""
    if not words:
        return ""
    lo, hi = abs_t, abs_t + 1.0
    toks = [w[2] for w in words if not (w[1] < lo or w[0] >= hi)]
    return " ".join(toks).strip()


# ---- main: describe one surviving window -------------------------------------
def describe_window(video_path: str, start_s: float, end_s: float,
                    transcript_words: list | None = None) -> dict:
    """Walk a survivor window second-by-second. Returns
    {seconds:[{t,abs_t,vision_desc,transcript_text,scene_change,face_mention,face_terms}],
     desc_face: bool, desc_face_terms:[...] }."""
    import detectors as D  # reuse the per-second frame access
    cap = get_captioner()
    words = transcript_words or []
    seconds = []
    prev_small = None
    last_desc = ""
    any_face_mention = False
    all_terms: set[str] = set()
    for k, t_mid, frame in D.iter_clean_seconds(video_path, start_s, end_s):
        small = _gray_small(frame)
        if _scene_changed(prev_small, small):
            last_desc = cap.caption(frame)
            changed = True
        else:
            changed = False
        prev_small = small
        terms = face_terms_in(last_desc)
        fm = bool(terms)
        if fm:
            any_face_mention = True
            all_terms.update(terms)
        seconds.append({
            "t": k, "abs_t": round(start_s + k, 2),
            "vision_desc": last_desc,
            "transcript_text": transcript_for_second(words, start_s + k),
            "scene_change": changed,
            "face_mention": fm, "face_terms": terms,
        })
    return {"seconds": seconds, "desc_face": any_face_mention,
            "desc_face_terms": sorted(all_terms)}


if __name__ == "__main__":
    import sys, json, subprocess
    # smoke test on a local clip: describe each second + show 4b verdict
    clip = sys.argv[1]
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", clip]).decode().strip())
    print("captioner:", get_captioner().name)
    res = describe_window(clip, 0.0, dur)
    for s in res["seconds"]:
        tag = "  <FACE-MENTION>" if s["face_mention"] else ""
        print(f"  t={s['t']:>2}s desc={s['vision_desc']!r}{tag}")
    print(f"\ndesc_face (drops window 4b): {res['desc_face']}  terms={res['desc_face_terms']}")

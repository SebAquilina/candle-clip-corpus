#!/usr/bin/env python3
"""clip_checks.py — deterministic per-clip content gates (Rules v22.3 / v22.4 / v22.6).

Catches HARD-rule violations PROGRAMMATICALLY, as a backstop to any vision review, by
sampling frames and running cheap deterministic detectors:

  * OVERLAY / BURNED-IN TEXT — captions, subscribe banners, lower-thirds, chyrons,
    on-screen titles. Detected with Tesseract OCR (pytesseract).
  * HUMAN FACES / TALKING HEADS — ANY human face on screen. The user's HARD rule is
    ZERO faces in any clip (not merely "no dominant talking head"). Detected with a
    backend chain: OpenCV YuNet (cv2.FaceDetectorYN) -> OpenCV DNN ResNet-10 SSD ->
    Haar cascades (always-present fallback). YuNet/SSD detect profile, tilted, small,
    partially-occluded and multiple faces far better than Haar.

These run in two places:
  1. Per-clip at render time (render_video.py): a clip that trips either detector is
     UNUSABLE — the renderer re-sources/borrows a different clip, so the offending
     footage never reaches the timeline.
  2. As a non-skippable final gate (validate_render.py): the finished mp4 is scanned
     end-to-end; text or a face anywhere FAILS the video.

ZERO-TOLERANCE FACE MODE (YTA_NO_FACES=1, default ON): ANY detected face — any size, on
even a single sampled frame — makes a clip unusable and fails the final gate at >=1 face
frame, with denser frame sampling so a brief face is not missed. Set YTA_NO_FACES=0 for
the legacy "dominant talking-head only" behaviour.

Dependencies: opencv-python-headless (cv2; Haar cascades are bundled). The YuNet ONNX
model (~233 KB) is read from assets/face_detection_yunet_2023mar.onnx, $WS/assets, or
$YTA_FACE_MODEL; if absent, detection degrades to SSD (if its files are present) then to
Haar — face detection is NEVER silently disabled. pytesseract + the `tesseract` binary
power the text detector. If cv2 is missing, FACE_AVAILABLE/TEXT_AVAILABLE report False and
the FINAL GATE treats "cannot verify" as a blocker (never a silent pass) — see
validate_render.py. Set VM_DISABLE_CLIP_CHECKS=1 to bypass (refused when YTA_REQUIRE_GATES=1).
"""
from __future__ import annotations
import os

# --- mode -------------------------------------------------------------------
# Zero-tolerance face mode: ANY face (any size, a single frame) -> reject. Default ON.
NO_FACES = os.environ.get("YTA_NO_FACES", "1").strip().lower() not in ("0", "false", "no", "")

# --- tunables (env-overridable) ---------------------------------------------
N_FRAMES        = int(os.environ.get("VM_CHECK_FRAMES", "24" if NO_FACES else "12"))  # frames sampled per clip
OCR_MAX_W       = int(os.environ.get("VM_OCR_MAX_W", "960"))        # downscale width for speed
TEXT_MIN_CONF   = float(os.environ.get("VM_TEXT_MIN_CONF", "55"))   # min OCR confidence to count a word
TEXT_MIN_CHARS  = int(os.environ.get("VM_TEXT_MIN_CHARS", "4"))     # min chars in a word to count
TEXT_AREA_FRAC  = float(os.environ.get("VM_TEXT_AREA_FRAC", "0.012"))   # confident text covering >=1.2% of a frame = "has text"
TEXT_FRAME_FRAC = float(os.environ.get("VM_TEXT_FRAME_FRAC", "0.25"))   # ... on >=25% of sampled frames = overlay
# Face thresholds. In zero-tolerance mode the area/frame fractions collapse to "any box,
# any frame"; the legacy "dominant talking-head" values apply only when YTA_NO_FACES=0.
FACE_AREA_FRAC  = float(os.environ.get("VM_FACE_AREA_FRAC", "0.045"))   # legacy: face box >=4.5% of frame = "dominant"
FACE_FRAME_FRAC = float(os.environ.get("VM_FACE_FRAME_FRAC", "0.45"))   # legacy: dominant face on >=45% of frames
FACE_SCORE      = float(os.environ.get("YTA_FACE_SCORE", "0.6"))       # min detector confidence to count a face (YuNet/SSD)
FACE_MIN_AREA_FRAC = float(os.environ.get("YTA_FACE_MIN_AREA_FRAC", "0.0"))  # zero-tol: min box area frac to count (0 = any size)
NO_FACE_CLIP_FRAMES = int(os.environ.get("YTA_NO_FACE_CLIP_FRAMES", "1"))    # zero-tol: this many face frames -> clip unusable
GATE_TEXT_HITS  = int(os.environ.get("VM_GATE_TEXT_HITS", "2"))     # whole-file gate: this many texted frames = fail
GATE_FACE_HITS  = int(os.environ.get("VM_GATE_FACE_HITS", "1" if NO_FACES else "3"))  # whole-file gate: this many face frames = fail
GATE_FPS        = float(os.environ.get("VM_GATE_FPS", "3.0" if NO_FACES else "0.7"))  # gate sampling cadence
GATE_MAX_FRAMES = int(os.environ.get("VM_GATE_MAX_FRAMES", "3000" if NO_FACES else "800"))  # safety cap on gate frames
REQUIRE_STRONG_FACE = os.environ.get("YTA_REQUIRE_STRONG_FACE", "0").strip().lower() in ("1", "true", "yes")
SSD_CONF        = float(os.environ.get("YTA_FACE_SSD_CONF", "0.5"))
# Haar recall is widened in zero-tolerance mode (more sensitive: more false positives, the safe direction).
_HAAR_NEIGHBORS = int(os.environ.get("VM_HAAR_MIN_NEIGHBORS", "3" if NO_FACES else "6"))
_HAAR_MINSIZE   = int(os.environ.get("VM_HAAR_MIN_SIZE", "20" if NO_FACES else "32"))

DISABLED = os.environ.get("VM_DISABLE_CLIP_CHECKS", "").strip().lower() in ("1", "true", "yes")

# --- optional deps ----------------------------------------------------------
try:
    import cv2
    import numpy as np
    _CV2 = True
except Exception:
    _CV2 = False
try:
    import pytesseract
    try:
        pytesseract.get_tesseract_version()
        _TESS = True
    except Exception:
        _TESS = False
except Exception:
    _TESS = False


# --- face backend chain: YuNet -> SSD -> Haar -------------------------------
def _model_path():
    """Locate the YuNet ONNX (bundled assets/, $WS/assets, or $YTA_FACE_MODEL)."""
    p = os.environ.get("YTA_FACE_MODEL", "")
    if p and os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "assets", "face_detection_yunet_2023mar.onnx"),
              os.path.join(os.environ.get("WS", ""), "assets", "face_detection_yunet_2023mar.onnx")):
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return ""


class _YuNet:
    name = "yunet"
    def __init__(self, model):
        self.det = cv2.FaceDetectorYN_create(model, "", (320, 320), FACE_SCORE, 0.3, 5000)
    def detect(self, frame):
        h, w = frame.shape[:2]
        try:
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(frame)
        except Exception:
            return []
        if faces is None:
            return []
        out = []
        for r in faces:
            x, y, bw, bh = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            sc = float(r[14]) if len(r) > 14 else 1.0
            if bw > 0 and bh > 0:
                out.append((x, y, bw, bh, sc))
        return out


class _SSD:
    name = "ssd"
    def __init__(self, prototxt, caffemodel):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    def detect(self, frame):
        h, w = frame.shape[:2]
        try:
            self.net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)))
            det = self.net.forward()
        except Exception:
            return []
        out = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < SSD_CONF:
                continue
            x1, y1, x2, y2 = (det[0, 0, i, 3:7] * [w, h, w, h]).tolist()
            out.append((x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1), conf))
        return out


class _Haar:
    name = "haar"
    def __init__(self):
        self.frontal = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        self.profile = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml"))
    def detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        ms = (_HAAR_MINSIZE, _HAAR_MINSIZE)
        out = []
        for casc in (self.frontal, self.profile):
            if casc is None or casc.empty():
                continue
            for (x, y, fw, fh) in casc.detectMultiScale(gray, scaleFactor=1.1,
                                                        minNeighbors=_HAAR_NEIGHBORS, minSize=ms):
                out.append((float(x), float(y), float(fw), float(fh), 1.0))
        if self.profile is not None and not self.profile.empty():
            for (x, y, fw, fh) in self.profile.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1,
                                                                minNeighbors=_HAAR_NEIGHBORS, minSize=ms):
                out.append((float(x), float(y), float(fw), float(fh), 1.0))
        return out


_FACE_BACKEND = None


def _build_backend():
    model = _model_path()
    if model and hasattr(cv2, "FaceDetectorYN_create"):
        try:
            return _YuNet(model)
        except Exception:
            pass
    pj = os.environ.get("YTA_FACE_SSD_PROTOTXT", "")
    cm = os.environ.get("YTA_FACE_SSD_MODEL", "")
    if pj and cm and os.path.exists(pj) and os.path.exists(cm):
        try:
            return _SSD(pj, cm)
        except Exception:
            pass
    return _Haar()


def _get_backend():
    global _FACE_BACKEND
    if _FACE_BACKEND is None:
        _FACE_BACKEND = _build_backend()
    return _FACE_BACKEND


# Resolve the backend at import so FACE_AVAILABLE reflects it. Haar is always available
# with cv2, so face detection is never silently off; YTA_REQUIRE_STRONG_FACE=1 turns a
# fall-through-to-Haar into an explicit "cannot verify" blocker (exit 8).
if _CV2:
    try:
        _BACKEND_NAME = _get_backend().name
    except Exception:
        _BACKEND_NAME = "haar"
else:
    _BACKEND_NAME = None

FACE_AVAILABLE = _CV2 and (not REQUIRE_STRONG_FACE or _BACKEND_NAME in ("yunet", "ssd"))
TEXT_AVAILABLE = _CV2 and _TESS


def availability() -> dict:
    return {"cv2": _CV2, "tesseract": _TESS,
            "text_available": TEXT_AVAILABLE, "face_available": FACE_AVAILABLE,
            "face_backend": _BACKEND_NAME, "no_faces": NO_FACES, "disabled": DISABLED}


# --- frame access -----------------------------------------------------------
def _downscale(frame, max_w=OCR_MAX_W):
    h, w = frame.shape[:2]
    if w > max_w:
        s = max_w / float(w)
        frame = cv2.resize(frame, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)
    return frame


def sample_frames(path, n=N_FRAMES):
    """Up to n evenly-spaced frames from the clip (for per-clip checks)."""
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 1:
            while len(out) < n:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                out.append(fr)
            return out
        idxs = sorted({int(total * (i + 0.5) / n) for i in range(n)})
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok and fr is not None:
                out.append(fr)
    finally:
        cap.release()
    return out


def iter_frames_at_fps(path, fps=GATE_FPS, max_frames=GATE_MAX_FRAMES):
    """Yield (timestamp_sec, frame) across the WHOLE video at ~fps cadence (for the gate)."""
    cap = cv2.VideoCapture(str(path))
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if src_fps <= 0:
            src_fps = 30.0
        step = max(1, int(round(src_fps / max(0.05, fps))))
        idx, yielded = 0, 0
        while total <= 0 or idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            yield (idx / src_fps, fr)
            yielded += 1
            if yielded >= max_frames:
                break
            idx += step
    finally:
        cap.release()


# --- text detection ---------------------------------------------------------
def frame_text_area_frac(frame) -> float:
    """Fraction of frame area covered by CONFIDENT OCR words. PSM 11 = sparse text
    (good for scattered overlays/captions). Grayscale + downscale for speed."""
    fr = _downscale(frame)
    h, w = fr.shape[:2]
    area = float(h * w) or 1.0
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    try:
        data = pytesseract.image_to_data(gray, config="--psm 11",
                                          output_type=pytesseract.Output.DICT)
    except Exception:
        return 0.0
    covered = 0.0
    for i in range(len(data.get("text", []))):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf >= TEXT_MIN_CONF and len(txt) >= TEXT_MIN_CHARS and any(c.isalnum() for c in txt):
            covered += float(data["width"][i]) * float(data["height"][i])
    return covered / area


def clip_has_overlay_text(path, n=N_FRAMES):
    """True if confident text covers >= TEXT_AREA_FRAC of the frame on
    >= TEXT_FRAME_FRAC of sampled frames. Returns (bool, reason)."""
    frames = sample_frames(path, n)
    if not frames:
        return False, "no frames"
    hits, max_frac = 0, 0.0
    for f in frames:
        frac = frame_text_area_frac(f)
        max_frac = max(max_frac, frac)
        if frac >= TEXT_AREA_FRAC:
            hits += 1
    return (hits / len(frames) >= TEXT_FRAME_FRAC,
            f"text on {hits}/{len(frames)} frames (max {max_frac*100:.2f}% area)")


# --- face / talking-head detection ------------------------------------------
def _faces_on(frame):
    """(boxes, area) for a frame: score-filtered face boxes on the downscaled frame.
    boxes = [(x, y, w, h, score), ...] from the active backend (YuNet/SSD/Haar)."""
    fr = _downscale(frame)
    h, w = fr.shape[:2]
    area = float(h * w) or 1.0
    boxes = [b for b in _get_backend().detect(fr) if b[4] >= FACE_SCORE]
    return boxes, area


def frame_dominant_face_frac(frame) -> float:
    """Largest detected face box as a fraction of the frame area (0 if none).
    PRESERVED API — used by backfill_signals.py and the legacy gate."""
    boxes, area = _faces_on(frame)
    return (max(b[2] * b[3] for b in boxes) / area) if boxes else 0.0


def frame_face_hit(frame) -> bool:
    """Does this frame count as a face hit under the ACTIVE mode?
    Zero-tolerance: ANY face whose area >= FACE_MIN_AREA_FRAC (0 = any size).
    Legacy: a DOMINANT face (>= FACE_AREA_FRAC)."""
    boxes, area = _faces_on(frame)
    if not boxes:
        return False
    if NO_FACES:
        return any((b[2] * b[3]) / area >= FACE_MIN_AREA_FRAC for b in boxes)
    return (max(b[2] * b[3] for b in boxes) / area) >= FACE_AREA_FRAC


def clip_has_talking_head(path, n=N_FRAMES):
    """Zero-tolerance: True if ANY face appears on >= NO_FACE_CLIP_FRAMES sampled frames
    (default 1). Legacy: a dominant face on >= FACE_FRAME_FRAC of frames. Returns (bool, reason)."""
    frames = sample_frames(path, n)
    if not frames:
        return False, "no frames"
    hits, max_frac = 0, 0.0
    for f in frames:
        boxes, area = _faces_on(f)
        frac = (max(b[2] * b[3] for b in boxes) / area) if boxes else 0.0
        max_frac = max(max_frac, frac)
        hit = (len(boxes) >= 1 and any((b[2] * b[3]) / area >= FACE_MIN_AREA_FRAC for b in boxes)) \
            if NO_FACES else (frac >= FACE_AREA_FRAC)
        if hit:
            hits += 1
    if NO_FACES:
        return (hits >= NO_FACE_CLIP_FRAMES,
                f"face on {hits}/{len(frames)} frames (backend={_BACKEND_NAME}, zero-tolerance)")
    return (hits / len(frames) >= FACE_FRAME_FRAC,
            f"dominant face on {hits}/{len(frames)} frames (max {max_frac*100:.2f}% of frame)")


# --- per-clip verdict (used by render_video.py) -----------------------------
def check_clip(path, n=N_FRAMES) -> dict:
    """Per-clip verdict. {'text', 'talking_head', 'ok', 'available', reasons...}.
    On any internal error, returns ok=True with available flags so render is never
    bricked — the FINAL GATE (validate_render) is the hard guarantee. (render_video
    itself fails CLOSED in zero-tolerance mode: see _content_ok.)"""
    if DISABLED:
        return {"text": False, "talking_head": False, "ok": True, "available": False,
                "text_reason": "disabled", "head_reason": "disabled"}
    text = head = False
    tr = hr = "unavailable"
    try:
        if TEXT_AVAILABLE:
            text, tr = clip_has_overlay_text(path, n)
        if FACE_AVAILABLE:
            head, hr = clip_has_talking_head(path, n)
    except Exception as e:
        return {"text": False, "talking_head": False, "ok": True, "available": False,
                "text_reason": f"error: {e}", "head_reason": f"error: {e}"}
    return {"text": bool(text), "talking_head": bool(head),
            "ok": not (text or head),
            "available": TEXT_AVAILABLE and FACE_AVAILABLE,
            "text_reason": tr, "head_reason": hr}


# --- whole-file scans (used by validate_render.py) --------------------------
def scan_video_text(path, fps=GATE_FPS):
    """Scan the whole file. Returns list of (timestamp, area_frac) for frames whose
    confident-text coverage >= TEXT_AREA_FRAC."""
    hits = []
    for t, fr in iter_frames_at_fps(path, fps):
        frac = frame_text_area_frac(fr)
        if frac >= TEXT_AREA_FRAC:
            hits.append((t, frac))
    return hits


def scan_video_talking_head(path, fps=GATE_FPS):
    """Scan the whole file. Returns list of (timestamp, face_frac) for frames that are a
    face hit under the active mode (zero-tolerance: ANY face; legacy: dominant face)."""
    hits = []
    for t, fr in iter_frames_at_fps(path, fps):
        boxes, area = _faces_on(fr)
        if not boxes:
            continue
        frac = max(b[2] * b[3] for b in boxes) / area
        hit = any((b[2] * b[3]) / area >= FACE_MIN_AREA_FRAC for b in boxes) if NO_FACES \
            else (frac >= FACE_AREA_FRAC)
        if hit:
            hits.append((t, frac))
    return hits


if __name__ == "__main__":
    import sys, json
    print(json.dumps(availability()))
    for p in sys.argv[1:]:
        print(p, json.dumps(check_clip(p)))

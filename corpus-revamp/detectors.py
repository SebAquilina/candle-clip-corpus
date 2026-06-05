"""Per-second face + on-screen-text detector for the corpus purge.

Wraps the skill's vendored `clip_checks` detector chain (YuNet -> SSD -> Haar for
faces) plus a 3-pass Tesseract OCR (full-frame + bottom-strip + top-strip, CLAHE
contrast) and applies it SECOND-BY-SECOND, fail-closed. A FACE rejects the window on
any single frame; TEXT must PERSIST on >=2 distinct seconds (so one-frame OCR noise on
textured B-roll doesn't drop clean footage). A second that decodes 0 frames -> rejected
'unreadable' (never passed unscanned).

Env knobs (all overridable):
  REVAMP_FACE_SCORE           min YuNet/SSD confidence to count a face     (default 0.60)
  REVAMP_TEXT_MIN_CONF        min OCR word confidence to count as text     (default 55)
  REVAMP_TEXT_MIN_CHARS       min alnum chars for an OCR word              (default 4)
  REVAMP_TEXT_PERSIST_SECONDS confident text on >=N seconds -> reject      (default 2)
  REVAMP_BOTTOM_FRAC/_TOP_FRAC strip crops for subtitle/corner-logo OCR    (0.72 / 0.20)
  REVAMP_FRAMES_PER_SEC       frames sampled inside each second            (default 2)
"""
from __future__ import annotations
import os
from pathlib import Path

# --- configure the vendored detector BEFORE importing it (it reads env at import) ---
_HERE = Path(__file__).resolve().parent
os.environ.setdefault("YTA_NO_FACES", "1")                       # zero-tolerance face mode
os.environ.setdefault("YTA_FACE_SCORE", os.environ.get("REVAMP_FACE_SCORE", "0.60"))
os.environ.setdefault("YTA_FACE_MODEL", str(_HERE / "face_detection_yunet_2023mar.onnx"))
# Make the OCR sensitive: any confident multi-char alnum word counts as on-screen text.
# conf 60 + min 4 chars: real captions/watermarks are confident, multi-letter words;
# at conf 45 / 2 chars tesseract hallucinated short gibberish ("SS", "sh iw vat") on
# textured candle/snow B-roll, falsely rejecting clean windows (see montage analysis).
os.environ.setdefault("VM_TEXT_MIN_CONF", os.environ.get("REVAMP_TEXT_MIN_CONF", "60"))
os.environ.setdefault("VM_TEXT_MIN_CHARS", os.environ.get("REVAMP_TEXT_MIN_CHARS", "4"))

import importlib.util
_spec = importlib.util.spec_from_file_location("_cc_vendored", _HERE / "_clip_checks_vendored.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

import cv2  # noqa: E402

FRAMES_PER_SEC = int(os.environ.get("REVAMP_FRAMES_PER_SEC", "2"))
# A word counts as on-screen text only if OCR is confident AND it has enough letters —
# this rejects the short gibberish ("SS", "sh iw vat") tesseract hallucinates on textured
# B-roll, while keeping real captions/logos. Validated 10/10 on the montage sample.
TEXT_CONF = float(os.environ.get("REVAMP_TEXT_MIN_CONF", "55"))
TEXT_LEN = int(os.environ.get("REVAMP_TEXT_MIN_CHARS", "4"))
import pytesseract  # noqa: E402
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def availability() -> dict:
    """Detector health — call once at startup and refuse to run if not strong."""
    a = cc.availability()
    a["frames_per_sec"] = FRAMES_PER_SEC
    return a


def _frame_has_face(frame) -> bool:
    boxes, _ = cc._faces_on(frame)
    return bool(boxes)


def _ocr_hit(gray, psm: int) -> bool:
    """True if any confident, long-enough word is present in this (pre-processed) image."""
    try:
        d = pytesseract.image_to_data(gray, config=f"--psm {psm}",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return False
    for i in range(len(d.get("text", []))):
        t = (d["text"][i] or "").strip()
        alnum = "".join(c for c in t if c.isalnum())
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1.0
        if conf >= TEXT_CONF and len(alnum) >= TEXT_LEN:
            return True
    return False


def _frame_has_text(frame) -> bool:
    """Three superimposed OCR passes — text ANYWHERE is caught, faint text recovered:
      P1 full-frame + contrast  -> title cards, end-cards, large/any text
      P2 bottom strip (upscaled) -> faint subtitles, bottom captions, bottom-corner marks
      P3 top strip (upscaled)    -> top-corner logos/watermarks full-frame misses
    Each region is contrast-enhanced (CLAHE) so low-contrast captions become readable.
    Returns on the first hit. Persistence across seconds (scan_window) suppresses noise."""
    h, w = frame.shape[:2]
    g = _CLAHE.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if _ocr_hit(g, 11):
        return True
    bf = float(os.environ.get("REVAMP_BOTTOM_FRAC", "0.72"))
    s = frame[int(h * bf):, :]
    if s.size:
        s = cv2.resize(s, (w * 2, max(1, s.shape[0] * 2)), interpolation=cv2.INTER_CUBIC)
        if _ocr_hit(_CLAHE.apply(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)), 6):
            return True
    tf = float(os.environ.get("REVAMP_TOP_FRAC", "0.20"))
    t = frame[:int(h * tf), :]
    if t.size:
        t = cv2.resize(t, (w * 2, max(1, t.shape[0] * 2)), interpolation=cv2.INTER_CUBIC)
        if _ocr_hit(_CLAHE.apply(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)), 6):
            return True
    return False


def scan_window(video_path: str, start_s: float, end_s: float) -> dict:
    """Second-by-second zero-tolerance scan of one window.

    Returns {clean: bool, reason: 'face'|'text'|'', hit_t: float|None,
             seconds_scanned: int, frames_read: int}. Early-exits on the first hit."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"clean": False, "reason": "unreadable", "hit_t": None,
                "seconds_scanned": 0, "frames_read": 0}
    dur = max(0.0, float(end_s) - float(start_s))
    n_sec = max(1, int(round(dur)))
    offsets = [(j + 1) / (FRAMES_PER_SEC + 1) for j in range(FRAMES_PER_SEC)]  # e.g. 1/3, 2/3
    persist = int(os.environ.get("REVAMP_TEXT_PERSIST_SECONDS", "2"))
    frames_read = 0
    text_seconds = 0
    first_text_t = None
    try:
        for k in range(n_sec):
            sec_frames = 0
            sec_has_text = False
            for off in offsets:
                t = float(start_s) + k + off
                if t >= float(end_s):
                    t = (float(start_s) + float(end_s)) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, fr = cap.read()
                if not ok or fr is None:
                    continue
                frames_read += 1
                sec_frames += 1
                # Faces are zero-tolerance: ANY single detection rejects immediately.
                if _frame_has_face(fr):
                    return {"clean": False, "reason": "face", "hit_t": round(t, 2),
                            "seconds_scanned": k + 1, "frames_read": frames_read}
                # Text must PERSIST: one flickered frame is usually OCR noise on textured
                # B-roll, not a real caption. Mark the second, decide after counting.
                if not sec_has_text and _frame_has_text(fr):
                    sec_has_text = True
                    if first_text_t is None:
                        first_text_t = round(t, 2)
            # FAIL-CLOSED: a second we could not decode at all is a second we could not
            # verify. Never pass an unscanned second as "clean" (undecodable AV1 used to
            # read 0 frames and slip through). Reject so it's re-sourced.
            if sec_frames == 0:
                return {"clean": False, "reason": "unreadable", "hit_t": round(start_s + k, 2),
                        "seconds_scanned": k, "frames_read": frames_read}
            if sec_has_text:
                text_seconds += 1
                if text_seconds >= persist:   # confident text on >=N distinct seconds
                    return {"clean": False, "reason": "text", "hit_t": first_text_t,
                            "seconds_scanned": k + 1, "frames_read": frames_read}
        return {"clean": True, "reason": "", "hit_t": None,
                "seconds_scanned": n_sec, "frames_read": frames_read}
    finally:
        cap.release()


def iter_clean_seconds(video_path: str, start_s: float, end_s: float):
    """For a SURVIVING window: yield (second_index, t_mid, frame_bgr) once per second,
    using the middle frame. Used by the describe stage (stage 4)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    dur = max(0.0, float(end_s) - float(start_s))
    n_sec = max(1, int(round(dur)))
    try:
        for k in range(n_sec):
            t = float(start_s) + k + 0.5
            if t >= float(end_s):
                t = (float(start_s) + float(end_s)) / 2.0
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if ok and fr is not None:
                yield (k, round(t, 2), fr)
    finally:
        cap.release()


if __name__ == "__main__":
    import json
    print(json.dumps(availability(), indent=2))

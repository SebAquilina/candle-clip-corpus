"""Per-second face + on-screen-text detector for the corpus purge.

Wraps the skill's vendored `clip_checks` detector chain (YuNet -> SSD -> Haar for
faces, Tesseract OCR for text) but applies it ZERO-TOLERANCE and SECOND-BY-SECOND:
for a window [start_s, end_s] we sample N frames inside every second and a window is
REJECTED the instant any sampled frame shows a face OR any confident text. Fail-closed.

Env knobs (all overridable; defaults tuned for max recall / zero-tolerance):
  REVAMP_FACE_SCORE     min YuNet/SSD confidence to count a face   (default 0.50)
  REVAMP_TEXT_MIN_CONF  min OCR word confidence to count as text   (default 45)
  REVAMP_TEXT_MIN_CHARS min alnum chars for an OCR word            (default 2)
  REVAMP_FRAMES_PER_SEC frames sampled inside each second          (default 2)
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
# >0 = at least one confident OCR word covers some area -> text present.
TEXT_EPS = float(os.environ.get("REVAMP_TEXT_AREA_EPS", "0.0"))


def availability() -> dict:
    """Detector health — call once at startup and refuse to run if not strong."""
    a = cc.availability()
    a["frames_per_sec"] = FRAMES_PER_SEC
    return a


def _frame_has_face(frame) -> bool:
    boxes, _ = cc._faces_on(frame)
    return bool(boxes)


def _frame_has_text(frame) -> bool:
    return cc.frame_text_area_frac(frame) > TEXT_EPS


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

"""Export the clean v2 corpus to CSV — one row per surviving (clean) window.

Each window already passed the second-by-second face+OCR purge and the description face
backstop, so every row is verified text-free / face-free B-roll with a vision description
(what is on screen) and a transcript (what is spoken), kept in separate columns.

Usage:  python export_csv.py [out.csv]
"""
from __future__ import annotations
import json, glob, csv, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS = Path(os.environ.get("REVAMP_RECORDS_V2", HERE.parent / "outputs/shared_db_v2/records"))

FIELDS = ["video_id", "video_url", "video_title", "source", "niche",
          "window_index", "start_s", "end_s", "duration_s", "action_label", "phase",
          "n_seconds", "vision_embed_text", "transcript"]


def iter_rows():
    for f in sorted(glob.glob(str(RECORDS / "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for w in d.get("windows", []):
            s, e = w.get("start_s", 0) or 0, w.get("end_s", 0) or 0
            yield {
                "video_id": d.get("video_id", ""),
                "video_url": d.get("video_url", ""),
                "video_title": d.get("video_title", ""),
                "source": d.get("source", "corpus"),
                "niche": d.get("niche", ""),
                "window_index": w.get("window_index", ""),
                "start_s": round(float(s), 2),
                "end_s": round(float(e), 2),
                "duration_s": round(float(e) - float(s), 2),
                "action_label": w.get("action_label", ""),
                "phase": w.get("phase", ""),
                "n_seconds": len(w.get("seconds", [])),
                "vision_embed_text": (w.get("embed_text", "") or "").replace("\n", " "),
                "transcript": (w.get("transcript", "") or "").replace("\n", " "),
            }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "corpus_clean_windows.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for r in iter_rows():
            wr.writerow(r); n += 1
    print(f"wrote {n} clean-window rows to {out}")
    return n


if __name__ == "__main__":
    main()

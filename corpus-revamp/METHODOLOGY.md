# How the clean candle corpus was built — full methodology

This document records, in detail, every technique and decision used to turn a
face/text-contaminated corpus into a **clean, verified corpus of 671 text-free /
face-free B-roll windows** with per-second vision + transcript descriptions.

> Status at completion: **671 clean windows** from 399 source videos
> (original corpus 196 · watch-later list 96 · title-searched 379),
> **5,333 per-second descriptions**, every record JSON-valid, exported to
> `corpus_clean_windows.csv`.

---

## 1. The problem we were solving

Rendered videos kept showing **human faces and on-screen text** (burned-in captions,
title cards, app watermarks like InShot/TikTok). The root cause was **not** the renderer —
it was the **corpus**:

- The old corpus (`outputs/shared_db/`) was built from **YouTube storyboard sprites** —
  low-resolution thumbnail mosaics — on 250 of 257 videos (`"source": "storyboard"`).
  The classifier **never saw full-resolution frames**, so it physically could not detect
  small corner watermarks, brief title cards, or medium-distance / profile faces.
- Flags were **per ~8-second window** (one `action_label` each), but contamination is
  usually **sub-window**: a single second of caption, or a presenter turning to camera.
- The skill's per-clip QC only sampled ~24 frames and otherwise trusted those coarse flags.

Net: contamination was baked into the "usable" set at ingest and leaked into final videos.

The fix: **go back to the real full-resolution footage and re-analyse every window
second-by-second**, then rebuild the corpus from only what is verifiably clean.

---

## 2. The detection stack (OpenCV + OCR)

Two independent detectors run on the real frames, reusing the skill's vendored
`clip_checks` chain (`_clip_checks_vendored.py`):

### Faces — OpenCV
- **YuNet** (`cv2.FaceDetectorYN`, ONNX model `face_detection_yunet_2023mar.onnx`, vendored)
  is the primary detector — it catches profile, tilted, small and medium-distance faces.
- **SSD ResNet-10** and **Haar cascades** are fallbacks so face detection is **never
  silently disabled**.
- Confidence threshold tuning was critical (see §4): **`REVAMP_FACE_SCORE = 0.60`**
  (the YuNet default). At an earlier 0.50 it **false-positived on hands and silicone
  moulds** (we verified a 0.54-confidence "face" that was actually a hand on a mould);
  real faces score 0.8+.

### Text — OCR (Tesseract via pytesseract)
- Tesseract reads on-screen text. A word counts only if it is **confident** and **long
  enough** (`conf >= 55`, `>= 4 alphanumeric chars`) — this rejects the short gibberish
  ("SS", "(FS", "sh iw vat ae") Tesseract hallucinates on busy/textured B-roll (snow, wax
  swirls, wood grain).

---

## 3. The v3 per-second pipeline (`reclassify.py`)

For **every usable window** of every video:

1. **Fetch** the source video once at ≤720p **H.264** into deletable scratch (`fetch.py`).
2. **Purge** (`detectors.scan_window`) — sample frames inside **every second** of the
   window and run face + OCR. **Zero-tolerance, fail-closed, drop-the-whole-window**: any
   face or persistent text in any second rejects the entire ~8-second window. Faces reject
   immediately; text must persist (§4).
3. **Describe survivors** (`describe.py`) — walk each surviving window second-by-second:
   - **Vision**: a **BLIP** caption (local, offline VLM) of the representative frame,
     recomputed only on a **scene change** (cheap frame-diff) and carried across static
     B-roll, so we don't re-caption identical seconds.
   - **Transcript**: the words spoken during that second — **YouTube captions** (json3 via
     yt-dlp) when present, else **faster-whisper** on the audio.
   - Vision and transcript are stored **separately** so a matcher can weigh them
     independently.
4. **Stage 4b — semantic face backstop** (`describe.py`): if any second's **caption itself**
   names a person ("a woman pours…", "man talking to camera"), the window is dropped even
   if the pixel detector missed the face. Word-boundary matching; `hands`/`fingers` are
   allowed.
5. **Write** a per-second record to `outputs/shared_db_v2/`, plus a window-level roll-up
   (`embed_text` = unique vision phrases, `transcript` = joined speech) for drop-in matcher
   compatibility, a `by_label/` index, and a `rejected/` quarantine with the drop reason.

Decisions baked in (chosen by the user): **grind to completion**, **local offline VLM
(BLIP)**, **drop the whole window** on any hit, **zero-tolerance / fail-closed**, plus the
**stage-4b** description-based face drop.

---

## 4. Making the text detector precise — the key technique

The naive "any confident OCR word on any single frame → reject" was **wrong in both
directions**, which we proved by pulling 10 text-rejected frames into a labelled montage:

- **False positives (~half the sample)**: Tesseract hallucinated short gibberish on
  textured footage (a **snowy scene** OCR'd as `"sh iw vat ae by +i 7 we"`; candle frames
  as `"SS"`, `"(FS"`). These were **clean footage being wrongly dropped** — gutting the corpus.
- **False negatives** appeared when we over-tightened (conf 60 / 4 chars, single full-frame
  pass): it then **missed** a faint bottom **subtitle**, a **"METHOD #1…" caption**, and a
  small top-left **"Osceola Library" logo**.

The fix was **three superimposed OCR passes + temporal persistence**, validated **10/10**
on the montage:

### (a) Region crops with contrast enhancement
Real text lives in predictable places, so we OCR three regions, each **contrast-enhanced
with CLAHE** so faint/low-contrast text becomes readable:
- **P1 — full frame** (+ CLAHE): title cards, end-cards, large/any text.
- **P2 — bottom strip** (bottom ~28%, **upscaled 2×**, CLAHE, line-mode PSM 6): faint
  **subtitles** and bottom captions. *This was the breakthrough for subtitles* — on the
  faint-subtitle frame, full-frame OCR read nothing, but the bottom-strip pass read
  "This"(95) … "everything"(93) … "need"(96) **consistently across frames**.
- **P3 — top strip** (top ~20%, upscaled 2×, CLAHE): top-corner **logos/watermarks** that
  full-frame missed (e.g. the "Osceola Library" logo).

A frame "has text" if **any** pass fires. Bottom-corner marks (InShot etc.) fall in P2;
top-corner in P3.

### (b) Persistence — "slow it down so it's not just noise"
A real caption/watermark sits in the **same place across many seconds**; OCR noise
**flickers** on a single frame. So a window is rejected for text only when confident text
appears on **≥ 2 distinct seconds** (`REVAMP_TEXT_PERSIST_SECONDS = 2`). This is what
separates a genuine subtitle from a one-frame Tesseract hallucination, and it is why we
deliberately **do not early-exit on the first text frame** the way we do for faces.

Combined effect: keeps every real text type (title card, subtitle, bottom caption, corner
logo, end-card) **and** rejects every OCR-noise false positive.

---

## 5. Critical bugs found and fixed

### 5.1 The AV1 "false-clean" bug (most important)
An early "successful" grind reported **1,566 clean windows** — but a spot-check found
**1,559 of them had empty descriptions**. Cause: yt-dlp had downloaded **AV1-codec** video,
which this platform's ffmpeg/OpenCV **cannot decode** (`Failed to get pixel format` on every
frame). `scan_window` read **zero frames** and **fell through to `clean = True`**, passing
windows it never actually looked at. The grind even finished suspiciously fast (no frames =
no work).

**Fix (two parts):**
1. `fetch.py` forces **`vcodec^=avc1` (H.264)**, which OpenCV can decode (format 136 at 720p).
2. `scan_window` is now **fail-closed**: if any second decodes **0 frames**, the window is
   **rejected as `unreadable`**, never passed.

Verified: a video that was falsely "kept 88/88" became a correct **11/12 text + 1 face**
rejection after the fix; the 1,566-window corpus was discarded and regenerated.

### 5.2 Hang handling (C-level hangs can't be interrupted by signals)
Videos occasionally **froze inside a C call** (OpenCV/torch) — the Python process sat at
**0% CPU**, fully alive but stuck. A Python **`SIGALRM` per-video timeout did not work**
(signals can't interrupt a blocking C call). Two robust mechanisms replaced it:
- **Attempt-marker** (`state/attempts.json`): a video is marked *before* the heavy work; on
  resume, any video marked-but-without-a-record is treated as previously-hung and
  **quarantined/skipped** instead of re-hanging.
- **External hang-detector** (in the monitor loop): if all Python workers sit at **<5% CPU
  for 3 minutes**, kill them, clear scratch, and **auto-restart** (the attempt-marker skips
  the culprit).

### 5.3 Other fixes
- **Per-video / socket timeouts** so a single stuck download can't freeze the run.
- The grind is **resumable + crash-safe**: one record per video, written **atomically**
  (temp + rename); a finished video is skipped on re-run; the container was restarted /
  the background grind died many times and every restart resumed losslessly.

---

## 6. Unblocking YouTube downloads

Modern YouTube hides downloadable formats behind a JS **"EJS" challenge** — yt-dlp returns
**only storyboards** ("Requested format is not available") unless **three** things are present:
1. **`yt-dlp-ejs`** installed (the challenge solver).
2. **`deno`** (and `node`) on `PATH` (the JS runtimes that solve the challenge).
3. **Fresh, full-auth cookies** (a complete browser export with `SID`/`SAPISID`/
   `__Secure-1PSID`/`LOGIN_INFO`; a thin cookie file fails the bot check).

With all three, real 144p–1080p formats download normally (a 720p clip in ~20 s). Stale
cookies caused ~38 download failures late in a run; refreshing them and retrying recovered
those videos.

---

## 7. Growing the corpus from user-provided sources

After rebuilding the original 257 videos, the corpus was grown from two user inputs, both
run through the **same** v3 pipeline (`reclassify.py --ingest`, with auto-segmentation of
new videos into ~8-second windows since they have no upstream window list):

1. **Watch-later list** (25 videos from an exported xlsx) → **96 clean windows from 19
   videos** (6 yielded nothing — talking-head/captioned). Per-video yield ~6× the original
   corpus.
2. **Title search** (100 "Ideas" titles from a workbook xlsx): for each title we derived
   candle topic keywords, ran a YouTube search (`ytsearch2:` — **2 videos per title**),
   **deduped** against the existing corpus and across titles → **126 new videos** →
   **379 clean windows** (`search_titles.py`).

Survival is **bimodal**: a video is either clean B-roll (e.g. `kept 18/18`) or captioned
throughout (`kept 0/69`). A large fraction of candle tutorials are subtitled/watermarked, so
the strict purge legitimately rejects most windows — what survives is genuinely clean.

---

## 8. The final corpus

- **`outputs/shared_db_v2/records/<video_id>.json`** — survivors only; each window has a
  `seconds[]` array (`abs_t`, `vision_desc`, `transcript_text`, `scene_change`,
  `face_mention`) plus window-level `embed_text` (vision) and `transcript` (speech).
- **`outputs/shared_db_v2/rejected/<video_id>.json`** — every dropped window + reason
  (`face` | `text` | `desc_face` | `unreadable`) + timestamp.
- **`outputs/shared_db_v2/by_label/<label>.jsonl`** — browse/match index.
- **`outputs/shared_db_v2/corpus_clean_windows.csv`** — one row per clean window.
- The old `outputs/shared_db/` is marked **`_DEPRECATED.md`** (kept only as the input list
  for re-runs; must not be used for video-making).

**Totals:** 671 clean windows · 5,333 per-second descriptions · sources: corpus 196,
watch-later 96, searched 379.

---

## 9. Tuning knobs (all env-overridable)

| var | default | meaning |
|---|---|---|
| `REVAMP_FACE_SCORE` | 0.60 | min YuNet/SSD confidence to count a face (0.50 false-positived on hands/moulds) |
| `REVAMP_TEXT_MIN_CONF` | 55 | min OCR word confidence to count as text |
| `REVAMP_TEXT_MIN_CHARS` | 4 | min alnum chars for an OCR word (rejects short gibberish) |
| `REVAMP_TEXT_PERSIST_SECONDS` | 2 | confident text must persist on ≥N seconds (rejects flicker noise) |
| `REVAMP_FRAMES_PER_SEC` | 1–2 | frames sampled inside each second during purge |
| `REVAMP_BOTTOM_FRAC` / `REVAMP_TOP_FRAC` | 0.72 / 0.20 | strip crops for subtitle / corner-logo OCR |
| `REVAMP_SCENE_DIFF` | 10.0 | mean-abs frame diff to trigger a re-caption |
| `REVAMP_VLM` | blip | local captioner (swap for a stronger one on GPU) |
| `REVAMP_MAX_HEIGHT` | 720 | download/scan resolution |
| `REVAMP_VIDEO_TIMEOUT` / hang-detector | 600 / 3 min idle | stall guards |

---

## 10. Companion skill fixes (yta-video-maker)

Alongside the corpus work, the video-maker skill received (see `../skill-patches/`):
- **v25.1** — 8 local-run hardening fixes (slug, bootstrap copy, tts import, local query
  rewrite, autobootstrap key gating, align forced-alignment fallback, numpy-vectorized
  matcher, model2vec floor).
- **v25.2** — resumable/atomic render + a resumable, chunked `validate_render`.
- **v25.3** — read the new `shared_db_v2` corpus (vision + transcript kept separate) +
  dark-footage `blackdetect` knob.


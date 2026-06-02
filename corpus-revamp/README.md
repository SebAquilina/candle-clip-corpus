# Corpus revamp — second-by-second face/text purge + per-second re-description

A full rebuild of the candle clip corpus that fixes the root cause of the "faces and text
keep slipping through" problem, and replaces coarse window-level labels with per-second
descriptions (vision + transcript).

## Why the old corpus leaks faces and text

The current corpus (`outputs/shared_db/`) was built from **YouTube storyboard sprites**
(`"source": "storyboard"` on 250/257 videos) — low-res thumbnail mosaics, never full-res
frames. The classifier physically could not see small corner watermarks (TikTok/InShot),
brief title cards/captions, or medium-distance/profile faces, so contamination was baked
into the "usable" set at ingest. Worse, flags are **per ~8s window** (one `action_label`),
but contamination is usually sub-window: one second of caption, a presenter turning to
camera for a beat. There is no per-second truth, and the skill's per-clip QC only samples
~24 frames and otherwise trusts those bad flags. Net: faces/text reach the final video.

## What this rebuild does

For **every usable window** (`is_step != 0`) of every corpus video, going to the **real
full-res footage** and analysing it **second by second**:

0. **(skill side)** resumable/time-budgeted render+validate fixes — see `../skill-patches/`.
1. **Fetch** the source video once (≤720p) into deletable scratch.
2. **Purge (pixel, zero-tolerance, fail-closed)** — every second is scanned with the strong
   detector chain (YuNet → SSD → Haar faces + Tesseract OCR). **Any** face or **any**
   confident text on **any** sampled second → the **whole ~8s window is dropped** and
   quarantined. (`detectors.py`, `purge.py`)
3. **Re-segment + describe survivors** — each surviving window is walked second-by-second;
   each second gets (a) a **local-VLM caption** (BLIP, offline; recomputed only on a scene
   change and carried across static B-roll), and (b) the **transcript** the speaker said
   during that second (YouTube captions, else faster-whisper). Kept **separate** so the
   video-maker can weigh vision vs. speech independently at match time. (`describe.py`,
   `transcript.py`)
4. **Stage 4b — semantic face backstop** — if any second's caption itself implies a
   face/person ("a woman pours…", "man talking to camera"), the window is dropped too.
   Conservative word-boundary matching; `hands`/`fingers` are allowed. (`describe.py`)
5. **Rebuild** the corpus into `outputs/shared_db_v2/` with the per-second schema, a
   window-level roll-up for drop-in matcher compatibility, a `by_label/` index, and a
   `rejected/` quarantine recording every drop + reason. (`reclassify.py`)

Decisions baked in (per your answers): **grind to completion**, **local offline VLM**,
**drop the whole window** on any hit, **zero-tolerance / fail-closed**, plus the **stage-4b**
description-based face drop you added.

## ⚠️ Where this must run

It **cannot run in the Claude Code web sandbox**: that container's datacenter IP is
bot-walled by YouTube (`yt-dlp` returns *"Sign in to confirm you're not a bot"* / 403 /
DRM-only formats, and the bundled cookies are stale). Every stage that needs frames or
captions therefore needs an environment where YouTube downloads work — **your local
machine or the Cowork environment that successfully downloaded clips before.** The code,
detector, VLM, schema and resumability are all built and **validated on real footage here**
(see "Validation"); only the bulk download/scan must run where the network allows it.

## Run it

```bash
cd corpus-revamp
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only boxes
# system deps: ffmpeg, tesseract-ocr, nodejs   (YuNet model is vendored here)

export REVAMP_COOKIES=/path/to/fresh/youtube_cookies.txt   # a FRESH full cookie export
python reclassify.py                 # the full grind — resumable, one video at a time
python reclassify.py --status        # progress + kept/dropped/reasons
python reclassify.py --reindex       # rebuild by_label/ from records/
```

Resumable & crash-safe: one record per video, written atomically; a finished video is
skipped on re-run, a download failure is recorded (not fatal) and the grind continues.
For windowed/cron runs use `--max-seconds N` (exits cleanly at the budget) or
`--max-videos N`.

### Scale & cost (CPU-only estimate)

257 videos · 5,499 usable windows · ~26 h of footage. Purge ≈ early-exits on contaminated
windows; clean windows + the BLIP describe dominate. Budget **roughly a day of wall-clock
on a 4-core CPU**; far faster with a GPU (set a stronger VLM — see Tuning). Disk: scratch
holds one ≤720p video at a time (auto-discarded).

## Output schema

`outputs/shared_db_v2/records/<video_id>.json` — survivors only; see
[`EXAMPLE_v2_record.json`](./EXAMPLE_v2_record.json). Each kept window carries a `seconds[]`
array (`abs_t`, `vision_desc`, `transcript_text`, `scene_change`, `face_mention`) plus a
window-level `embed_text` (unique vision phrases) and `transcript` so the **existing matcher
can consume v2 unchanged**. `rejected/<video_id>.json` records every dropped window with its
reason (`face` | `text` | `desc_face`) and timestamp. `by_label/<label>.jsonl` is the
browse/match index.

## Validation performed here (on real prior-run footage, no YouTube needed)

- **Detector loads YuNet** (strong backend), not the weak Haar fallback.
- **Face detector fires** on a real frontal face (1 box, score 0.80); **ignores** blank frames.
- **OCR fires** on real text ("ECOSOYA PILLAR FAILED" → flagged); **ignores** blank frames.
- **BLIP describe** runs per-second on real segments with scene-change dedup (12 s of static
  B-roll → 1 caption computed).
- **Stage-4b** regex: flags `woman`/`man`/`face`/`talking`, ignores `hands holding a candle`.
- **Whisper transcript** path: 70 accurate word timings from 30 s of narration.
- **Full pipeline** (purge→describe→4b→roll-up) end-to-end on real segments → valid v2 record.

## Tuning (env knobs)

| var | default | meaning |
|---|---|---|
| `REVAMP_FACE_SCORE` | 0.50 | min YuNet/SSD confidence to count a face (lower = stricter) |
| `REVAMP_TEXT_MIN_CONF` | 45 | min OCR word confidence to count as text |
| `REVAMP_FRAMES_PER_SEC` | 2 | frames sampled inside each second during purge |
| `REVAMP_SCENE_DIFF` | 10.0 | mean-abs frame diff to trigger a re-caption |
| `REVAMP_VLM` | blip | `blip` (offline) or `stub`; swap `REVAMP_BLIP_MODEL` for a stronger captioner if you have a GPU (e.g. BLIP-2, Florence-2, moondream) |
| `REVAMP_MAX_HEIGHT` | 720 | download/scan resolution (higher = better small-text/face recall, slower) |

Start strict (defaults) and watch `--status` reject rates; if false positives gut the
corpus, raise `REVAMP_FACE_SCORE` / `REVAMP_TEXT_MIN_CONF` a little. The old corpus is left
untouched in `shared_db/` until you bless `shared_db_v2/`.

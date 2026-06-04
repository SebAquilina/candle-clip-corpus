---
name: clip-corpus-builder
description: Build or grow a CLEAN B-roll clip corpus from a user-provided CSV or Excel of videos (YouTube IDs, URLs, or titles). It downloads each source video, scans it SECOND-BY-SECOND and drops every ~8s window that contains a human face or on-screen text (OpenCV YuNet face detection + a 3-pass Tesseract OCR), then describes every surviving clean window per-second with a local vision caption (BLIP) and the spoken transcript — kept separate. Produces a per-window CSV plus a JSON corpus (outputs/shared_db_v2). Resumable and idempotent. Trigger when the user hands a CSV/Excel/list of videos and asks to add them to the corpus, build a clean clip library, scrape clean B-roll windows, or "put these through the pipeline". Claude does NOT search for videos itself unless the user explicitly asks (then it can resolve titles via search).
---

# Clip Corpus Builder

Turns a list of source videos into a **clean, described B-roll corpus**: every clip window
in the output is verified **face-free and text-free**, and carries a per-second **vision**
description (what is on screen) and **transcript** (what is spoken), kept in separate fields
so a downstream video-maker can weigh them independently.

This skill is the **one task Claude runs for the user**: *given videos, produce clean
windows.* It does the sorting/deduping/scraping; the user only supplies the videos.

## When to use
- The user provides a **CSV or Excel** (or a pasted list) of YouTube videos — as IDs, full
  URLs, or **titles** — and wants them added to / used to build the clean corpus.
- The user says "add these to the corpus", "scrape clean windows from these", "put these
  through the pipeline", "grow the corpus".
- Do **not** go find videos yourself unless the user asks; then use the `--titles` mode.

## Setup (self-bootstrapping)
```bash
cd <skill-dir>
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only boxes
# system binaries: ffmpeg, tesseract-ocr, nodejs, deno   (the YuNet model is vendored here)
export REVAMP_COOKIES=/path/to/fresh/youtube_cookies.txt   # FRESH full-auth browser export
python csv_ingest.py --selftest      # preflight: detector + VLM + a real test download
```
**Downloads require three things or YouTube returns only storyboards:** `yt-dlp-ejs`
(installed by requirements), `deno`+`node` on PATH, and fresh full-auth cookies.

## Run it
```bash
# videos given as IDs / URLs in the file:
python csv_ingest.py videos.csv
python csv_ingest.py videos.xlsx

# the file only has TITLES and the user asked you to source videos:
python csv_ingest.py content_plan.xlsx --titles 2     # search 2 videos per title

# check progress / re-export at any time:
python reclassify.py --status
python export_csv.py outputs/shared_db_v2/corpus_clean_windows.csv
```
`csv_ingest.py`: parses every cell → extracts video IDs from IDs/URLs (or resolves titles
via search when `--titles N`) → **dedupes** against the existing corpus and within the list
→ runs each NEW video through the pipeline → **re-exports the full clean CSV**.

**Resumable & crash-safe.** One record per video, written atomically; a finished video is
skipped on re-run; a hung video is quarantined (`state/attempts.json`) and skipped. If the
process dies, just re-run the same command — it picks up where it left off. For very long
runs, monitor with `reclassify.py --status` and restart on death.

## How the pipeline works (per video → per window → per second)
1. **Fetch** the video once at ≤720p **H.264** (avc1) into deletable scratch. H.264 is
   forced because OpenCV cannot decode AV1 (which would silently read 0 frames).
2. **Purge** (`detectors.scan_window`), fail-closed, drop-the-whole-window:
   - **Faces** — OpenCV **YuNet** (→ SSD → Haar fallback), `REVAMP_FACE_SCORE` default 0.60.
     Any face on any sampled frame rejects the window immediately.
   - **Text** — a **3-pass Tesseract OCR**: full frame, **bottom strip** (subtitles), and
     **top strip** (corner logos/watermarks), each contrast-enhanced (CLAHE). A word counts
     only at conf ≥ 55 and ≥ 4 alphanumeric chars (rejects OCR noise on textured B-roll).
     Text must **persist on ≥ 2 distinct seconds** to reject (so a one-frame hallucination
     doesn't drop clean footage). A second that decodes 0 frames → rejected `unreadable`.
3. **Describe survivors** (`describe.py`): per second, a **BLIP** caption (re-run only on a
   scene change, carried across static B-roll) + the **transcript** for that second
   (YouTube captions, else faster-whisper). Vision and transcript are stored separately.
4. **Stage 4b** semantic face backstop: if any second's caption itself names a person
   ("a woman pours…"), drop the window too (`hands`/`fingers` allowed).
5. **Write** `outputs/shared_db_v2/records/<id>.json` (survivors, with `seconds[]` +
   window-level `embed_text` (vision) + `transcript` (speech) roll-up), a `rejected/`
   quarantine with reasons, a `by_label/` index, and the CSV.

New videos (no upstream window list) are **auto-segmented into ~8-second windows** before
the same purge+describe runs.

## Output the user gets
- `outputs/shared_db_v2/corpus_clean_windows.csv` — **one row per clean window**:
  `video_id, video_url, video_title, source, niche, window_index, start_s, end_s,
  duration_s, action_label, phase, n_seconds, vision_embed_text, transcript`.
- `outputs/shared_db_v2/records/*.json` — full per-second corpus (drop-in for the matcher).
- Deliver the CSV to the user when done; report kept/dropped totals from `--status`.

## Survival reality (set expectations)
Survival is **bimodal**: a video is either clean B-roll (`kept 18/18`) or captioned/talking
throughout (`kept 0/69`). Many social tutorials are subtitled/watermarked and legitimately
drop most windows — what survives is genuinely clean. This is expected, not a bug.

## Tuning (env, all optional)
| var | default | meaning |
|---|---|---|
| `REVAMP_FACE_SCORE` | 0.60 | min face confidence (lower = stricter; 0.50 false-positives on hands/moulds) |
| `REVAMP_TEXT_MIN_CONF` / `_MIN_CHARS` | 55 / 4 | OCR word confidence / min length |
| `REVAMP_TEXT_PERSIST_SECONDS` | 2 | confident text on ≥N seconds → reject (kills flicker noise) |
| `REVAMP_VLM` | blip | local captioner (swap `REVAMP_BLIP_MODEL` for a stronger one on GPU) |
| `REVAMP_MAX_HEIGHT` | 720 | download/scan resolution |
| `REVAMP_NICHE_HINT` | candle making | topic words appended to title searches |
| `REVAMP_COOKIES` | — | path to fresh YouTube cookies |

See `METHODOLOGY.md` for the full reasoning behind every design choice.

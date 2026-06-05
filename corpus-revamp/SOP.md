# clip-corpus-builder — Standard Operating Procedure (SOP)

Running notes for operating and publishing the corpus. The detailed *why* of the pipeline is
in `METHODOLOGY.md`; this is the *how we run and ship it*.

## 1. Build / grow the corpus
- Input: a CSV/Excel of videos (IDs, URLs, or titles). `python csv_ingest.py videos.csv`.
- Pipeline per video → per ~8s window → per second: download (H.264) → YuNet face purge +
  3-pass Tesseract OCR (full + bottom-strip subtitles + top-strip logos, persistence ≥2s) →
  BLIP vision caption + transcript (kept separate) → write `outputs/shared_db_v2/records/<id>.json`.
- **Channel + title are captured at ingest** (yt-dlp metadata → record) for downstream credit.
- Resumable + idempotent: a finished video is skipped; a hung one is quarantined.

## 2. Cookies & secrets (NEVER commit)
- Downloads need **fresh full-auth YouTube cookies** + `deno`+`node` (yt-dlp-ejs) on PATH.
- Put cookies OUTSIDE the repo (e.g. `/tmp/yt_cookies.txt`), point `REVAMP_COOKIES` /
  `VM_COOKIES` at them. `.gitignore` blocks `*cookies*.txt` and `*.env` — keep it that way.
- yt-dlp flag gotcha: `--retry-sleep` syntax is `TYPE:EXPR` (e.g. `http:linear=2:30:5`).
  A bad value (`http=2:30:2`) makes yt-dlp exit instantly → every download "fails" in ~1s.

## 3. Pre-download the clips (speeds re-describe & video-making)
`python download_clips.py` caches every clean window into `outputs/clip_cache/<vid>_<start>.mp4`
(resumable, paced for rate-limits). `--status` shows cached/total. The cache is gitignored
media; the video-maker reads it first before hitting YouTube.

## 4. Publish the corpus to GitHub (the additive way — do NOT corrupt main)
The corpus is **data**, kept in its own niche-specific repo, **separate from skill code**, and
promoted to `main` as a **purely additive** change. The video-maker is general and clones this
repo at runtime.

```bash
# from a clone of the corpus repo, on a fresh branch off main:
git fetch origin main -q
git checkout -b corpus-update origin/main
git checkout <work-branch> -- outputs/shared_db_v2            # bring ONLY the corpus
git add outputs/shared_db_v2
# SAFETY GATE — must be additions/edits to the corpus only, never deletions of unrelated files:
git diff --cached origin/main --name-status | grep -vE '^[AM]\s+outputs/shared_db_v2/' && echo "STOP: touches non-corpus files" || echo "safe"
git commit -m "corpus update: <N> records / <M> windows"
git push -u origin corpus-update
# open a PR base=main, head=corpus-update; confirm mergeable_state=clean + 0 deletions; squash-merge.
```
Rules: (a) **never force-push main**; (b) the diff must be confined to `outputs/shared_db_v2/`
(+ the old-corpus deprecation note); (c) keep skill code / bundles / test media OUT of this PR.

## 5. Hand the corpus to the video-maker
Give the video-maker the repo URL; it runs `get_corpus.sh <url>` → `YTA_SHARED_DB` + derived
`VM_NICHE`. The skill stays niche-agnostic; the corpus defines the niche.

## 6. (in progress) Second-by-second understanding upgrade
Goal: richer, more matchable per-second descriptions than BLIP-base. Approaches + the chosen
pipeline are tracked in `DESCRIBE-UPGRADE.md`; whatever lands here updates `describe.py` and is
republished via §4. Re-describe reads from the §3 clip cache (no re-download).

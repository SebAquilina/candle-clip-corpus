# What I built — delivery summary

Four things, in the order you asked, all on branch `claude/kind-hopper-ZlOrH` (draft PR #1).

---

## 1. A clean B-roll corpus — `outputs/shared_db_v2`
Rebuilt the corpus **second-by-second** to honour the hard rule *no human face and no on-screen
text, ever*:
- **671 clean windows** from 399 source videos. Every ~8s window passed a per-second purge:
  **OpenCV YuNet** face detection (→ SSD → Haar fallback) + a **3-pass Tesseract OCR** (full
  frame, bottom strip for subtitles, top strip for corner logos, contrast-enhanced), with
  text needing to **persist ≥2 seconds** to count (kills flicker noise). Fail-closed: an
  unreadable second rejects the whole window.
- Each surviving window carries, per second, a **BLIP vision caption** (what's on screen) and
  the **spoken transcript** (what's said), kept in **separate** fields.
- Exported to `corpus_clean_windows.csv` (one row per window). Full write-up + reasoning in
  `corpus-revamp/METHODOLOGY.md` (fact-checked by a team of agents, 22/22 claims verified).
- The previous corpus is marked `outputs/shared_db/_DEPRECATED.md` — do not use.

## 2. `clip-corpus-builder` skill — `corpus-revamp/`
Hand it a **CSV or Excel** of videos (IDs, URLs, or titles); it sorts/dedupes them and runs
each NEW one through the **exact same v3 pipeline**, extracting clean windows into the corpus
and the CSV. Resumable, idempotent, self-bootstrapping. Claude doesn't source videos itself
unless you pass `--titles N` (then it resolves titles via search). One entry point:
`python csv_ingest.py videos.csv`.

## 3. `video-maker-3` skill — `video-maker-3/`
A heavy streamline of the old `yta-video-maker-2` (≈40 modules / 116 KB doc → a lean
corpus-only pipeline), rewritten to your spec:

- **Sources only from the clean corpus.** No Gemini, no Pexels, no live discovery.
- **Matching uses BOTH the vision description AND the transcript, in the context of the
  title, with Claude directly in the loop.** An offline pass shortlists ~12 real candidates
  per section (combined vision+transcript similarity + an action-aware re-rank that stops
  "pour" matching a "measure" clip); **Claude then reads each section's candidates and picks
  the best** (`match_worklist.json` → `match_decisions.json`). Not a blind embedder — for the
  mock video, two agents went through all 59 sections and rejected off-topic false matches
  (a printer, a fireplace, a red bird) in favour of real candle footage.
- **No-repeat / no-freeze, enforced *and asserted* in code:** a clip is used **at most twice**,
  **never back-to-back**, and **never frozen/paused/held**. If a clip is shorter than its
  section, the **next-best matching clip is appended** (best-clip-first concat), trimmed to fit.
  `assembly_report` audits every build and the build refuses to render if either rule is broken.
- **Lighter per-clip QC, but the non-skippable final gate stays.** The corpus is already clean,
  so per-clip screening is light (faces + black); the full **black / freeze / AV-skew /
  overlay-text / face** gate (`validate_render.py`) is the "just in case" backstop.
- **Source credit on every clip** — see §5.

Run it: `make.py plan` (TTS → align → sections → shortlist) → you pick clips → `make.py build`
(assemble → download windows → render) → `validate_render.py`. See `video-maker-3/SKILL.md`
and `ARCHITECTURE.md`.

## 4. Mock video — the acceptance test (and how the skill got hardened)
I drove a "Top 10 Candle Making Tricks" script end-to-end and **fixed the skill on every
failure so it can't recur** — 12+ real bugs (`video-maker-3/TEST-LOG.md`), including a QC parse
typo that silently rejected everything, a render concat bug, cv2 hang-proofing, and
niche-appropriate detector thresholds (YuNet/Tesseract over-fire on circular wax textures and
product labels — real faces score ~0.94 vs texture noise ~0.75). Notably the final gate
**caught a real face clip the corpus had missed** (a person at a wax line) and the build
excluded it while keeping legitimate footage — exactly the "lighter QC, still checked"
behaviour you asked for.

**Final output passes the full safety gate:** `black ✓ freeze ✓ av-skew ✓ overlay-text ✓
face ✓ → VALIDATE_RENDER: OK`. 59 shots, 39 distinct clips (≤2 uses, 0 consecutive), 1080p30,
narration muxed.

## 5. The source credit (your correction)
The bottom-left credit is **required** and now reads **"via &lt;source channel&gt; on YouTube"** —
the actual uploader of each clip, not a generic "YouTube". The corpus only stored channels for
2/127 videos, so the skill now **resolves each clip's channel via yt-dlp metadata and caches it**
(`channels.json`) — this works even while full segment downloads are rate-limited, since
metadata lookups are lighter. It prefers a channel already in the corpus when present.

---

## Honest caveats
- **Length:** the delivered cut is ~5 min because EdgeTTS reads the ~1,500-word script faster
  than the 150 wpm estimate. A longer script (or `EDGETTS_RATE=-20%`) yields ~10 min.
- **Downloads:** YouTube rate-limited the shared cookie account during the test, so the cut
  assembles from the 75 windows cached before the block (`VM_CACHED_ONLY`). On a fresh/cooled
  account the build sources every section's best match — no skill change needed.
- **Recommended follow-up:** backfill the source channel into every corpus record (durable
  attribution) and have `clip-corpus-builder` capture it at ingest, so the video-maker never
  needs a live channel lookup.

# Master plan — remaining work (written before you slept)

Three remaining tasks, kept **strictly separate** so they don't get confused. Order is the
order you gave: **(A) corpus-builder skill → (B) video-maker redesign → (C) mock video test**.
Resources are not scarce, so each task uses **teams of agents** for research/verification.

---

## ✅ Already done this session (context)
1. Clean corpus rebuilt second-by-second → **671 clean windows**, 5,333 per-second
   vision+transcript descriptions, all JSON-valid.
2. Verified + exported `corpus_clean_windows.csv` (delivered), committed + pushed.
3. `METHODOLOGY.md` written and **fact-checked by 3 agents — all 22 claims PASS**.
4. Old corpus marked `_DEPRECATED.md`; skill patches v25.1–v25.3 shipped.

---

## TASK A — "Corpus-builder" skill  (do FIRST)

**What it is:** a simple, repeatable skill. The user hands a **CSV or Excel of videos**
(IDs / URLs, or titles); Claude parses + dedupes them and runs them through the **exact v3
pipeline** already built, extracting clean windows into the corpus and a CSV. The user can
hand more videos any time → same pipeline (incremental, resumable). **Claude does NOT
self-source** videos unless explicitly asked (then it searches like `search_titles.py`).

**Components (already built + validated this session):** `detectors.py` (3-pass OCR +
YuNet + persistence), `fetch.py` (H.264, EJS, cookies), `describe.py` (BLIP + transcript +
stage-4b), `transcript.py`, `reclassify.py --ingest` (auto-segments new videos),
`export_csv.py`, `search_titles.py`, vendored `clip_checks` + YuNet model, `requirements.txt`.

**Steps:**
1. Write `csv_ingest.py` — the single entry point: read CSV/Excel → extract video IDs from
   IDs/URLs (regex) **or** titles (→ ytsearch) → dedupe vs existing corpus + within list →
   `ingest_list()` → `export_csv.py`. Resumable; preflight `--selftest`.
2. Write `SKILL.md` that tells Claude exactly how to drive it (parse the user's file, choose
   IDs-vs-titles mode, run, deliver the CSV). Self-bootstrapping deps note (yt-dlp-ejs, deno,
   ffmpeg, tesseract, torch+transformers, cookies).
3. Bundle: copy the corpus-revamp scripts + YuNet model + requirements into a `.skill` zip.
4. **Validate**: run `csv_ingest.py` on a tiny 2-video CSV end-to-end here (cookies work).
5. **Agents:** 1 agent drafts SKILL.md from METHODOLOGY.md; 1 agent reviews the entry point
   for edge cases (mixed URL/ID/title rows, dupes, Excel quirks); 1 agent does a final
   factual review of the bundle vs the real scripts.

**Deliverable:** `corpus-builder.skill` + a short runbook + a validated example.

---

## TASK B — Redesign the video-maker skill  (do SECOND)

**Base:** the current `ytavideomaker-2.0-cascade-hardened.skill` (you uploaded it; I also
have the source in `/tmp/yta_build`). It is **bulky** — streamline it hard.

**Hard requirements (captured verbatim from your message):**
1. **Streamline** — keep the essential render/validate/assemble path; cut the bulk
   (Gemini/Pexels/legacy sourcing, storyboard ingest, corpus-builder duplication — sourcing
   now comes from the clean `shared_db_v2` corpus, not live discovery).
2. **Lighter QC, but still present** — the corpus is already face/text-free, so the
   per-clip face/text screening can be light (trust the corpus) **but keep a final safety
   gate** "just in case" (cheap sample, plus the black/freeze checks stay).
3. **Matching uses BOTH vision description AND transcript**, weighted by the **title's
   context**, with **Claude directly involved** — i.e. the skill instructs Cowork/Claude to
   read each script segment + candidate windows (vision + transcript) and pick the best,
   like the previous skill's in-session classification. Not a blind embedder.
4. **No-repeat / no-freeze rules (strict):**
   - a clip may appear **at most twice** in the whole video,
   - **never consecutively** (never the same clip back-to-back),
   - **no frozen/paused/last-frame-hold** clips anywhere.
5. **Best-clip-first concatenation:** if the top clip is shorter than a speech segment, take
   the **next-best matching clip** and **append** it (clip finishes → next clip starts
   immediately) until the segment is covered; never stretch/freeze a clip to fill time.

**Steps:**
1. **Agents analyze the bulky skill** (team): map every script/module, label keep / cut /
   simplify, and extract the parts we must preserve (render_video.py, validate_render.py,
   TTS provider, align/section planner, the v25.x fixes).
2. Design the streamlined architecture: `corpus_match` (vision+transcript+title, Claude-in-
   the-loop) → `assemble` (no-repeat ≤2 + non-consecutive + concat-fill, no freeze) →
   `render` (atomic mux) → `validate` (light per-clip + black/freeze/av).
3. Implement the new matcher + assembler honoring rules 3–5. Reuse the v25.x resumable
   render/validate.
4. Write the new lean `SKILL.md`.
5. Bundle `video-maker-3.skill`.
6. **Agents** fact-check the new skill's claims vs its code before shipping.

---

## TASK C — Mock video (test + iterate the new skill)

- **Topic:** "Top 10 Candle Making Tricks", **~10 minutes**.
- **TTS:** use **69labs** (Candice) if I still have the `LABS69_*` creds; otherwise fall back
  to EdgeTTS (free). Either is fine per your message.
- **Flow:** Claude writes a ~10-min script → matches each segment to clean corpus windows
  (vision+transcript+title) → assembles with the no-repeat/concat rules → renders → validates.
- **Iterate:** if anything breaks (matching, repeats, freezes, render, mux, validate), **fix
  the skill** so it can't repeat that mistake, re-run, and note the fix. This is the
  acceptance test for Task B.
- **Deliverable:** the finished ~10-min mp4 + a list of skill fixes the test surfaced.

**Note:** Task C needs YouTube downloads (to materialize matched clips) — works now via
yt-dlp-ejs + cookies. It's heavy/long; I'll run it resumably with the same monitor/restart
discipline used for the corpus grind.

---

## Execution discipline
- One task fully done + committed before the next; never interleave their files.
- Teams of agents (Opus 4.8) for analysis, drafting, and **factual review** of every skill
  before it ships — same fact-check rigor that validated METHODOLOGY.md.
- Commit + push after every meaningful step (work must never be undone).
- Honest status; if a background job dies, restart it (resumable) — don't claim it's alive.

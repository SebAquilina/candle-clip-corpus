---
name: video-maker-3
description: Turn a narration script into a finished B-roll video, sourcing every clip from the CLEAN pre-built corpus (outputs/shared_db_v2). It narrates the script (69labs or EdgeTTS), aligns it to speech-timed sections, then SHORTLISTS the best corpus windows per section by a combined score over BOTH the on-screen vision caption AND the spoken transcript — and hands that shortlist to Claude to PICK the best clip(s) per section in the context of the video title. It assembles best-clip-first (append the next-best to cover a section), with every clip used at most TWICE, NEVER back-to-back, and NEVER frozen/held; downloads exactly those windows; renders with an atomic mux; and runs a non-skippable final gate (black/freeze/AV-skew/text/face). Trigger when the user wants to make/build/produce a video from a script using the clean clip corpus. The corpus must already exist (build it with the clip-corpus-builder skill).
---

# Video Maker 3

Makes a narrated B-roll video from a script, using **only** the clean corpus the
clip-corpus-builder produced (`outputs/shared_db_v2`: every window is face-free and
text-free, with a per-window **vision** caption and **transcript** kept separate). No live
discovery, no Gemini, no Pexels — the corpus is the single source. Claude does the matching.

## What makes this version different (the brief)
- **Streamlined**: the old skill's discovery/pool/Hungarian-assign/Gemini/Pexels machinery is
  gone. The path is just `TTS → align → sections → shortlist → [Claude picks] → assemble →
  download → render → validate`.
- **Matching uses BOTH vision and transcript, with Claude in the loop.** An offline pass
  shortlists ~12 real candidates per section by a **combined vision+transcript similarity**
  (the `VM_W_*` weights) + action-aware rerank + topic/niche expansion; **Claude then reads
  each section's candidates and picks the best in the context of the video title**, because a
  generic caption often under-rates a window whose speaker is literally narrating the action.
  (The offline scorer weights the two signals and uses the topic; the *title* context is
  applied by Claude — that is the "directly involved" step.)
- **No-repeat / no-freeze (enforced in code):** a clip is used **at most twice**, **never
  consecutively**, and **never frozen/paused/held**. If a clip is shorter than its section,
  the **next-best matching clip is appended** (best-clip-first concat), trimmed to fit.
- **Lighter QC, but still checked:** the corpus is already clean, so per-clip screening is a
  light black/face/text sample — but the **non-skippable final gate stays** (`validate_render.py`).

## The corpus is EXTERNAL and niche-specific (this skill is general)
This skill is **niche-agnostic** — it makes a video for whatever corpus you point it at. The
corpus is a **separate GitHub repo** built by the `clip-corpus-builder` skill (e.g. a candle
corpus, a soap corpus). **The user supplies that repo URL at runtime; it is NOT baked into
the skill.** Fetch it and let it set the corpus path + niche for you:
```bash
eval "$(bash scripts/get_corpus.sh https://github.com/<user>/<corpus-repo>)"
# -> clones it, then exports YTA_SHARED_DB=.../outputs/shared_db_v2 and VM_NICHE=<corpus niche>
```
The niche (`VM_NICHE`) is **derived from the corpus records** when unset, so candle/soap/etc.
all work without code changes.

## Setup
```bash
cd <skill-dir>
bash bootstrap.sh                       # venv + deps; lists any missing system binaries
eval "$(bash scripts/get_corpus.sh <corpus-repo-url>)"  # external corpus -> YTA_SHARED_DB + VM_NICHE
export VM_COOKIES=/path/to/fresh/youtube_cookies.txt    # fresh full-auth cookies (for downloads)
# optional 69labs voice (else EdgeTTS is used automatically):
export LABS69_API_KEY=... LABS69_VOICE_ID=...
python scripts/make.py --selftest       # offline: assembly rules + matcher + embedder
```
System binaries: **ffmpeg/ffprobe**, **tesseract** (final-gate text), **deno + node**
(yt-dlp-ejs, to download corpus windows). Downloads need all three + fresh cookies, exactly
like the corpus builder. (If the corpus repo ships an `outputs/clip_cache/`, matched windows
are read from there first — no download needed.)

## Script writing — Retention Opening (REQUIRED)

Every script passed via `--script` **must** open with a **Retention Opening** before the
body. This is a small, persona-independent structure that front-loads what the video covers
and gives the viewer a concrete reason to stay to the end — it materially lifts retention.

In order, the opening must contain:
1. **Signature open** *(optional)* — channel name / catchphrase / framing device if the host
   has one; 2-4 sentences. Skip if none.
2. **Hook** — one vivid scene or bold claim ending in a clear *promise* of what the viewer
   will walk away with.
3. **Preview** — plainly list, in order, the 3-4 chapters the body will cover (use noun
   phrases that map 1:1 to the body).
4. **Open loop** — name that the most valuable item is saved for last, **without revealing
   what it is**. Tie it to a payoff ("the one that changed everything", "the part nobody
   tells you").
5. **Transition** into the body chapters.

Then near the end:

6. **Payoff (close the loop)** — explicitly deliver the saved-for-last item and call back to
   the hook's promise so the loop visibly closes.
7. **Outro / sign-off** in the host's normal style.

**Mechanics are fixed across all personas; only the wording adapts to the voice** (hype host,
warm host, expert host — same skeleton, different delivery). Never break character for hype.

Drop-in instruction for any script-writing agent or prompt:

> "Every script must open with a Retention Opening: (1) optional signature open, (2) a hook
> ending in a promise, (3) a plain preview of the 3-4 chapters in order, (4) an open loop
> that saves the best payoff for last without revealing it. Deliver that saved payoff near
> the end and call back to the hook's promise. Keep the mechanics fixed across all personas;
> adapt only the wording to the character's voice."

Full guide + worked examples (energetic + calm hosts): `references/RETENTION_OPENING.md`.

## Run it (TEAMS of Claude agents are in the loop between `plan` and `build`)

The skill is **built for parallel agents at every Claude-in-the-loop step.** It splits work
into N slices and the operator spawns N agents in parallel. Use `--slices N` or set
`VM_PICK_SLICES` / `VM_REVIEW_SLICES` (default 4 of each).

```bash
# 1. PLAN — narrate, section, shortlist candidates, EMIT N SLICES for parallel pickers
python scripts/make.py plan --topic top10 --script script.md \
       --title "Top 10 Candle Making Tricks" --slices 4
```
This writes `state/runs/top10/match_worklist.json` (full) AND
`state/runs/top10/worklist_slices/slice_00.json … slice_03.json` (one per agent).

**Now spawn a team of agents IN PARALLEL** — one per slice. Each agent's task is:

> Read `state/runs/<topic>/worklist_slices/slice_<i>.json`. It has the project title + niche
> and, per section in that slice, the narration text and ~12 candidate windows — each with its
> **vision** caption, its **transcript**, `summary_v2`, `tags_v2` (action / stage / tools /
> materials / colors), action label, source title, and offline score. For each section pick
> the clip(s) whose **vision and transcript together** best convey the narration **in the
> context of the title**. Prefer a clip that both *shows* and (when spoken) *describes* the
> action. Write `state/runs/<topic>/decisions_<i>.json`:
> `{"<section.index>": ["<cand_id>", "<cand_id>", ...]}` — **key by the section's `index`
> FIELD (not slice/list position)**, AS A STRING. Every cand_id MUST come from THAT section's
> `candidates`. List 1-4 best-first per section so short clips can be concatenated to cover it.
> You don't need to track repeats — the assembler enforces ≤2 uses / never-consecutive.

```bash
# 2. BUILD — auto-merges every decisions_*.json (safely, via cand_id recovery if any
#    agent accidentally keyed by positional index), then assembles + renders.
python scripts/make.py build --topic top10

# 3. VALIDATE — the non-skippable safety gate (black/freeze/AV-skew/text/face)
python scripts/validate_render.py state/runs/top10/top10.mp4

# 4. (Optional) REVIEW — emit N spot-check slices for a parallel "did the matcher pick well?"
#    audit by another team of agents. Catches off-topic/wrong-clip placements the gate misses.
python scripts/make.py review --topic top10 --slices 4
```
The build's parallel-decisions merge is **safe**: every cand_id is unique to its section's
shortlist, so even if an agent mis-keys with slice-local indices the merge recovers the
correct global section. Missing sections fall through to the offline cosine order.

`make.py all --topic ... --script ...` runs plan+build with the **offline** order (no agents
in the loop) — a CI/fallback path; the intended flow is plan → team-of-pickers → build.

**Resumable:** downloaded windows are cached under `state/runs/<topic>/raw/`; QC verdicts in
`qc_cache.json`; segments in `segs/`. Re-running `build` reuses everything. The final gate has
its own resumable mode (`VM_VALIDATE_BUDGET_SEC>0`).

## Corpus describe — v2 fields (preferred when present)

When the corpus has been enriched via the corpus-builder's `enrich_v2.sh` (agent-vision
describe), each window carries **additive v2 fields** that this skill automatically prefers:

- `embed_text_v2` — rich, action-led, object-rich vision text (~2.4× richer than baseline BLIP).
  The matcher's vision cosine runs against this when present.
- `summary_v2` — one-sentence per-clip summary; sent to Claude in the worklist.
- `seconds_v2[]` — per-second context-aware descriptions; available for inspection.
- `tags_v2` — structured `{action, stage, tools[], materials[], container, colors[], setting}`;
  passed to Claude in the worklist for richer per-section picks.
- `usable_v2` — soft cleanup. `shared_library` emits `talking_head: true` for any window with
  `usable_v2=false`, so the matcher skips windows the agent QC flagged as off-topic,
  burned-in-overlay, or face-bearing.

Graceful fallback: any window without v2 fields still works via the original BLIP
`embed_text` + `transcript` — the matcher tolerates either shape.

## How a section becomes clips (the assembler)
`plan_from_worklist` walks each section best-first over **[your picks] + [offline order]** and,
via a global use-count + last-clip state:
1. lays the best candidate that (a) is under the 2-use cap, (b) isn't the clip just played
   (non-consecutive), and (c) **actually downloads + passes the light QC** (else it reaches
   for the next-best — an editor's move);
2. if that clip is shorter than the section, **appends the next-best** until covered, trimming
   the last to land on the section duration;
3. never holds/freezes a frame; an unfillable tail (rare with a 671-window corpus) takes the
   globally least-used window — still a real moving clip.
`assembly_report` audits this every build (`max_uses_seen`, `consecutive_violations`) and the
build **asserts** both rules before rendering.

## Output
- `state/runs/<topic>/<topic>.mp4` — the finished 1920×1080/30fps video, narration muxed
  (loudnorm, +faststart), one bottom-left source credit per clip.
- `match_worklist.json` / `match_decisions.json` / `shots.json` — the full, auditable trail of
  what was considered, what Claude picked, and what was placed where.

## Tuning (env, all optional)
| var | default | meaning |
|---|---|---|
| `VM_MAX_CLIP_USES` | 2 | max times any one clip may appear (never consecutive regardless) |
| `VM_W_VISION` / `VM_W_TRANSCRIPT` | 0.55 / 0.45 | offline shortlist weighting of the two signals |
| `VM_SHORTLIST_K` | 12 | candidates shown to Claude per section |
| `VM_MAX_SECTION_SEC` | 9 | split a longer sentence into sub-sections (keeps each query focused) |
| `VM_LIGHT_QC_FACES` | 1 | per-clip face QC at materialize (gate-aligned, subprocess+timeout, cached) — excludes any window the corpus purge missed; `VM_QC_TEXT=1` also text-scans per clip |
| `YTA_FACE_SCORE` | 0.6 | face confidence threshold. **Use 0.85 for candle/dark-texture corpora** — YuNet false-positives on circular wax textures (~0.75) below real faces (~0.94) |
| `VM_TEXT_AREA_FRAC` | 0.012 | min frame fraction of text to flag. **Use 0.05 for candle corpora** so incidental product labels pass and only real overlay captions fail |
| `VM_BLACK_PIX_TH` | 0.02 | black-detect pixel threshold (0.02 = only true black; dark candle footage passes) |
| `VM_CACHED_ONLY` | 0 | assemble only from already-downloaded windows (use when YouTube is rate-limiting 429/503) |
| `VM_DOWNLOAD_PAD` | 1.5 | tail seconds added to each window download so a clip is never short |
| `LABS69_API_KEY`/`LABS69_VOICE_ID` | — | use 69labs; otherwise EdgeTTS (`EDGETTS_VOICE`/`EDGETTS_RATE`) |
| `YTA_SHARED_DB` / `VM_COOKIES` | — | corpus dir / fresh YouTube cookies |

**For this candle corpus**, run build + gate with `YTA_FACE_SCORE=0.85 VM_TEXT_AREA_FRAC=0.05
VM_BLACK_PIX_TH=0.02` (see `TEST-LOG.md` for why). See `ARCHITECTURE.md` for the keep/cut
rationale vs the previous skill.

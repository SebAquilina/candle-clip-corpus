# Second-by-second understanding upgrade (describe v2)

**Goal:** richer, more matchable per-second understanding of each clean window than the generic
BLIP-base caption ("a candle on a table") — to improve narration→clip matching.

**Chosen approach (user-directed):** teams of **Claude vision-agents**. Claude's vision beats
any locally-runnable captioner on a 4-core CPU box, and the corpus is only 671 windows. The
local BLIP path stays as an offline fallback in `describe.py`.

**Granularity:** **literal per-second, but context-aware.** Each window's frames (1/sec) are
handed to ONE agent so it sees the whole action arc; it then writes a description for EACH
second informed by that context (not isolated frame captions), plus a structured rollup.

## Pipeline (`describe_v2.py`, runs off the pre-downloaded clip cache)
1. `download_clips.py` → cache every window at `outputs/clip_cache/<key>.mp4` (key = `<vid>_<start>`).
2. `describe_v2.py extract` → 1 frame/sec → `outputs/describe_v2/frames/<key>/sec_NN.jpg`.
3. `describe_v2.py worklist` → windows needing description (frames + transcript + title + niche).
4. **Teams of agents** each take a slice; per window they Read the frames in order and Write
   `outputs/describe_v2/desc/<key>.json`:
   - `seconds[]` — one context-aware description per second
   - `summary` — one sentence
   - `embed_text` — dense, action-led, object-rich matching text (the key field)
   - `tags` — action, stage, tools[], materials[], container, colors[], setting, on_screen_text, person_visible
   - `qc_flags` — person_or_face / on_screen_text (a backstop: the corpus should be clean)
5. `describe_v2.py merge` → folds into corpus records as **additive** fields: `embed_text_v2`,
   `summary_v2`, `tags_v2`, `seconds_v2`, `qc_v2` (BLIP `embed_text` + `transcript` untouched).
6. Republish corpus to `main` via the additive PR pattern (SOP §4).

## Skill changes this drives
- **corpus-builder:** the describe stage gains an agent-vision path (emit frames + worklist →
  agents → merge). BLIP remains the offline fallback. Channel/title already captured at ingest.
- **video-maker:** matcher prefers `embed_text_v2` when present (richer) and can use `tags_v2`
  for structured boosts; `shared_library` surfaces the v2 fields. Falls back to BLIP `embed_text`
  for any window not yet upgraded, so it works mid-migration.

## Status
Pilot (6 windows) validating schema/quality, then scale to teams over all framed windows as
the download grind completes (resumable; both grinds auto-restart on sandbox reaping).

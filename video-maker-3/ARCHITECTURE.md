# video-maker-3 — architecture & redesign rationale

Streamlined from `yta-video-maker-2` (the cascade-hardened skill). That skill had ~40 modules
and a 116 KB SKILL.md spanning live YouTube discovery, Gemini grounding, Pexels fallback,
Hungarian global assignment, corpus building/sync, and multi-pass vision QC. video-maker-3
keeps only the path needed now that a **clean, pre-built corpus** (`outputs/shared_db_v2`)
is the single source, and rewrites the matcher + renderer for the brief.

## The pipeline (everything else was cut)
```
TTS(script)            tts_provider.py  (69labs | EdgeTTS)            [kept verbatim]
  → align              app/services/v2/align.py                       [kept verbatim]
  → sections           section_planner.sentences_from_words/sections_from_sentences
  → shortlist          matcher.build_worklist  (vision + transcript + action rerank)   [NEW]
  → [Claude picks]     match_worklist.json → match_decisions.json     [the in-loop step]
  → assemble           section_planner.plan_from_worklist  (≤2 uses, non-consec, concat) [REWORKED]
  → materialize        youtube.download_segment + duration_ladder.fit_clip              [SLIMMED]
  → render             render.py  (concat + atomic mux + loudnorm)    [REWRITTEN, no freeze]
  → validate           validate_render.py + clip_checks.py            [kept ~verbatim: the gate]
```

## KEEP (verbatim or near)
- `tts_provider.py`, `tts_69labs.py`, `tts.py` — TTS dispatch (69labs → EdgeTTS), resumable.
- `v2/align.py` — faster-whisper word timings **+ known-text forced-alignment fallback** so it
  works with no ASR installed.
- `v3/embeddings.py` (+ bundled `assets/potion-base-8M`), `action_rerank.py`, `niche_vocab.py`
  — the **offline** semantic pre-ranker that shortlists candidates for Claude.
- `v3/shared_library.py` — reads the `per_second_v1` corpus, keeping **vision** (`embed_text`)
  and **transcript** separate (one line added: surface the source `video_title`).
- `clip_checks.py`, `validate_render.py` — the deterministic detectors + the **non-skippable
  final gate** (black / freeze / AV-skew / overlay-text / face). The "just in case" check.

## REWORKED / NEW
- `v3/section_planner.py` — dropped the old phrase-concept handoff + the use-once cosine fill;
  added `sections_from_sentences`, the **vision+transcript+title** flow, and the assembler:
  best-clip-first concat, **≤2 uses, never consecutive, no freeze**, with a `materialize`
  callback so only clips that actually download+pass-QC are placed. `assembly_report` audits it.
- `v3/matcher.py` (NEW) — combined `w_vision*cos(vision) + w_transcript*cos(transcript)` +
  action rerank → top-K worklist; tolerant `load_decisions`. The Claude touchpoint.
- `render.py` (NEW, replaces render_video.py's 613 lines) — concat pre-materialized segments +
  atomic mux. **Removed:** last-frame-hold/freeze, `_still_has_face` freeze guards, the
  from-scratch `_source_fresh` YouTube re-sourcing, the borrow/cascade `__hold__` sentinel, and
  the black-panel fallback. Kept: atomic `.muxing.mp4` + `os.replace`, `loudnorm`, `+faststart`,
  blackdetect.
- `v3/duration_ladder.py` — `fit_clip` reduced to **trim-from-start** (on the matched moment) or
  **natural length** + one credit overlay; the slow-mo / Ken-Burns / freeze ladder is gone.
- `app/services/youtube.py` — reduced to `download_segment` (forces avc1 so cv2 can read it) +
  `fetch_channel`; removed the top-level `google.genai` import, `search`, `find_segments`.
- `app/config.py` — minimal (drops pydantic + Gemini/Pexels/CORS); just `storage_path`.

## CUT (with the discovery model)
`gemini_discovery, relevance_verifier, query_rewriter, sourcing_v2, topic_library,
hungarian_assigner, cost_matrix, pool_v3, pipeline_v3, pipeline_v13, corpus_builder,
drive_sync, git_sync, transcript_probe, vision_qc(_v13), claude_qc, pexels(_fallback,_image),
v2/shot_planner, backfill_signals, eval_matching, v14/v15 dumps`, and all their tests.

## One live-YouTube touchpoint remains, by necessity
The corpus stores window **metadata** (`url, start_s, end_s, vision, transcript`), not clip
bytes. To render a matched window we download exactly that range once
(`download_segment` + EJS + cookies) and cache it under `state/runs/<topic>/raw/`. This is a
small deterministic fetch per placed window — not search, not a pool.

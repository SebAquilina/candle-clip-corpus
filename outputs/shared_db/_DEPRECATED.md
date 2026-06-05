# ⚠️ DEPRECATED — do not use this corpus

`outputs/shared_db/` is the **old** corpus, built from low-res YouTube storyboard sprites.
It leaks faces and on-screen text (watermarks, captions) because it was never validated
against real frames. **It must not be used for video-making.**

**Use `outputs/shared_db_v2/` instead** — every window there was verified face-free and
text-free second-by-second (YuNet + multi-region OCR), with real per-second vision +
transcript descriptions.

This directory is retained ONLY as the input list for `corpus-revamp/reclassify.py`
(it defines which source videos + windows to re-scan). When the skill is overhauled,
point its loader exclusively at `shared_db_v2/` and treat `shared_db/` as read-only legacy.

The skill already auto-prefers `shared_db_v2/` (see skill-patches v25.3); this note makes
the intent explicit and permanent.

# yta-video-maker-2 — v25.3 consume the v2 corpus + dark-footage knob

Wires the skill to actually read the revamped per-second corpus, and makes the dark-footage
black gate tunable.

## Changes

1. **`shared_library.py` — read `shared_db_v2`.** `load_index` now recognises the per-second
   schema (`schema: "per_second_v1"`): every window in a v2 record is a vetted clean survivor
   (it already passed the second-by-second face+OCR purge and the description face backstop),
   so none of the v1 `is_step`/`flagged`/`face_score` logic applies. The window-level
   `embed_text` (**vision**, what is on screen) and `transcript` (**speech**) are read
   **separately** so the matcher can weigh them independently. `_store_dir` now auto-prefers
   `shared_db_v2/` when present, falling back to the old `shared_db/`. v1 loading is unchanged
   (verified: old corpus still loads 220 videos / 5,499 moments). [`changes.diff`](./changes.diff), +30/−1.

2. **`validate_render.py` — dark-footage knob.** The black/freeze/av thresholds are now
   env-overridable (`VM_BLACK_PIX_TH`, `VM_MAX_BLACK_SEC`, `VM_FREEZE_PIX_TH`, …). Genuinely
   dark, moody candle B-roll can read as "black" at the default 0.10 pixel threshold; lower
   `VM_BLACK_PIX_TH` to ~0.05 so only true black fails. Defaults unchanged (strict).

Shipped in `ytavideomaker-2.0-cascade-hardened-v3.skill` (now carries v25.1 + v25.2 + v25.3).

## How the skill picks up the new corpus

Once the `corpus-revamp` grind has produced `outputs/shared_db_v2/`, point the skill at it
(or rely on auto-detection):

```
export YTA_SHARED_DB=/path/to/candle-clip-corpus/outputs/shared_db_v2
```

Each clean window is offered to the matcher with `talking_head=false`, its vision caption as
`embed_text`, and the speaker's words as a separate `transcript` field.

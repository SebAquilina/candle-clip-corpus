# yta-video-maker-2 — v25.1 local-run hardening

Eight surgical fixes so the default **YouTube-only / no-Gemini** path runs clean end-to-end on
a fresh machine. No change to a correctly-configured Gemini run; everything degrades gracefully
and stays env-overridable.

Applied against the **`ytavideomaker-2.0-cascade.skill`** bundle (the canonical post-update
release). Re-bundled as `ytavideomaker-2.0-cascade-hardened.skill` (same 82-file layout, same
skill name, installs over the old one).

Each fix surfaced during a live run of the skill against the candle corpus — see the unified
diff in [`changes.diff`](./changes.diff) (+181 / −33 across 10 files).

## The eight fixes

| # | File | What broke | What changed |
|---|---|---|---|
| 1 | `make_video.sh` | Stripping `^_`/`_$` *before* `cut -c1-40` let the truncation re-introduce a trailing `_` at the 40th char | Order swapped: collapse → cut → strip |
| 2 | `scripts/bootstrap.sh` | Copied only top-level `*.py` to `$WS/scripts`; runners’ `import app.services.*` then failed and the runner had to be launched from the skill dir | Also copies the whole `scripts/app/` package tree |
| 3 | `scripts/app/services/tts_provider.py` | `import tts_69labs` failed — the module is at `app/services/tts_69labs.py`, not on `sys.path` | Imports `from app.services import tts_69labs`, falls back to the flat import for the legacy layout |
| 4 | `scripts/app/services/v3/query_rewriter.py` | `rewrite_shots` called Gemini unconditionally → crash on the default YouTube-only path | Lazy Gemini import; default path uses a local narration→keyword rewrite (the corpus matcher embeds the narration itself, so no API needed). Opt-in Gemini path still works |
| 5 | `scripts/_autobootstrap.py` | Probed `faster_whisper` + `google.genai`, then refused to bootstrap without `GOOGLE_API_KEY` / `PEXELS_API_KEY` — even in YouTube-only mode | Probes restricted to what the YT-only path actually needs (`yt_dlp`, `edge_tts`, `cv2`, `pytesseract`, …); keys required only when `YTA_YOUTUBE_ONLY=0` |
| 6 | `scripts/app/services/v2/align.py` + `pipeline_v3.py` | `transcribe_words` hard-imported `faster_whisper`; on read-only mounts where ctranslate2 can’t install, plan crashed | `transcribe_words` returns `[]` if ASR missing. New `synthesize_word_timings(text, duration)` spreads the script across the measured audio duration; `pipeline_v3.plan` wires it in (Rule v17.4) |
| 7 | `scripts/app/services/v3/library_match.py` | Pure-Python double loop over ~148 shots × ~6,200 moments couldn’t finish in the per-window budget | Vectorized cosine: one normalized matmul `Q @ S.T`. **Bit-identical results** verified vs the original pure path; falls back to it if numpy is unavailable |
| 8 | `scripts/app/services/v3/embeddings.py` | Default `model2vec` floor `0.32` was too tight for short corpus descriptions vs full narration sentences — most shots dropped to per-shot YT search | Tuned to `0.18` against the real candle corpus → ~95% of shots sourced from the corpus (Rule v26.0). Env override (`YTA_LIBRARY_MIN_SIM_FLOOR`) unchanged |

## Verification

Run against the patched modules using the workspace venv (`/tmp/yta_ws/.venv` —
model2vec + numpy present):

```
[bug1 slug]   no leading/trailing _, ≤40 chars on tricky titles                    PASS
[bug2 boot]   $WS/scripts importable end-to-end after bootstrap copy               PASS
[bug3 tts]    from app.services import tts_69labs → ok; default provider=edgetts  PASS
[bug4 rewr]   default path (no key) yields ShotQuery w/ scene_description=narr     PASS
              opt-in path without a key degrades to local rewrite (no crash)       PASS
[bug5 boot]   probes don't require faster_whisper/genai; keys only when YT_ONLY=0  PASS
[bug6 align]  synthesize_word_timings → 2-sentence boundaries land at 0..dur       PASS
              transcribe_words([]) when faster_whisper import fails                PASS
[bug7 match]  numpy vs pure-Python results identical (3/3 shots, same cosines)     PASS
[bug8 thr]    backend=model2vec → thresholds (min=0.50, floor=0.18)                PASS
[pipeline]    pipeline_v3.plan wired to synthesize_word_timings fallback           PASS
```

The hardened `.skill` bundle was extracted fresh and grep-verified to confirm all 8 patches
survived the rebuild (`Reading patches from extracted bundle: 1/1/1/1/2/1/1/1`).

## Why so many “Gemini” mentions surfaced during the run

The skill **states** v25.0 / v26.0: YouTube-only, no Gemini in the default path. But four
legacy coupling points still called or required it:

1. `query_rewriter.rewrite_shots` — top-level `from google import genai` + unconditional call.
2. `_autobootstrap.is_bootstrapped` — probed `from google import genai`.
3. `_autobootstrap.auto_bootstrap_if_needed` — refused to bootstrap without `GOOGLE_API_KEY`.
4. `make_video.sh` header — “Required env: GOOGLE_API_KEY, PEXELS_API_KEY”.

All four are now gated on `YTA_YOUTUBE_ONLY` (default `1`). The skill behaves as advertised.

## Install

Replace the existing skill bundle with `ytavideomaker-2.0-cascade-hardened.skill`. The skill
name (`yta-video-maker-2`) is unchanged so it overlays cleanly. Existing workspaces will pick
up the fixes on the next bootstrap (or once the patched `scripts/` is copied in by
`bootstrap.sh`).

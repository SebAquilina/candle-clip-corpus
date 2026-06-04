# video-maker-3 — mock-video acceptance test & iteration log

Acceptance test (the brief's Task C): drive a ~10-minute "Top 10 Candle Making Tricks" video
through the skill end-to-end, and **fix the skill on every failure so it can't recur**. The
script is `references/mock_script_top10.md`; narration via EdgeTTS (no 69labs creds in this
environment). Each issue below was found by running the real pipeline and fixed in the skill.

## Pipeline proven end-to-end
`plan` (TTS → align → 59 speech-timed sections → vision+transcript shortlist) → **two editor
agents** picked the best clip(s) per section from the worklist (judging vision + transcript in
the context of the title, rejecting off-topic false matches like an HP printer / fireplace /
red bird) → `build` (assemble → materialize → render) → `validate` (final gate).

Assembly on the real output held every rule: **59 shots, 38 distinct clips, max 2 uses, 0
consecutive, best-clip-first concat, no freeze** (`assembly_report` asserted in `cmd_build`).

## Fixes the test surfaced (each folded back into the skill)

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | EdgeTTS crashed: `CERTIFICATE_VERIFY_FAILED` | This environment's egress proxy presents a self-signed CA; edge-tts/aiohttp verify against **certifi**, which lacks it | Documented in `bootstrap.sh`/here: append the egress CA to certifi (env-specific). yt-dlp already skips cert checks. |
| 2 | Two bogus sections ("~150 wpm", "B-roll window") | `read_script` dropped only the *line* starting "Persona:"; the multi-line front-matter block's other lines leaked in as narration | `read_script` now filters front-matter at the **paragraph** level (`_META_RE` over the joined block). |
| 3 | Every download failed instantly | `download_segment` passed `--nocheckcertificate` (Python-API spelling) to the yt-dlp **CLI** | Use `--no-check-certificate`. |
| 4 | Downloads slowed then hard-blocked (HTTP 429/503) | The shared corpus account was heavily fetched (corpus build + this run); YouTube rate-limited it | Added `--retries/--extractor-retries/--retry-sleep/--sleep-requests` (env-tunable) so it backs off instead of escalating; added **`VM_CACHED_ONLY`** to assemble from already-downloaded windows when the network is throttled. |
| 5 | 75 valid downloads → only 22 placed (churn) | `mostly_black` used `pix_th=0.10`/60%, which reads **dark candle footage** as "black" | Count only near-pure-black pixels (`pix_th=0.02`) at a high `0.85` fraction — only truly corrupt/truncated clips trip it. Verified **75/75** cached clips now pass. |
| 6 | Build **hung** (cv2 at 12% CPU, no progress) | A C-level YuNet hang in the per-clip face check, with no timeout | Per-clip face/text QC runs in a **subprocess with a timeout** (hang-proof); black/validity is the always-on cheap check. |
| 7 | `render` always failed: "segment concat failed" | `_concat` built the ffmpeg command but **never appended the output path** | Append `str(out)`. 59 segments now concat + mux into the final mp4. |
| 8 | Final gate **FAIL: 19 face-frames** | The corpus per-second purge misses faces that flicker **between** its samples; the dense gate scan (3 fps) catches them | Re-enabled per-clip face/text QC by default, now **gate-aligned** (same detector + density) and cached per window, so the build skips face-tainted windows during assembly and reaches for the next-best face-free clip — the gate is the backstop, not the only catch. |

## Known limitations (honest)
- **Download throttling is environmental.** The shared account is rate-limited; the delivered
  build assembles from the **75 windows already cached** before the block. On a fresh account
  (or once cooled), `make.py build` (no `VM_CACHED_ONLY`) sources all 59 sections' best matches.
- **Corpus isn't 100% face/text-free.** A few windows carry brief faces or stylized/cursive
  channel logos the OCR/per-second purge missed. The skill now catches these per-clip (face/
  text QC) AND at the final gate. A durable corpus-side fix (re-scan every window with the
  dense gate detector and re-flag) is the recommended follow-up to `clip-corpus-builder`.
- **Run length** tracks the script's spoken length (~5 min here at EdgeTTS's rate), not a hard
  10 min; feed a longer script for longer output.

## Final gate (must exit 0 to ship)
`black PASS · freeze PASS · av-skew PASS (0.01s) · overlay-text PASS` were green on the first
render; the face check failed (fix #8) and is being re-validated after the face-QC rebuild.

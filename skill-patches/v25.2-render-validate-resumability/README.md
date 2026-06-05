# yta-video-maker-2 — v25.2 render/validate resumability

Fixes the *other* class of failure from the last run: the render and the non-skippable
post-render gate had **no resumability or time budget**, so on a box with short execution
windows they burned dozens of windows on kills, lingering children, a **corrupt no-moov-atom
mux**, and a `validate_render` whole-file OCR/face scan (~300 s on a 16-min video) that could
never finish. None of these weaken a single quality threshold — they let the *same* checks
finish across windows without redoing work.

Applied on top of [v25.1](../v25.1-local-run-hardening); both are in the shipped bundle
`ytavideomaker-2.0-cascade-hardened-v2.skill`. See [`changes.diff`](./changes.diff)
(+165 / −6 across `render_video.py` + `validate_render.py`; `make_video.sh` and `SKILL.md`
also get the gate loop + version note).

## The three fixes

| # | File | What broke | What changed |
|---|---|---|---|
| 1 | `render_video.py` | the final mux wrote straight to the output path; a mux killed mid-write left a corrupt, unplayable no-moov-atom file | **atomic mux** — encode to `*.muxing.mp4` then `os.replace`; the previous good output survives until the rename, `+faststart` added |
| 2 | `render_video.py` | the silent concat of all segments was rebuilt on every resumed run | **reuse a complete silent concat** when it already exists and its duration matches the audio (segment-skip resume already existed) |
| 3 | `validate_render.py` | one-shot whole-file scan; the ~300 s text/face pass couldn't finish in a short window and restarted from zero every time | **opt-in resumable mode** (`VM_VALIDATE_BUDGET_SEC>0`, `VM_VALIDATE_CHUNK_SEC`): black/freeze/av run once, then text/face is scanned in **persisted chunks** (using the *canonical* `clip_checks` scan functions on each sub-clip), exiting **75** ("run again") until the whole file is covered. `make_video.sh` loops the gate until a terminal verdict. |

Design guarantees:
- **Identical verdict.** Per-chunk text/face uses the same `clip_checks.scan_video_text` /
  `scan_video_talking_head` (offset by the chunk start) and the same thresholds/exit codes
  as the one-shot path. Verified: resumable exit == one-shot exit on real clips.
- **No livelock.** The chunk loop always does **≥1 chunk per call** before honouring the
  budget, so a budget smaller than one chunk still makes progress (overshoots by ≤1 chunk).
- **Crash-safe & backward-compatible.** State persists atomically off the output mount,
  keyed by file size+mtime; **default `VM_VALIDATE_BUDGET_SEC=0` keeps the exact one-shot
  behaviour** (75 never returned).

## Validation performed (on real prior-run footage)

```
resumable ≡ one-shot:   seg_000 -> exit 2 == 2  MATCH ;  seg_030 -> exit 2 == 2  MATCH
resume loop (tiny budget, chunk=2s):
   call1 black done ->75 ; call2 freeze ->75 ; call3 1/4 ; call4 2/4 ; call5 3/4 ;
   call6 -> TERMINAL exit 2 (== one-shot)            # forward progress, correct verdict
livelock bug found and fixed (budget < one chunk previously stuck at 0/N).
```

> Note: the resumability is a robustness/interruptibility win that matters most under tight
> execution windows; in a normal environment the gate already completes. The corpus
> contamination that caused faces/text to ship is fixed separately by `corpus-revamp/`.

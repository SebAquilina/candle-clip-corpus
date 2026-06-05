#!/usr/bin/env bash
# enrich_v2.sh — orchestrate the describe_v2 enrichment for the existing corpus.
#
# Phase 1 (this script): download missing clips, extract frames, emit AGENT BATCHES.
# Phase 2 (Claude, OUTSIDE this script): per batch, spawn a vision-agent that follows
#         outputs/describe_v2/AGENT_INSTRUCTIONS.md and writes outputs/describe_v2/desc/<key>.json.
# Phase 3 (this script when re-invoked with --merge): merge desc into records and apply qc cleanup.
#
# Idempotent + resumable. Re-run any time to enrich newly-ingested windows.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CMD="${1:-status}"

# always available: status, extract, worklist, next_batches, merge, apply_qc
case "$CMD" in
  status)
    python3 describe_v2.py status
    echo
    python3 apply_qc_v2.py --status
    ;;

  download)
    : "${REVAMP_COOKIES:?set REVAMP_COOKIES to fresh youtube cookies file}"
    echo "[enrich_v2] downloading any missing clips for the corpus..."
    python3 download_clips.py
    ;;

  extract)
    python3 describe_v2.py extract
    ;;

  batches)
    : "${DV2_BATCH:=12}"; : "${DV2_MAXNEW:=48}"
    export DV2_BATCH DV2_MAXNEW
    echo "[enrich_v2] emitting agent batches (BATCH=$DV2_BATCH, MAXNEW=$DV2_MAXNEW):"
    python3 describe_v2.py next_batches
    echo
    echo "next: SPAWN ONE AGENT PER BATCH FILE (Claude vision); each agent should follow"
    echo "      outputs/describe_v2/AGENT_INSTRUCTIONS.md. Then run: $0 merge"
    ;;

  merge)
    python3 describe_v2.py merge
    python3 apply_qc_v2.py
    echo
    echo "[enrich_v2] merge + qc cleanup complete. Commit the corpus records:"
    echo "  git add outputs/shared_db_v2/records/ && git commit -m 'corpus: describe_v2 update'"
    ;;

  all)
    "$0" download
    "$0" extract
    "$0" batches
    echo
    echo "Now spawn one Claude vision-agent per batch file above (read AGENT_INSTRUCTIONS.md)."
    echo "When agents finish, run: $0 merge"
    ;;

  *)
    cat <<EOF
usage: $0 {status|download|extract|batches|merge|all}

  status     show clip cache + describe + qc tallies
  download   pre-download every corpus window's clip into outputs/clip_cache/ (resumable)
  extract    1 frame/sec from each cached clip -> outputs/describe_v2/frames/<key>/
  batches    emit JSON batch files (one per agent) under outputs/describe_v2/batches/
  merge      fold finished outputs/describe_v2/desc/<key>.json into records, apply qc cleanup
  all        download + extract + batches; then operator spawns agents; then run merge

Env vars:
  REVAMP_COOKIES   path to fresh YouTube cookies (required for download)
  DV2_BATCH        windows per agent batch (default 12)
  DV2_MAXNEW       max new windows per batches call (default 48)
EOF
    ;;
esac

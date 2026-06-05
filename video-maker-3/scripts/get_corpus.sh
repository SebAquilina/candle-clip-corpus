#!/usr/bin/env bash
# get_corpus.sh — fetch an EXTERNAL, niche-specific clip corpus for the (general) video maker.
#
# The video-maker skill is niche-agnostic; the CORPUS defines the niche and lives in its OWN
# GitHub repo (built/published by the clip-corpus-builder skill). The user supplies that repo
# URL at runtime — it is NOT baked into the skill. This clones it and prints the exports.
#
# Usage:  bash scripts/get_corpus.sh <github-repo-url> [dest_dir]
# Then:   eval "$(bash scripts/get_corpus.sh <url>)"   # to set YTA_SHARED_DB / VM_NICHE
set -e
URL="${1:-}"; DEST="${2:-$PWD/corpus_repo}"
if [ -z "$URL" ]; then
  echo "usage: get_corpus.sh <github-repo-url> [dest_dir]" >&2; exit 1
fi
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --ff-only -q >&2 || true
else
  git clone --depth 1 "$URL" "$DEST" >&2
fi
SDB="$DEST/outputs/shared_db_v2"
if [ ! -d "$SDB/records" ]; then
  echo "ERROR: no outputs/shared_db_v2/records in $DEST (is this a clip-corpus-builder repo?)" >&2
  exit 2
fi
SDB_ABS="$(cd "$SDB" && pwd)"
# derive the niche from the corpus (most common record niche) so the skill stays niche-agnostic
NICHE="$(python3 - "$SDB" <<'PY' 2>/dev/null || true
import sys, glob, json, collections
c=collections.Counter()
for p in glob.glob(sys.argv[1]+"/records/*.json"):
    try: c[(json.load(open(p)).get("niche") or "").strip()]+=1
    except Exception: pass
c.pop("", None)
print(c.most_common(1)[0][0] if c else "")
PY
)"
echo "# corpus: $(ls "$SDB"/records/*.json | wc -l) records at $SDB_ABS" >&2
echo "export YTA_SHARED_DB=$SDB_ABS"
[ -n "$NICHE" ] && echo "export VM_NICHE=\"$NICHE\""

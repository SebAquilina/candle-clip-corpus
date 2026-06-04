#!/usr/bin/env bash
# video-maker-3 setup — venv + python deps + a note on system binaries.
# Streamlined: no Gemini/Pexels, no yt-dlp cookie auto-discovery, no YouTube smoke test.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "== venv =="
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip

echo "== python deps =="
pip install -q -r requirements.txt
# CPU-only torch is NOT required (the embedder is model2vec/numpy; whisper is optional).

# Behind an SSL-intercepting egress proxy (self-signed CA in the chain), edge-tts/aiohttp
# verify against certifi, which lacks the proxy CA -> TTS fails with CERTIFICATE_VERIFY_FAILED.
# If such a CA is present on this host, append it to certifi so EdgeTTS can connect. No-op
# elsewhere. (yt-dlp already skips cert checks via --no-check-certificate.)
for CA in /usr/local/share/ca-certificates/egress-gateway-ca-production.crt \
          /usr/local/share/ca-certificates/*.crt; do
  [ -f "$CA" ] || continue
  CB="$(python -c 'import certifi;print(certifi.where())' 2>/dev/null)" || break
  if [ -n "$CB" ] && ! grep -qFf "$CA" "$CB" 2>/dev/null; then
    cat "$CA" >> "$CB"; echo "  appended $(basename "$CA") to certifi bundle"
  fi
done

echo "== system binaries (need to be on PATH) =="
for b in ffmpeg ffprobe tesseract deno node; do
  if command -v "$b" >/dev/null 2>&1; then echo "  ok: $b"; else echo "  MISSING: $b"; fi
done
cat <<'EOF'

If any are MISSING, install them:
  ffmpeg + ffprobe   : apt-get install -y ffmpeg
  tesseract          : apt-get install -y tesseract-ocr        (final-gate text detector)
  deno + node        : needed by yt-dlp-ejs to download corpus windows (YouTube `n` challenge)

Then point the skill at the clean corpus and (for downloads) fresh cookies:
  export YTA_SHARED_DB=/path/to/outputs/shared_db_v2
  export VM_COOKIES=/path/to/fresh/youtube_cookies.txt
  python scripts/make.py --selftest
EOF
echo "== done =="

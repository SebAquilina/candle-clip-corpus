"""69labs Text-to-Speech client — TTS only.

69labs (https://69labs.vip) is an ElevenLabs-compatible TTS platform. The job
flow is async: create a job, poll for completion, download the mp3.

Auth: Authorization: Bearer vk_...  (API keys start with vk_)

This module covers ONLY text-to-speech (no image/video/motion-graphics).

Public functions:
- synthesize(api_key, text, voice_id, out_path, ...)  -> dict with job + file info
- list_voice_clones(api_key, scope="mine"|"library")  -> list of cloned voices
- synthesize_with_clone(api_key, text, voice_clone_id, out_path, ...)
"""
from __future__ import annotations
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://69labs.vip"

# Status values from the docs
TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED"}
TERMINAL_CENSORED = {"CENSORED"}
IN_PROGRESS = {"PENDING", "PROCESSING", "FINALIZING"}


class TTSError(RuntimeError):
    def __init__(self, message, code=None, http_status=None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def _request(method, path, api_key, body=None, raw_response=False, timeout=60):
    """Low-level HTTP. Returns parsed JSON dict, or raw bytes if raw_response."""
    url = f"{BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        # Cloudflare in front of 69labs rejects the default Python-urllib UA
        # with error 1010. Present a normal browser signature.
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            if raw_response:
                return payload, resp.headers
            return json.loads(payload.decode("utf-8")) if payload else {}
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", "replace")
        code = None
        msg = body_txt
        try:
            j = json.loads(body_txt)
            code = j.get("code")
            msg = j.get("error", body_txt)
        except Exception:
            pass
        raise TTSError(f"HTTP {e.code} on {method} {path}: {msg}", code=code, http_status=e.code)
    except urllib.error.URLError as e:
        raise TTSError(f"network error on {method} {path}: {e.reason}")


def create_job(api_key, text, voice_id, voice_provider="elevenlabs",
               model_id=None, voice_name=None, voice_settings=None,
               minimax_settings=None):
    """POST /api/v1/tts/generate. Returns {id, status, queuePosition}."""
    body = {"text": text, "voiceId": voice_id, "voiceProvider": voice_provider}
    if model_id:
        body["modelId"] = model_id
    if voice_name:
        body["voiceName"] = voice_name
    if voice_settings:
        body["voiceSettings"] = voice_settings
    if minimax_settings:
        body["minimaxSettings"] = minimax_settings
    return _request("POST", "/api/v1/tts/generate", api_key, body=body)


def get_status(api_key, job_id):
    """GET /api/v1/tts/status/:jobId."""
    return _request("GET", f"/api/v1/tts/status/{job_id}", api_key)


def download(api_key, job_id, out_path):
    """GET /api/v1/tts/download/:jobId -> writes mp3 to out_path."""
    payload, _headers = _request("GET", f"/api/v1/tts/download/{job_id}", api_key,
                                  raw_response=True, timeout=120)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return out


def synthesize(api_key, text, voice_id, out_path, *,
               voice_provider="elevenlabs", model_id=None, voice_name=None,
               voice_settings=None, minimax_settings=None,
               poll_interval=3.0, timeout_sec=300, on_progress=None):
    """Full flow: create job -> poll -> download. Returns a result dict.

    on_progress(status_dict) is called on each poll if provided.
    Raises TTSError on FAILED/CANCELLED, or returns a dict with status=CENSORED
    and the blocked chunks for the caller to handle.
    """
    job = create_job(api_key, text, voice_id, voice_provider=voice_provider,
                     model_id=model_id, voice_name=voice_name,
                     voice_settings=voice_settings, minimax_settings=minimax_settings)
    job_id = job.get("id")
    if not job_id:
        raise TTSError(f"no job id in response: {job}")

    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        st = get_status(api_key, job_id)
        last = st
        status = st.get("status")
        if on_progress:
            on_progress(st)
        if status in TERMINAL_OK:
            out = download(api_key, job_id, out_path)
            return {"ok": True, "job_id": job_id, "status": status,
                    "file": str(out), "bytes": out.stat().st_size, "status_detail": st}
        if status in TERMINAL_BAD:
            raise TTSError(f"job {job_id} ended with status {status}", code=status)
        if status in TERMINAL_CENSORED:
            # The caller decides whether to rewrite + retry-censored.
            return {"ok": False, "job_id": job_id, "status": "CENSORED",
                    "blocked_chunks": st.get("blockedChunks", []),
                    "retry_expires_at": st.get("retryExpiresAt"), "status_detail": st}
        time.sleep(poll_interval)
    raise TTSError(f"job {job_id} did not complete within {timeout_sec}s; last status={last}")


def list_voice_clones(api_key, scope="mine"):
    """GET /api/v1/voice-clones (scope='mine') or /library (scope='library')."""
    path = "/api/v1/voice-clones" + ("/library" if scope == "library" else "")
    data = _request("GET", path, api_key)
    return data.get("voiceClones", [])


def synthesize_with_clone(api_key, text, voice_clone_id, out_path, *,
                          model_id=None, speed=None, pitch=None, volume=None,
                          language_boost=None, poll_interval=3.0, timeout_sec=300,
                          on_progress=None):
    """POST /api/v1/voice-clones/generate then poll the standard TTS status/download.

    Voice clone generation uses MiniMax-style params (speed/pitch/volume/language_boost).
    """
    body = {"voiceCloneId": voice_clone_id, "text": text}
    if model_id: body["model"] = model_id
    if speed is not None: body["speed"] = speed
    if pitch is not None: body["pitch"] = pitch
    if volume is not None: body["volume"] = volume
    if language_boost: body["language_boost"] = language_boost
    job = _request("POST", "/api/v1/voice-clones/generate", api_key, body=body)
    job_id = job.get("id")
    if not job_id:
        raise TTSError(f"no job id in voice-clone response: {job}")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        st = get_status(api_key, job_id)
        if on_progress: on_progress(st)
        status = st.get("status")
        if status in TERMINAL_OK:
            out = download(api_key, job_id, out_path)
            return {"ok": True, "job_id": job_id, "status": status,
                    "file": str(out), "bytes": out.stat().st_size}
        if status in TERMINAL_BAD:
            raise TTSError(f"clone job {job_id} ended with status {status}", code=status)
        if status in TERMINAL_CENSORED:
            return {"ok": False, "job_id": job_id, "status": "CENSORED",
                    "blocked_chunks": st.get("blockedChunks", [])}
        time.sleep(poll_interval)
    raise TTSError(f"clone job {job_id} did not complete within {timeout_sec}s")

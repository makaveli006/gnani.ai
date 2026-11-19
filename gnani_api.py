"""
Small client for Gnani's Prisma (ASR) and Timbre (TTS) APIs.
Docs: https://docs.gnani.ai

Every response is saved to results/raw/ so running the evaluation again
does not spend credits again.
"""
import base64
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GNANI_API_KEY")
BASE_URL = "https://api.vachana.ai"
RAW_DIR = Path("results/raw")


def post_with_retry(url, tries=4, **kwargs):
    # the API allows one request at a time, and sometimes returns 5xx;
    # wait a bit and try again
    headers = {"X-API-Key-ID": API_KEY}
    for attempt in range(tries):
        start = time.perf_counter()
        r = requests.post(url, headers=headers, timeout=120, **kwargs)
        elapsed = time.perf_counter() - start
        if r.status_code == 402:
            raise RuntimeError("Out of credits (402)")
        if r.status_code == 429 or r.status_code >= 500:
            print(f"  {r.status_code}, retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
            continue
        return r, elapsed
    return r, elapsed


def transcribe(wav_path, name, model=None, language="en-IN"):
    """Send one audio file to Prisma. Returns the JSON response
    plus 'client_latency' (seconds, measured on our side)."""
    cache = RAW_DIR / "asr" / (model or "prisma-v2.5") / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    data = {"language_code": language}
    if model:
        data["model"] = model
    # read the bytes once, so a retry sends the full file again
    audio_bytes = Path(wav_path).read_bytes()
    r, elapsed = post_with_retry(
        f"{BASE_URL}/stt/v3",
        files={"audio_file": (Path(wav_path).name, audio_bytes, "audio/wav")},
        data=data,
    )

    try:
        result = r.json()
    except ValueError:
        result = {"success": False, "error": r.text[:200]}
    result["http_status"] = r.status_code
    result["client_latency"] = round(elapsed, 3)

    # cache good answers and client errors (they fail the same way every time);
    # server errors (5xx) are not cached so they are retried next run
    if result.get("success") or r.status_code in (400, 403, 413, 422):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def tts_payload(text, voice, language, model, sample_rate):
    return {
        "text": text,
        "voice": voice,
        "model": model,
        "language": language,
        "speed": 1.0,
        "audio_config": {
            "sample_rate": sample_rate,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }


def synthesize(text, out_wav, voice="Kaveri", language="en-IN", model="timbre-v2.5", sample_rate=24000):
    """Timbre REST call. Saves the wav and returns timing info."""
    out_wav = Path(out_wav)
    meta_path = out_wav.with_suffix(".json")
    if out_wav.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    r, elapsed = post_with_retry(
        f"{BASE_URL}/api/v1/tts/inference",
        json=tts_payload(text, voice, language, model, sample_rate),
    )
    if r.status_code != 200:
        return {"ok": False, "http_status": r.status_code, "error": r.text[:200]}

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(r.content)
    meta = {"ok": True, "total_latency": round(elapsed, 3)}
    meta_path.write_text(json.dumps(meta))
    return meta


def time_to_first_audio(text, voice="Kaveri", language="en-IN", model="timbre-v2.5", sample_rate=24000):
    """Timbre streaming (SSE) call. Only used to measure how fast
    the first audio chunk arrives, which is what a voice agent feels."""
    headers = {"X-API-Key-ID": API_KEY}
    start = time.perf_counter()
    first = None
    with requests.post(
        f"{BASE_URL}/api/v1/tts/sse",
        headers=headers,
        json=tts_payload(text, voice, language, model, sample_rate),
        stream=True,
        timeout=120,
    ) as r:
        if r.status_code != 200:
            return None
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event.get("audio") and first is None:
                first = time.perf_counter() - start
            if event.get("is_final"):
                break
    return round(first, 3) if first else None

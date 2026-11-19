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

import numpy as np
import requests
import soundfile as sf
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
    """Timbre streaming (SSE) call. One call gives us everything:
    - ttfb: time until the first audio chunk arrives (what a caller feels)
    - total: time until the last chunk
    - the audio itself, saved as out_wav
    The timing is saved next to the wav as .json, so a second run costs nothing."""
    out_wav = Path(out_wav)
    meta_path = out_wav.with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    for attempt in range(4):
        start = time.perf_counter()
        r = requests.post(
            f"{BASE_URL}/api/v1/tts/sse",
            headers={"X-API-Key-ID": API_KEY},
            json=tts_payload(text, voice, language, model, sample_rate),
            stream=True,
            timeout=120,
        )
        if r.status_code == 402:
            raise RuntimeError("Out of credits (402)")
        if r.status_code == 429 or r.status_code >= 500:
            print(f"  {r.status_code}, retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
            continue
        break

    if r.status_code != 200:
        return {"ok": False, "http_status": r.status_code, "error": r.text[:200]}

    chunks, ttfb, error = [], None, None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        event = json.loads(line[5:])
        if "message" in event and "audio" not in event:
            error = event["message"]
            break
        if event.get("audio"):
            if ttfb is None:
                ttfb = time.perf_counter() - start
            chunks.append(base64.b64decode(event["audio"]))
        if event.get("is_final"):
            break
    total = time.perf_counter() - start

    if error or not chunks:
        return {"ok": False, "http_status": r.status_code, "error": error or "no audio"}

    # first chunk starts with a 44-byte wav header whose size field is a
    # "streaming" placeholder, so drop it and write a proper wav ourselves
    pcm = b"".join(chunks)[44:]
    audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, audio, sample_rate, subtype="PCM_16")

    meta = {"ok": True, "ttfb": round(ttfb, 3), "total": round(total, 3),
            "audio_seconds": round(len(audio) / sample_rate, 3)}
    meta_path.write_text(json.dumps(meta))
    return meta

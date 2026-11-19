"""
Record the sentences in recording_script.csv with your microphone.

    python record.py          record the ones not done yet
    python record.py h03      re-record one sentence

Press Enter to start, Enter again to stop. Saved as data/samples/own/<id>.wav
"""
import csv
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
OUT = Path("data/samples/own")


def record():
    chunks = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
        input("  recording... press Enter to stop")
    return np.concatenate(chunks)[:, 0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None

    with open("recording_script.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        path = OUT / f"{row['id']}.wav"
        if only and row["id"] != only:
            continue
        if path.exists() and not only:
            continue

        print(f"\n[{row['id']}]  {row['text']}")
        input("  press Enter and read the sentence")
        audio = record()
        sf.write(path, audio, SR, subtype="PCM_16")
        print(f"  saved {path} ({len(audio) / SR:.1f} s)")


if __name__ == "__main__":
    main()

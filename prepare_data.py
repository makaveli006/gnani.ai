"""
Builds the ASR test set from three local datasets:
  - LibriSpeech test-clean (US English, studio quality)
  - Common Voice 26 South Asian English
  - Common Voice 26 Irish English

Picked clips are converted to 16 kHz mono wav and copied to data/samples/.
Noisy and phone-quality versions are made from 10 of the Indian clips.
Everything is listed in data/manifest.csv.

Run once:  python prepare_data.py
"""
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from scipy.signal import resample_poly

load_dotenv()
LIBRI_DIR = os.getenv("LIBRISPEECH_DIR")
CV_SA_DIR = os.getenv("CV_SOUTH_ASIAN_DIR")
CV_IE_DIR = os.getenv("CV_IRISH_DIR")

OUT = Path("data/samples")
SR = 16000
SEED = 42

# ---- hand-picked Common Voice clips (number in common_voice_en_<n>.mp3) ----
# picked for clear, everyday sentences with correct transcripts, one per speaker

INDIAN = [
    "40921264", "20264170", "20405318", "27302134", "645517", "18387833",
    "25433114", "17425029", "19612757", "31790766", "17391991", "25038482",
    "20607251", "17283024", "19690292", "689462", "153333", "17434904",
    "17281454", "17247732", "1009486", "17783018", "17322377", "19595948",
    "496232", "26626458", "193310", "18841412", "612325", "22210611",
]

SHORT = [
    "18359293", "689461", "17758196", "17726956", "17434909",
    "22210624", "23394443", "22220440", "22069670", "22210467",
]

NUMBERS = [
    "17783003", "17726958", "544199", "496237", "18359260",
    "17259911", "612330", "22350455", "1009475", "36506656",
]

PROPER_NOUNS = [
    "23853393", "19673642", "18846092", "26587760", "22069689",
    "17758217", "24775870", "20405322", "18840660", "21765880",
]

IRISH = [
    "19941072", "39602140", "25650380", "26949296", "19853173",
    "18953293", "27165979", "18845497", "17576768", "17576777",
    "17403009", "17728604", "22112265", "202179", "17367840",
    "24986825", "20483338", "25167305", "591885", "3247",
]

# 10 of the Indian clips get noisy and phone versions
NOISE_BASE = [
    "40921264", "20264170", "27302134", "17425029", "689462",
    "17281454", "1009486", "17322377", "193310", "17434904",
]
SNR_LEVELS = [20, 10, 0]

# speakers NOT in the test set, mixed together to make background chatter
BABBLE = ["37024617", "38007655", "37032601", "36643257", "21671780", "39940785"]

N_LIBRI = 20
N_LONG = 5


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = resample_poly(audio, SR, sr).astype("float32")
    return audio


def save_wav(path, audio, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.99
    sf.write(path, audio, sr, subtype="PCM_16")


def read_cv_tsv(folder):
    rows = {}
    for split in ["test", "dev"]:
        with open(f"{folder}/{split}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                rows[r["path"]] = r
    return rows


def short_speaker_ids(rows):
    # client_id is a long hash, swap it for a short label like spk3
    ids = {}
    for r in rows.values():
        if r["client_id"] not in ids:
            ids[r["client_id"]] = f"spk{len(ids)}"
    return ids


def gender(r):
    g = r.get("gender", "")
    return "f" if g.startswith("female") else "m" if g.startswith("male") else ""


def add_cv_clips(manifest, folder, numbers, category, source):
    rows = read_cv_tsv(folder)
    spk = short_speaker_ids(rows)
    for n in numbers:
        name = f"common_voice_en_{n}.mp3"
        r = rows[name]
        audio = load_audio(f"{folder}/clips/{name}")
        out = OUT / category / f"{n}.wav"
        save_wav(out, audio)
        manifest.append({
            "id": f"{category}_{n}",
            "category": category,
            "source": source,
            "speaker": f"{source}_{spk[r['client_id']]}",
            "gender": gender(r),
            "duration": round(len(audio) / SR, 2),
            "reference": r["sentence"],
            "wav_path": out.as_posix(),
        })


def read_librispeech():
    # returns {speaker: [(utt_id, text, flac_path), ...]} in file order
    speakers = {}
    for trans in sorted(glob.glob(f"{LIBRI_DIR}/*/*/*.trans.txt")):
        folder = os.path.dirname(trans)
        with open(trans) as f:
            for line in f:
                utt, text = line.strip().split(" ", 1)
                spk = utt.split("-")[0]
                speakers.setdefault(spk, []).append((utt, text, f"{folder}/{utt}.flac"))
    return speakers


def libri_genders():
    genders = {}
    with open(f"{LIBRI_DIR}/../SPEAKERS.TXT", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            genders[parts[0]] = parts[1].lower()
    return genders


def add_librispeech(manifest):
    speakers = read_librispeech()
    genders = libri_genders()
    males = [s for s in speakers if genders[s] == "m"][: N_LIBRI // 2]
    females = [s for s in speakers if genders[s] == "f"][: N_LIBRI // 2]

    # clean_us: first utterance with 8-20 words from each speaker
    for s in males + females:
        utt, text, path = next(u for u in speakers[s] if 8 <= len(u[1].split()) <= 20)
        audio = load_audio(path)
        out = OUT / "clean_us" / f"{utt}.wav"
        save_wav(out, audio)
        manifest.append({
            "id": f"clean_us_{utt}", "category": "clean_us", "source": "librispeech",
            "speaker": f"librispeech_{s}", "gender": genders[s],
            "duration": round(len(audio) / SR, 2), "reference": text.lower(),
            "wav_path": out.as_posix(),
        })

    # long: join consecutive utterances of one chapter until 20-28 s (API limit is 30 s)
    other = [s for s in speakers if s not in males + females]
    gap = np.zeros(int(0.3 * SR), dtype="float32")
    for s in other[:N_LONG]:
        parts, texts = [], []
        for utt, text, path in speakers[s]:
            audio = load_audio(path)
            if sum(len(p) for p in parts) + len(audio) > 28 * SR:
                break
            parts += [audio, gap]
            texts.append(text.lower())
        audio = np.concatenate(parts)
        out = OUT / "long" / f"{s}_long.wav"
        save_wav(out, audio)
        manifest.append({
            "id": f"long_{s}", "category": "long", "source": "librispeech",
            "speaker": f"librispeech_{s}", "gender": genders[s],
            "duration": round(len(audio) / SR, 2), "reference": " ".join(texts),
            "wav_path": out.as_posix(),
        })


def make_babble(length, rng):
    babble = np.zeros(length, dtype="float32")
    for n in BABBLE:
        voice = load_audio(f"{CV_SA_DIR}/clips/common_voice_en_{n}.mp3")
        voice = np.tile(voice, length // len(voice) + 1)[:length]
        voice = np.roll(voice, rng.integers(length))
        babble += voice / (np.sqrt(np.mean(voice ** 2)) + 1e-9)
    return babble


def mix_at_snr(speech, noise, snr_db):
    # scale noise so that 10*log10(P_speech / P_noise) = snr_db
    p_speech = np.mean(speech ** 2)
    p_noise = np.mean(noise ** 2)
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return speech + noise * scale


def phone_quality(audio):
    # 8 kHz + 8-bit mu-law, like a normal phone call
    narrow = resample_poly(audio, 8000, SR)
    narrow = np.clip(narrow, -1, 1)
    mu = 255
    encoded = np.sign(narrow) * np.log1p(mu * np.abs(narrow)) / np.log1p(mu)
    encoded = np.round((encoded + 1) / 2 * mu) / mu * 2 - 1   # 256 levels
    decoded = np.sign(encoded) * ((1 + mu) ** np.abs(encoded) - 1) / mu
    return decoded.astype("float32")


def add_noisy_and_phone(manifest):
    rng = np.random.default_rng(SEED)
    base = {row["id"].split("_")[-1]: row for row in manifest if row["category"] == "accent_indian"}
    for n in NOISE_BASE:
        row = base[n]
        speech = load_audio(row["wav_path"])
        babble = make_babble(len(speech), rng)

        for snr in SNR_LEVELS:
            out = OUT / f"noisy_{snr}db" / f"{n}.wav"
            save_wav(out, mix_at_snr(speech, babble, snr))
            manifest.append({**row, "id": f"noisy_{snr}db_{n}", "category": f"noisy_{snr}db",
                             "wav_path": out.as_posix()})

        out = OUT / "phone_8k" / f"{n}.wav"
        save_wav(out, phone_quality(speech), sr=8000)
        manifest.append({**row, "id": f"phone_8k_{n}", "category": "phone_8k",
                         "wav_path": out.as_posix()})


def main():
    manifest = []
    add_librispeech(manifest)
    add_cv_clips(manifest, CV_SA_DIR, INDIAN, "accent_indian", "cv_south_asian")
    add_cv_clips(manifest, CV_SA_DIR, SHORT, "short", "cv_south_asian")
    add_cv_clips(manifest, CV_SA_DIR, NUMBERS, "numbers_spoken", "cv_south_asian")
    add_cv_clips(manifest, CV_SA_DIR, PROPER_NOUNS, "proper_nouns", "cv_south_asian")
    add_cv_clips(manifest, CV_IE_DIR, IRISH, "accent_irish", "cv_irish")
    add_noisy_and_phone(manifest)

    with open("data/manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    counts = {}
    for row in manifest:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    for cat, c in counts.items():
        print(f"{cat:16s} {c}")
    print("total", len(manifest))


if __name__ == "__main__":
    main()

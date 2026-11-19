"""
Part 2: evaluate Timbre (TTS).

    python eval_tts.py              run everything
    python eval_tts.py --limit 3    quick test on 3 sentences

For each sentence in tts_sentences.csv and each voice:
  1. Timbre makes the audio (cached in results/tts_audio/)
  2. Whisper (a different ASR, runs locally) writes down what it hears
  3. we score it:
       - intelligibility: WER/CER of Whisper's text vs what should be said
       - normalization: did numbers / dates / acronyms come out right
       - naturalness: DNSMOS predicted MOS (1-5)
       - speed: time to first audio, total time, RTF
       - reliability: empty, clipped, long silences, odd speaking rate
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from gnani_api import synthesize
from metrics import error_counts, matches_any

MODEL = "timbre-v2.5"
VOICES = {"en-IN": ["Kaveri", "Pranav"], "hi-en": ["Poorvi", "Kaveri"]}
AUDIO_DIR = Path("results/tts_audio")

# categories where the sentence has no digits or symbols, so WER is fair.
# number / date / acronym sentences are judged with the accept list instead,
# because Whisper may write "two lakh" as "2,00,000" and both are correct.
WER_CATEGORIES = ["plain", "indian_names", "tricky_words", "long_form", "hinglish"]

ORDER = ["plain", "numbers_raw", "numbers_prewritten", "dates_times", "phone_ids",
         "acronyms", "indian_names", "hinglish", "tricky_words", "email_url", "long_form"]


def load_whisper():
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel("small", device="cuda", compute_type="float16")
        list(model.transcribe(np.zeros(16000, dtype="float32"))[0])  # fails here if CUDA libs are missing
        return model
    except Exception as e:
        print(f"GPU not usable ({e}), using CPU")
        return WhisperModel("small", device="cpu", compute_type="int8")


def whisper_text(model, wav_path, hinglish=False):
    # a short Hinglish prompt nudges Whisper to write Hindi words in English letters
    prompt = "Aapka order dispatch ho gaya hai, kal tak deliver ho jayega." if hinglish else None
    segments, _ = model.transcribe(str(wav_path), language="en", beam_size=5,
                                   initial_prompt=prompt, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments)


def mos_scores(audio, sr):
    # DNSMOS (Microsoft): predicts the 1-5 score listeners would give.
    # p808_mos = overall quality, ovrl_mos = overall incl. noise/distortion
    from speechmos import dnsmos
    audio16 = np.clip(resample_poly(audio, 16000, sr), -1, 1).astype("float32")
    result = dnsmos.run(audio16, sr=16000)
    return round(float(result["p808_mos"]), 2), round(float(result["ovrl_mos"]), 2)


def audio_checks(audio, sr, n_words):
    loud = np.where(np.abs(audio) > 0.02)[0]
    if len(loud) == 0:
        return {"lead_silence": None, "trail_silence": None, "words_per_sec": 0,
                "clipped_pct": 0, "flags": "empty"}
    lead = loud[0] / sr
    trail = (len(audio) - loud[-1]) / sr
    speech = (loud[-1] - loud[0]) / sr
    rate = n_words / speech if speech > 0 else 0
    clipped = float(np.mean(np.abs(audio) >= 0.999)) * 100

    flags = []
    if len(audio) / sr < 0.3:
        flags.append("empty")
    if clipped > 0.1:
        flags.append("clipped")
    if lead > 1.0:
        flags.append("long_start_silence")
    if trail > 1.5:
        flags.append("long_end_silence")
    if n_words >= 4 and not 1.5 <= rate <= 4.5:
        flags.append("odd_speaking_rate")
    return {"lead_silence": round(lead, 2), "trail_silence": round(trail, 2),
            "words_per_sec": round(rate, 2), "clipped_pct": round(clipped, 3),
            "flags": ",".join(flags)}


def human_baseline(whisper):
    # the same checks on 30 real Indian English recordings from Part 1:
    # how good is Whisper itself, and how fast and how "good" do real people sound?
    m = pd.read_csv("data/manifest.csv")
    m = m[m.category == "accent_indian"]
    errors = words = 0
    rates, mos = [], []
    for _, row in m.iterrows():
        c = error_counts(row.reference, whisper_text(whisper, row.wav_path))
        errors += c["sub"] + c["del"] + c["ins"]
        words += c["ref_words"]
        audio, sr = sf.read(row.wav_path, dtype="float32")
        rates.append(audio_checks(audio, sr, c["ref_words"])["words_per_sec"])
        mos.append(mos_scores(audio, sr)[0])
    return errors / words, np.median(rates), np.mean(mos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with open("tts_sentences.csv", encoding="utf-8") as f:
        sentences = list(csv.DictReader(f))
    if args.limit:
        sentences = sentences[: args.limit]

    # 1. synthesize (cached)
    jobs = []
    for s in sentences:
        for voice in VOICES[s["language"]]:
            wav = AUDIO_DIR / f"{MODEL}_{voice}" / f"{s['id']}.wav"
            meta = synthesize(s["text"], wav, voice=voice, language=s["language"], model=MODEL)
            status = "ok" if meta.get("ok") else f"FAILED {meta.get('http_status')} {meta.get('error')}"
            print(f"{s['id']:5s} {voice:7s} {status}")
            jobs.append((s, voice, wav, meta))

    # 2. score
    print("\nloading Whisper...")
    whisper = load_whisper()

    rows = []
    for s, voice, wav, meta in jobs:
        if not meta.get("ok"):
            rows.append({"id": s["id"], "category": s["category"], "voice": voice,
                         "text": s["text"], "flags": f"api_error {meta.get('http_status')}"})
            continue
        audio, sr = sf.read(wav, dtype="float32")
        heard = whisper_text(whisper, wav, hinglish=s["language"] == "hi-en")
        mos, ovrl = mos_scores(audio, sr)
        counts = error_counts(s["expected_spoken"], heard)
        rows.append({
            "id": s["id"],
            "category": s["category"],
            "voice": voice,
            "text": s["text"],
            "whisper_heard": heard,
            **counts,
            "wer": round((counts["sub"] + counts["del"] + counts["ins"]) / counts["ref_words"], 3),
            "cer": round(counts["char_errors"] / counts["ref_chars"], 3),
            "accept_ok": matches_any(heard, s["accept"]) if s["accept"] else None,
            "mos": mos,
            "mos_ovrl": ovrl,
            "ttfb": meta["ttfb"],
            "total_time": meta["total"],
            "audio_seconds": meta["audio_seconds"],
            "rtf": round(meta["total"] / meta["audio_seconds"], 3),
            **audio_checks(audio, sr, counts["ref_words"]),
        })
        print(f"{s['id']:5s} {voice:7s} WER={rows[-1]['wer']:.2f} MOS={rows[-1]['mos']:.2f} {rows[-1]['flags']}")

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/tts_results.csv", index=False)
    ok = df[df["flags"].fillna("").str.contains("api_error") == False]

    # 3. summary per category and voice
    summary = []
    for cat in [c for c in ORDER if c in set(ok.category)]:
        for voice in sorted(set(ok[ok.category == cat].voice)):
            d = ok[(ok.category == cat) & (ok.voice == voice)]
            wer_cat = cat in WER_CATEGORIES
            acc = d.accept_ok.dropna()
            summary.append({
                "category": cat,
                "voice": voice,
                "n": len(d),
                "WER": round((d["sub"] + d["del"] + d["ins"]).sum() / d.ref_words.sum(), 3) if wer_cat else None,
                "CER": round(d.char_errors.sum() / d.ref_chars.sum(), 3) if wer_cat else None,
                "accept_pass": f"{int(acc.sum())}/{len(acc)}" if len(acc) else None,
                "MOS": round(d.mos.mean(), 2),
                "ttfb_s": round(d.ttfb.mean(), 2),
                "RTF": round(d.rtf.mean(), 3),
                "flagged": int((d["flags"].fillna("") != "").sum()),
            })
    summary = pd.DataFrame(summary)
    summary.to_csv("results/tts_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    # raw text vs text already written as words
    pairs = ok[ok.category.isin(["numbers_raw", "numbers_prewritten"])]
    if len(pairs):
        p = pairs.groupby(["category", "voice"]).accept_ok.agg(["sum", "count"]).reset_index()
        p.to_csv("results/tts_pairs.csv", index=False)
        print("\nRaw numbers vs written-out numbers (correct / total):")
        print(p.to_string(index=False))

    # overall speed
    print(f"\nTime to first audio: median {ok.ttfb.median():.2f}s, p90 {ok.ttfb.quantile(0.9):.2f}s")
    print(f"RTF: median {ok.rtf.median():.3f}, worst {ok.rtf.max():.3f}")
    print("MOS by voice:", ok.groupby("voice").mos.mean().round(2).to_dict())

    print("\nFailed normalization checks:")
    for _, r in ok[ok.accept_ok == False].iterrows():
        print(f"  {r.id} {r.voice}: {r.text}\n        heard: {r.whisper_heard}")

    print("\nWorst intelligibility (WER categories):")
    w = ok[ok.category.isin(WER_CATEGORIES)].sort_values("wer", ascending=False).head(8)
    for _, r in w.iterrows():
        print(f"  {r.wer:.2f} {r.id} {r.voice}: {r.text}\n        heard: {r.whisper_heard}")

    flagged = ok[ok["flags"].fillna("") != ""]
    print(f"\nReliability flags: {len(flagged)} clips")
    for _, r in flagged.iterrows():
        print(f"  {r.id} {r.voice}: {r['flags']}")

    if not args.limit:
        judge_wer, human_rate, human_mos = human_baseline(whisper)
        print(f"\nHuman baseline (30 real Indian English clips from Part 1):")
        print(f"  Whisper WER on them = {judge_wer:.3f}  (the judge's own error rate)")
        print(f"  speaking rate = {human_rate:.2f} words/s  (Timbre median {ok.words_per_sec.median():.2f})")
        print(f"  DNSMOS = {human_mos:.2f}  (Timbre mean {ok.mos.mean():.2f})")

        # one call to see if the newer model works at all (Prisma v3.0 did not)
        meta = synthesize(sentences[0]["text"], AUDIO_DIR / "timbre-v3.0_Sohini" / "p01.wav",
                          voice="Sohini", language="en-IN", model="timbre-v3.0")
        print(f"\ntimbre-v3.0 check: {'works' if meta.get('ok') else meta}")

    # chart
    by_cat = ok.groupby("category").mos.mean().reindex([c for c in ORDER if c in set(ok.category)])
    plt.figure(figsize=(10, 4))
    plt.bar(by_cat.index, by_cat.values)
    plt.ylabel("DNSMOS P.808 (1-5)")
    plt.ylim(1, 5)
    plt.title(f"{MODEL} predicted naturalness by category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/tts_mos_by_category.png", dpi=120)


if __name__ == "__main__":
    main()

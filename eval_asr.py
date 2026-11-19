"""
Part 1: evaluate Prisma (ASR).

    python eval_asr.py              run everything
    python eval_asr.py --limit 3    quick test on 3 clips
    python eval_asr.py --model prisma-v3.0

Reads data/manifest.csv plus the clips recorded with record.py,
sends each to Prisma (cached), and writes:
    results/asr_results_<model>.csv     one row per clip
    results/asr_by_category_<model>.csv the summary table
    results/asr_by_category_<model>.png bar chart
"""
import argparse
import csv
import difflib
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gnani_api import transcribe
from metrics import cer, error_counts, normalize, wer

ORDER = [
    "clean_us", "accent_irish", "accent_indian", "short", "long",
    "numbers_spoken", "proper_nouns", "noisy_20db", "noisy_10db", "noisy_0db",
    "phone_8k", "own_numbers", "own_hinglish", "own_banking_names",
]


def load_clips():
    clips = pd.read_csv("data/manifest.csv").to_dict("records")
    with open("recording_script.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wav = Path(f"data/samples/own/{row['id']}.wav")
            if wav.exists():
                clips.append({
                    "id": f"{row['category']}_{row['id']}",
                    "category": row["category"],
                    "source": "own_recording",
                    "reference": row["text"],
                    "expected_formatted": row["expected_formatted"],
                    "wav_path": wav.as_posix(),
                })
    return clips


def numbers_ok(expected, transcript):
    # e.g. expected "250000", Prisma transcript "the loan amount is ₹2,50,000"
    if not isinstance(expected, str) or not expected:
        return None
    text = transcript.lower().replace(",", "").replace(" ", "")
    return all(part in text for part in expected.split(";"))


def substitutions(ref, hyp):
    # which reference words were heard as something else (for error analysis)
    r, h = normalize(ref).split(), normalize(hyp).split()
    pairs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r, h).get_opcodes():
        if op == "replace":
            pairs.append((" ".join(r[i1:i2]), " ".join(h[j1:j2])))
        elif op == "delete":
            pairs.append((" ".join(r[i1:i2]), "<missing>"))
    return pairs


def summarize(df):
    rows = []
    for cat in [c for c in ORDER if c in set(df.category)]:
        d = df[df.category == cat]
        errors = d["sub"].sum() + d["del"].sum() + d["ins"].sum()
        rows.append({
            "category": cat,
            "clips": len(d),
            "words": d.ref_words.sum(),
            "WER": round(errors / d.ref_words.sum(), 3),
            "CER": round(d.char_errors.sum() / d.ref_chars.sum(), 3),
            "sub": d["sub"].sum(), "del": d["del"].sum(), "ins": d["ins"].sum(),
            "perfect_clips": int((d.wer == 0).sum()),
            "empty_transcripts": int((d.hypothesis.fillna("") == "").sum()),
            "avg_latency_s": round(d.latency.mean(), 2),
        })
    errors = df["sub"].sum() + df["del"].sum() + df["ins"].sum()
    rows.append({
        "category": "ALL", "clips": len(df), "words": df.ref_words.sum(),
        "WER": round(errors / df.ref_words.sum(), 3),
        "CER": round(df.char_errors.sum() / df.ref_chars.sum(), 3),
        "sub": df["sub"].sum(), "del": df["del"].sum(), "ins": df["ins"].sum(),
        "perfect_clips": int((df.wer == 0).sum()),
        "empty_transcripts": int((df.hypothesis.fillna("") == "").sum()),
        "avg_latency_s": round(df.latency.mean(), 2),
    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=None, help="default is prisma-v2.5")
    args = parser.parse_args()
    model_name = args.model or "prisma-v2.5"

    clips = load_clips()
    if args.limit:
        clips = clips[: args.limit]

    results = []
    for i, clip in enumerate(clips, 1):
        res = transcribe(clip["wav_path"], clip["id"], model=args.model)
        if not res.get("success"):
            print(f"[{i}/{len(clips)}] {clip['id']} FAILED: {res.get('error')}")
            continue

        # 'literal' is spoken form ("twenty three"), same style as our references
        literal = res.get("output", {}).get("literal") or res["transcript"]
        counts = error_counts(clip["reference"], literal)
        results.append({
            "id": clip["id"],
            "category": clip["category"],
            "reference": clip["reference"],
            "hypothesis": literal,
            "formatted": res["transcript"],
            "wer": round(wer(clip["reference"], literal), 3),
            "cer": round(cer(clip["reference"], literal), 3),
            **counts,
            "numbers_ok": numbers_ok(clip.get("expected_formatted"), res["transcript"]),
            "latency": res.get("client_latency"),
        })
        print(f"[{i}/{len(clips)}] {clip['id']}  WER={results[-1]['wer']:.2f}")

    df = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/asr_results_{model_name}.csv", index=False)

    summary = summarize(df)
    summary.to_csv(f"results/asr_by_category_{model_name}.csv", index=False)
    print("\n" + summary.to_string(index=False))

    nums = df.dropna(subset=["numbers_ok"])
    if len(nums):
        print(f"\nNumber formatting correct: {int(nums.numbers_ok.sum())}/{len(nums)}")

    print("\nWorst 10 clips:")
    for _, r in df.sort_values("wer", ascending=False).head(10).iterrows():
        print(f"  {r.wer:.2f}  {r.id}\n        ref: {r.reference}\n        hyp: {r.hypothesis}")

    pairs = Counter()
    for _, r in df.iterrows():
        pairs.update(substitutions(r.reference, r.hypothesis))
    print("\nMost common errors (reference -> heard):")
    for (a, b), n in pairs.most_common(15):
        print(f"  {n}x  {a} -> {b}")

    chart = summary[summary.category != "ALL"]
    plt.figure(figsize=(10, 4))
    plt.bar(chart.category, chart.WER * 100)
    plt.ylabel("WER (%)")
    plt.title(f"{model_name} WER by category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/asr_by_category_{model_name}.png", dpi=120)


if __name__ == "__main__":
    main()

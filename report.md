# Evaluating Gnani Prisma (ASR) and Timbre (TTS)

Subin Krishna K T, AI Solutions Engineer take-home

**Approach.** This is the check I would run before recommending Prisma and Timbre to a typical Gnani customer: an Indian bank, lender or support centre running voice bots on phone calls. All clips, API responses and scores are in the repo, so everything can be re-scored without API credits.

## Part 1: Prisma (ASR)

**Test set.** 170 clips in 14 categories, from Common Voice 26 (South Asian and Irish English) and LibriSpeech. The noisy and phone-line clips reuse the same 10 Indian clips, so any change in WER comes from the noise or channel alone. I recorded 25 clips myself for what public data lacks: Indian numbers, Hinglish and banking terms.

**Method.** WER and CER are implemented by hand (`metrics.py`): a Levenshtein alignment counts substitutions, deletions and insertions, and WER = (S + D + I) / reference words. The code is checked against `jiwer` in the tests. Category scores are total errors over total reference words, after lowercasing, removing punctuation and expanding contractions on both sides. WER is scored on Prisma's spoken-form output (`literal`), which matches the reference style.

**Results** (Prisma v2.5, `en-IN`)

| Category | Clips | WER | CER | Notes |
|---|---|---|---|---|
| Indian English | 30 | **1.6%** | 1.2% | 25 clips perfect |
| Irish English | 20 | 2.6% | 0.9% | |
| Short (1–3 words) / spoken numbers | 2×10 | 0.0% / 4.7% | 0.0% / 1.1% | |
| Names and places | 10 | 11.4% | 3.9% | Tevez → "stevens", Qureish → "kerish" |
| Phone 8 kHz / chatter 20 dB / 10 dB | 3×10 | 0.9% / 0.9% / 1.8% | 0.6% / 0.6% / 1.0% | Hardly affected |
| Chatter at 0 dB SNR | 10 | **43.8%** | 31.0% | |
| Indian numbers (own) | 10 | **0.0%** | 0.0% | Formatted right 10/10 ("₹2,50,000") |
| Banking terms and names (own) | 7 | 1.7% | 0.7% | KYC, CIBIL, UPI, NACH correct |
| Hinglish (own) | 8 | 15.9% | 4.7% | 5 of 11 errors are spelling only |
| Clean US (LibriSpeech) | 19 | 16.7% | 15.8% | **3 empty**; 3.0% WER on the rest |
| Long (17–27 s) | 5 | **49.3%** | 47.6% | **Whole sections silently removed** |
| All | 169 | 14.2% | 11.7% | 9.6% without the empty transcripts |

Speed: median server processing time 0.14 s; about 0.8 s per request from my laptop.

**Where Prisma breaks down**

1. **Silent failures.** Prisma returned `success: true` with an empty or partial transcript, and the documented `truncated` flag was never set. Long clips lost whole sentences from the middle, and 3 of the 19 scored LibriSpeech clips came back empty, also when re-sent as FLAC and louder. No Indian or Irish clip failed this way. A voice bot would act on half a sentence with no error to catch.
2. **0 dB noise.** When background speech is as loud as the speaker, Prisma follows the wrong voice or makes up words.
3. **Rare names** are spelled phonetically. v3.0's `keyterms` parameter could help, but v3.0 returned HTTP 500 on every request.
4. **Integration.** The firewall returned a 403 HTML page for one valid WAV every time (accepted as FLAC). Only one request can run at a time, and audio over 30 s must be split.

## Part 2: Timbre (TTS)

**What "good" means.** Five criteria for a bank or support call, each with an automatic, repeatable measurement:

| Criterion | Measurement |
|---|---|
| Understood correctly | **Whisper small** (local, not Gnani) transcribes it; WER/CER with the Part 1 code |
| Reads Indian formats | Pass if Whisper heard an accepted reading ("two lakh fifty thousand" or "250000") |
| Sounds natural | Predicted listener scores (1–5): **UTMOS** (trained on TTS ratings), **NISQA**, **DNSMOS** |
| Fast enough for a live call | Time to first audio chunk, and **RTF = generation time / audio length** |
| Reliable | No empty audio, clipping or long silences; speaking rate within 1.5–4.5 words/s |

**Test set.** 55 sentences in 11 categories, spoken by Kaveri (F) and Pranav (M), with Hinglish in `hi-en` by Poorvi and Kaveri: 110 clips. Number sentences appear both as raw text ("₹2,50,000") and in words, because Gnani's docs say to write numbers as words. Human reference: the 30 Indian English clips from Part 1 give Whisper WER 5.1% and 2.8 words/s.

**Results** (timbre-v2.5)

| Category | Kaveri | Pranav | Notes |
|---|---|---|---|
| Plain / tricky / long text (WER) | 0 / 0 / 4% | 0 / 0 / 4% | Whisper can't tell read/read apart |
| Numbers, raw text / in words | 8/8, 8/8 | 8/8, 8/8 | ₹2,50,000 → "two lakh fifty thousand" |
| Dates, times, phone, PIN, OTP | 9/9 | 9/9 | "A/C" read as "account" |
| Acronyms (KYC, EMI, UPI, NACH) | 6/6 | 5/6 | CIBIL heard as "Sibyl" |
| Email / URL | 0/2 | 0/2 | Whisper wrote "gyani.ai" *(check by ear: TO CONFIRM)* |
| Indian names (WER) | 5.3% | 15.8% | CER 2.1% for both; spelling only |
| Hinglish (CER) | 7.9% | 5.7% (Poorvi) | Mostly spelling |
| MOS: UTMOS / NISQA / DNSMOS | 4.14 / 4.45 / 3.28 | **4.28 / 4.78 / 3.70** | Poorvi 4.21 / 4.85 / 3.77; human clips 2.50 / 2.85 / 3.49 |

Speed: median time to first audio is **0.35 s**, but p90 is 1.29 s (worst 2.6 s). Median RTF is **0.23**.

**Performs well.** Timbre reads Indian numbers, currency, dates and banking acronyms correctly from raw text, so customers don't need their own normalisation step. Intelligibility matches a real speaker. All three scorers put Kaveri below Pranav and Poorvi. Timbre beats the human clips on UTMOS and NISQA only because those are crowd-sourced home recordings. Clip by clip the scorers agree poorly (rank correlation 0.17–0.37), so only averages mean much. **It struggles** in two places: it speaks fast (median 3.9 words/s against 2.8 for the human clips), and about 1 in 9 requests waited over a second for the first audio, which a caller hears as dead air.

## Recommendation

**Timbre: ready to recommend** for Indian English and Hinglish voice bots. Default to Pranav or Poorvi at `speed=0.9`, with a filler sound to cover the occasional 1–2 s start delay.

**Prisma: recommend for short Indian English turns** (IVR answers, phone audio, normal noise, numbers, banking terms, simple Hinglish), where it is excellent. It is not ready for long recordings or very noisy rooms. Before deployment I would require a client-side guard (re-prompt on an empty or suspiciously short transcript), audio in segments under 15 s, and follow-ups with Gnani on the silent drops, the `truncated` flag, v3.0 and the firewall 403.

**Limitations.** Samples are small (5–30 clips per category), so differences of a few points are within noise. My own recordings are one speaker. The MOS scores are predictions, not listener ratings; a small listener panel is the next step. Only the v2.5 models were fully tested.

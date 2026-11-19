# Evaluating Gnani Prisma (ASR) and Timbre (TTS)

Subin Krishna K T, AI Solutions Engineer take-home

## Approach

I treated this as the check I would run before recommending Prisma and Timbre to a typical Gnani customer: an Indian bank, lender or support centre running voice bots and phone calls. So the test set is built around Indian English, phone audio, background noise, numbers, Hinglish and banking terms. Clean American English is included only as a reference point.

Everything runs from the repo: picked audio clips, saved API responses, scripts, and results. Scores can be recomputed without spending API credits.

## Part 1: Prisma (ASR)

**Test set (145 clips; 25 of my own recordings to be added).** Clips were hand-picked for clean, correct transcripts, one per speaker where possible.

| Category | Source | Clips |
|---|---|---|
| Clean US English (reference) | LibriSpeech test-clean | 20 |
| Indian English | Common Voice 26, South Asian accent | 30 |
| Irish English | Common Voice 26, Irish accent | 20 |
| Short (1–3 words, e.g. "yes", "no") | Common Voice South Asian | 10 |
| Spoken numbers | Common Voice South Asian | 10 |
| Names and places | Common Voice South Asian | 10 |
| Long (17–27 s, the API limit is 30 s) | LibriSpeech, consecutive sentences joined | 5 |
| Background chatter at 20 / 10 / 0 dB SNR | 10 Indian clips + chatter from 6 other speakers | 30 |
| Phone line (8 kHz, mu-law) | the same 10 Indian clips | 10 |

The noisy and phone clips reuse the same 10 recordings, so any change in WER comes from the noise or the channel alone.

**Method.** WER and CER are implemented by hand (`metrics.py`). A word-level Levenshtein alignment counts substitutions, deletions and insertions, and WER = (S + D + I) / reference words. The code is checked against `jiwer` in the tests.

- **Corpus-level averages.** Category scores are total errors divided by total reference words, so a 3-word clip doesn't count as much as a 20-word one.
- **Text cleanup before scoring.** Both sides are lowercased and stripped of punctuation, and contractions are expanded ("she's" = "she is").
- **Which Prisma field is scored.** Prisma returns two transcripts. `literal` is in spoken words ("thirteen minutes") and `transcript` is formatted ("13 minutes"). WER is scored on `literal`, which matches the reference style.

**Results** (Prisma v2.5, `en-IN`)

| Category | WER | CER | Notes |
|---|---|---|---|
| Indian English | **1.6%** | 1.2% | 25 of 30 clips perfect |
| Irish English | 2.6% | 0.9% | |
| Short utterances | 0.0% | 0.0% | |
| Spoken numbers | 4.7% | 1.1% | Formatted output correct, e.g. "8:30 am to 5:30 pm", "12,469 dollars" |
| Phone line, 8 kHz | 0.9% | 0.6% | Same as the clean version |
| Chatter 20 dB / 10 dB | 0.9% / 1.8% | | Hardly affected |
| Chatter 0 dB | **43.8%** | 31.0% | Breaks down when the background is as loud as the speaker |
| Names and places | 11.4% | 3.9% | Tevez → "stevens", Qureish → "kerish" |
| Clean US English | 16.7% | 15.8% | **3 of 19 clips returned an empty transcript**; 3.0% WER on the rest |
| Long (17–27 s) | **49.3%** | 47.6% | **Whole sections silently removed** |
| All | 15.3% | 13.0% | 10.1% if empty transcripts are excluded |

Speed: median server processing time 0.14 s; 0.7 s per request from my laptop in Kerala, including upload.

**Where Prisma breaks down**

1. **Silent failures.** Prisma returned `success: true` with an empty or partial transcript, and the documented `truncated` flag was never set.
   - **Long clips:** whole sentences went missing from the middle, and the output had a double space where the text had been ("a few minutes after␣␣incoherent and unaccountable"). That suggests a filter after recognition that drops low-confidence segments.
   - **Short clean clips:** 3 of the 20 LibriSpeech clips came back completely empty. Sending the original FLAC files and a louder copy gave the same result, so it isn't my conversion or the volume. None of the 70 Indian or Irish clips failed this way.

   A voice bot built on this would act on half a sentence with no error to catch.
2. **0 dB noise.** At 10 dB SNR Prisma is still fine. When the background speech is as loud as the speaker, it switches to the wrong voice or makes up words.
3. **Unfamiliar names** are spelled phonetically. Prisma v3.0 has a `keyterms` parameter that should help with this, but v3.0 returned HTTP 500 on every request during testing, so I couldn't test it.
4. **Integration issues:**
   - The Cloudflare firewall returned a **403 HTML page** for one valid WAV file every time. The same audio sent as FLAC was accepted.
   - Only one request can run at a time (429 otherwise).
   - Clips longer than 30 s need to be split by the client.
5. **Spelling, not errors.** Some leftover "errors" are spelling differences: neighbors/neighbours (Prisma uses British spelling, right for Indian English) and backslashes/back slashes. I counted these as errors rather than tuning the normalization to hide them.

## Part 2: Timbre (TTS)

**What "good" means.** For a bank or support call, a voice is good if it is:
1. **understood correctly**
2. **reads Indian formats correctly** (₹ amounts, lakh/crore, dates, phone numbers, acronyms)
3. **sounds natural**
4. **fast enough for a live call**
5. **reliable** (no empty, cut-off or clipped audio)

Each of these becomes an automatic, repeatable measurement:

| Criterion | Measurement |
|---|---|
| Understood | Transcribe Timbre's audio with **Whisper small**, a local, independent ASR, so Gnani is not grading Gnani. Score WER/CER against what should be said, using the same metric code as Part 1. |
| Indian formats | Each sentence lists correct readings ("two lakh fifty thousand" or "250000"). Pass if Whisper heard one of them. |
| Natural | **DNSMOS P.808**, Microsoft's predicted listener score (1–5), run locally. |
| Speed | Time to first audio chunk (streaming endpoint), total time, and **RTF = generation time / audio length**. |
| Reliable | Empty audio, clipping, long silences, and speaking rate outside 1.5–4.5 words/s. |

**Human reference.** The same checks run on the 30 real Indian English recordings from Part 1:
- Whisper's own WER on them is **5.1%**, so a TTS WER below that means "as clear as a real speaker".
- Their DNSMOS is 3.49.
- Their speaking rate is 2.8 words/s.

**Test set.** 55 sentences in 11 categories, voices **Kaveri (F)** and **Pranav (M)**. The Hinglish sentences use `hi-en` with **Poorvi** and Kaveri. That makes 110 clips.

The 8 number sentences appear twice: as raw text ("₹2,50,000", "8.5%", "1800-209-4555") and written out as words. Gnani's docs say to write numbers as words, so the pair shows whether customers need their own normalization step.

**Results** (timbre-v2.5)

| Category | Kaveri | Pranav | Notes |
|---|---|---|---|
| Plain sentences | WER 0% | 0% | |
| Numbers, raw text | 8/8 | 8/8 | ₹2,50,000 → "two lakh fifty thousand", ₹1.2 crore, Rs. 499, 15%, 8.5% |
| Numbers, written as words | 8/8 | 8/8 | No difference from raw text |
| Dates and times | 5/5 | 5/5 | 15/08/2025, 31-12-2026, 10:30 AM, 2 PM |
| Phone numbers / PIN / OTP / account | 4/4 | 4/4 | "A/C" read as "account" |
| Acronyms (KYC, EMI, UPI, NACH, PAN, Dr., Rd.) | 6/6 | 5/6 | Miss: CIBIL heard as "Sibyl" |
| Email / URL | 0/2 | 0/2 | See note below |
| Indian names | WER 5.3%, CER 2.1% | WER 15.8%, CER 2.1% | Only spelling differences ("Venkat Araman", "Subramanyam", "call back") |
| Hinglish | CER 7.9% (Kaveri) | 5.7% (Poorvi) | Mostly spelling ("shubho", "kredit", "5 to 7") |
| Tricky words (read/read, wind/wind) | WER 0% | 0% | Whisper can't hear which pronunciation was used; checked by ear |
| Long paragraphs (~100 words) | 4.0% | 4.0% | |
| DNSMOS (mean) | 3.28 | **3.70** | Poorvi 3.77; human recordings 3.49 |

Speed:
- **Time to first audio:** median **0.35 s**, but p90 is 1.29 s, and 12 of 110 calls took over 1 s (worst 2.6 s).
- **RTF:** median **0.23** (about 4× faster than real time). One short clip reached 1.17 because of a slow start.

The email/URL "misses" need care. Whisper wrote "gyani.ai" for www.gnani.ai, and "Gnani" is pronounced "gyani". It wrote "hdfsbank.com", and "Sibyl" for CIBIL, which is how CIBIL is pronounced. These look like the scorer's spelling rather than Timbre errors. *(Listening check: TO CONFIRM.)*

**Where Timbre performs well and where it struggles**

- **Strong:**
  - It reads Indian numbers, dates, currency and banking acronyms correctly straight from raw text. Customers don't need to write numbers as words first, despite what the docs suggest.
  - Intelligibility matches a real speaker.
  - The male voice Pranav scores higher than Kaveri for naturalness in every category.
  - timbre-v3.0 also worked in a single test call.
- **Weak:**
  - **It speaks fast:** median 3.9 words/s vs. 2.8 for the human recordings, so 21 clips were flagged. For collections calls or older callers I would start at `speed=0.9`.
  - **Time to first audio is uneven:** about 1 in 9 requests waited more than a second before any audio, which a caller hears as dead air.

## Recommendation

**Timbre: ready to recommend** for Indian English and Hinglish voice bots. Suggest Pranav or Poorvi as default voices and a slightly slower speed. Customers should play a filler sound or start audio early to cover the occasional 1–2 s delay.

**Prisma: recommend for short Indian English turns** (IVR answers, phone audio, normal office noise), where it is excellent. Before any deployment, I would require:
- **Client-side guards:** treat an empty transcript, or one with far fewer words than expected for the audio length, as a failure and re-prompt.
- **Audio in short segments (under 15 s)**, never long clips.
- **Follow-ups with Gnani:** the silent drops, the missing `truncated` flag, the v3.0 errors and the firewall 403.

Prisma is not yet ready for long recordings or very noisy environments.

## Limitations

- **Small samples:** 10–30 clips per category, so differences under a few points are within noise.
- **One run per request,** so latency includes my home network.
- **Speaker coverage:** Common Voice labels the accent only as "South Asia", and my own recordings are a single speaker.
- **Automatic scores are approximate.** DNSMOS was built to score audio quality, so it is only a rough stand-in for naturalness. Whisper misspells names and Hinglish, so CER is the fairer score there. Homograph pronunciation was checked by ear.
- **Only the v2.5 models were fully tested.**
- **Whisper scores depend on library versions.** The first run used numpy 1.23 and changed a few name spellings compared with the locked environment (numpy 2.4). All numbers here come from the locked `uv.lock` environment.

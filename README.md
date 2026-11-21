# Gnani.ai: Prisma (ASR) and Timbre (TTS) evaluation

Take-home assignment for the AI Solutions Engineer role. The write-up is in [report.md](report.md).

## Setup

Uses [uv](https://docs.astral.sh/uv/) and Python 3.11. Exact versions are locked in `uv.lock`.

```
uv sync                         # creates .venv with everything from pyproject.toml
copy .env.example .env          # then put your Gnani API key in .env
```

The first TTS run downloads the Whisper `small` model (~460 MB), the UTMOS model (~390 MB, via `torch.hub` from `tarepan/SpeechMOS` v1.2.0) and the NISQA weights (to `~/.torchmetrics/NISQA`). Whisper uses the GPU if CUDA 12 is available and otherwise falls back to the CPU.

## Run

```
uv run pytest tests             # WER/CER and matcher tests
uv run python eval_asr.py       # Part 1: Prisma
uv run python eval_tts.py       # Part 2: Timbre
```

All API responses are already saved in `results/raw/` and `results/tts_audio/`, so both scripts re-score everything **without calling the API**. Only new or missing clips are sent. Use `--limit 3` for a quick test.

Optional:

```
uv run python prepare_data.py   # rebuild data/samples from the original datasets (paths in .env)
uv run python record.py         # record the sentences in recording_script.csv with your mic
```

## Files

| File | What it does |
|---|---|
| `metrics.py` | WER / CER (own Levenshtein implementation), text normalization, number/acronym matcher |
| `gnani_api.py` | Prisma and Timbre calls with retry and on-disk caching |
| `prepare_data.py` | picks the test clips, converts to 16 kHz, adds background chatter and phone-line versions |
| `eval_asr.py` | Part 1: runs Prisma, writes per-clip and per-category results and a chart |
| `eval_tts.py` | Part 2: runs Timbre, scores with Whisper, three predicted-MOS models (UTMOS, NISQA, DNSMOS), latency and audio checks |
| `tts_sentences.csv` | the 55 TTS test sentences, with expected readings |
| `recording_script.csv`, `record.py` | own recordings: Indian numbers, Hinglish, banking terms |
| `data/samples/`, `data/manifest.csv` | the ASR test clips and their reference transcripts |
| `results/` | all results, logs, charts, Timbre audio and raw API responses |

## Data and licenses

- **LibriSpeech** test-clean: CC BY 4.0. Panayotov et al., 2015, openslr.org/12
- **Mozilla Common Voice 26** (South Asian English, Irish English): CC0. Speaker IDs are replaced with short labels.
- Background chatter is made by mixing other Common Voice South Asian speakers.
- `data/samples/own/`: 25 clips I recorded myself with `record.py` (Indian numbers, Hinglish, banking terms).

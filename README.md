# AURIS — Smart Subtitle Glasses: Speaker Identification

Final Year Project (Mechatronics Engineering, NUST) — real-time, speaker-labelled captioning
on a wearable display for hearing-impaired children.

This repo covers **Sprint 1**: can pre-trained speaker embeddings enrol a few speakers from
audio alone, re-identify them, and reject unknown voices, before any hardware integration?

## Status (end of September 2026)

We compared three pre-trained speaker-embedding models on our own recordings
(2 enrolled speakers, 10 known-speaker test clips, 4 unknown-speaker test clips).

| Model | Threshold | Known clips correct | Unknowns rejected | Separation (d′) |
|---|---|---|---|---|
| WeSpeaker ResNet34-LM (pyannote) | **0.4986** (calibrated) | 10/10 | 4/4 | 4.49 |
| TitaNet-Large (NVIDIA NeMo) | **0.4854** (calibrated) | 10/10 | 4/4 | 4.72 |
| WavLM Base+ SV, default threshold | 0.86 (fixed default) | 10/10 | **0/4** | 3.22 |
| WavLM Base+ SV, recalibrated | 0.927 (calibrated offline) | 10/10 | 4/4 | 3.22 |

*Results use the averaged enrollment profile. d′ measures how far apart genuine-match and
impostor scores are, independent of each model's score range; higher is better.*

**Key findings**

- **WeSpeaker and TitaNet are effectively tied** on our data. TitaNet separates the two
  enrolled speakers slightly better; WeSpeaker keeps unknown speakers further below its threshold.
- **WavLM (`wavlm-base-plus-sv`) is not suitable.** Its scores bunch between ~0.8 and 0.98,
  so at its default threshold it accepted every unknown speaker. After recalibration only
  0.03 separates its worst genuine match from its best impostor (vs 0.15 for the other two).
- **Short enrollment is the main failure mode.** With 5–10 s enrollment, both WeSpeaker and
  TitaNet falsely rejected some genuine clips; with ≥15 s, both were correct. All errors came
  from one speaker, in both models, which points to recording quality rather than the model.
- **Proposed model: WeSpeaker ResNet34-LM** — equal accuracy, smallest (6.6M parameters),
  ONNX builds exist for ARM, and it is the embedding model inside pyannote's diarization
  pipeline. TitaNet is kept as the comparison model.

Full comparison, literature numbers, and reasoning: [`docs/sprint1_model_comparison.md`](docs/sprint1_model_comparison.md).
Summary table: [`results/model_comparison.csv`](results/model_comparison.csv).

## Correction to the earlier version of this README

The first commit reported a threshold of **0.40** and 100% results. That threshold was tuned
on the same clips it was evaluated on, so those numbers were optimistic. The pipeline now
calibrates the threshold on odd-numbered clips and reports threshold-dependent metrics on
even-numbered clips. The earlier notebook, results, and one-pager are kept in
`notebooks/archive/`, `results/archive_v1/`, and `reports/archive/` for reference only.

## Repo structure

```
AURIS-/
├── src/
│   ├── speaker_id_pipeline.py     # enrol → score → calibrate threshold → CSVs + Google Sheets
│   └── slice_segments.py          # cut test clips into 0.5–5 s windows (window-length experiment)
├── notebooks/
│   ├── 01_wespeaker_wavlm.ipynb   # Colab: WeSpeaker + WavLM runs, model comparison
│   ├── 02_titanet.ipynb           # Colab: TitaNet run (separate runtime; NeMo clashes with pyannote)
│   └── archive/                   # first-pass notebook (superseded)
├── results/
│   ├── model_comparison.csv       # one row per model: threshold, accuracy, separation
│   ├── runs.csv                   # log of every pipeline run
│   ├── wespeaker/                 # per-clip scores, enrollment-length comparison, failures, unknowns
│   ├── titanet/                   # same files for TitaNet
│   ├── wavlm/                     # WavLM scores (separate script, fixed threshold)
│   └── archive_v1/                # first-pass results (superseded)
├── docs/
│   └── sprint1_model_comparison.md
└── reports/archive/               # first-pass one-pager and workbook (superseded)
```

## Reproducing this

1. Put audio in Google Drive under `MyDrive/speaker_id_fyp/`:
   - `audio/enrollment/` — `Name_Ns.wav`, e.g. `Ayesha_15s.wav`; extra takes as `Ayesha_15s_b.wav`
   - `audio/tests/` — `Name#.wav`, e.g. `Ayesha3.wav`; unknown speakers as `U#.wav`
2. Open `notebooks/01_wespeaker_wavlm.ipynb` in Colab and run cells top to bottom.
   It writes `speaker_id_pipeline.py` into Drive itself. You'll need a Hugging Face token
   (read access) for the pyannote model.
3. Open `notebooks/02_titanet.ipynb` in a **separate** Colab runtime for TitaNet.
4. Each model's results go to its own folder on Drive and its own tabs in the Google Sheet;
   `sp.compare_models(ROOT)` lines them up.

Design rules the pipeline follows: raw scores are logged for every clip against every
enrolled speaker (unknowns included); one threshold per run, stored with the results; the
threshold is calibrated on dev clips and evaluated on eval clips; every rate is reported
with a Wilson 95% interval.

**Privacy:** raw audio and voice embeddings (`enrolled_speakers.json`,
`test_embeddings_cache.json`) are biometric data tied to named people and are not committed
(see `.gitignore`).

## Limitations

- Very small sample (10 known + 4 unknown clips), so accuracy differences between
  WeSpeaker and TitaNet are not statistically meaningful yet.
- The runs did not use identical enrollment data: WeSpeaker used 8 enrollment files,
  TitaNet 15 (extra takes for one speaker), and WavLM was run with a separate script.
- The odd/even dev–eval split comes from a single recording session, so it is optimistic.
- Clean audio only; no noise, distance, Urdu, or children's speech tested yet.

## Next steps

1. Re-record enrollment: quieter room, ≥15 s clips, two sessions on different days.
2. Record more unknown speakers.
3. Re-run all models on identical data (WavLM through the pipeline).
4. Window-length experiment with `slice_segments.py`.
5. Measure latency on Raspberry Pi 5 (ONNX / sherpa-onnx).

## Team

Ayesha & Sabeeh — Mechatronics Engineering, NUST

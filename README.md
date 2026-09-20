# Smart Subtitle Glasses — Speaker Identification

Final Year Project (Mechatronics Engineering) — real-time, speaker-labeled captioning
on a wearable display for hearing-impaired children.

This repo currently covers **Sprint 1**: validating whether speaker embeddings can
reliably enroll and re-identify two speakers from audio alone, before any hardware
integration.

## Status (September 2026)

| Question | Answer |
|---|---|
| Can the embedding model tell two enrolled speakers apart? | **Yes — 100%** of test clips ranked the correct speaker as closest match, at every enrollment length tested (5s–30s) |
| Can it reject an unenrolled stranger? | **Yes — 100%** rejection rate across all 7 unknown-speaker test clips |
| What acceptance threshold should the system use? | **0.40** cosine similarity (calibrated empirically — see `reports/`); the initial default of 0.70 was too conservative and caused false rejections |
| Does enrollment length matter? | Ranking correctness was unaffected by enrollment length (even 5s worked); longer enrollment mainly increases confidence margin |

Full breakdown, methodology, and caveats are in
[`reports/Speaker_ID_Results_OnePager.pdf`](reports/Speaker_ID_Results_OnePager.pdf).

## Repo structure

```
smart-subtitle-glasses/
├── notebooks/
│   └── speaker_id_experiment.ipynb   # enrollment, saving, identification, diarization matching
├── results/
│   ├── identification_results.csv        # raw per-clip results across all thresholds/durations
│   ├── enrollment_length_comparison.csv  # accuracy by enrollment duration
│   └── failure_log.csv                   # every incorrect prediction
└── reports/
    ├── Speaker_ID_Results_OnePager.pdf   # one-page visual summary for supervisor review
    └── Speaker_ID_Results.xlsx           # interactive workbook — adjustable threshold, live charts
```

## Reproducing this

1. Open `notebooks/speaker_id_experiment.ipynb` in Google Colab
2. Run cells top to bottom; you'll be prompted for a HuggingFace token
   ([get one here](https://huggingface.co/settings/tokens), read access is enough)
3. Provide your own enrollment audio (`Name_Ns.wav` format, e.g. `Ayesha_5s.wav`)
   and test audio (`Name#.wav`, e.g. `Ayesha1.wav`; unknown speakers as `U#.wav`)

**Note:** raw `.wav` audio files are intentionally not committed to this repo (see
`.gitignore`) — audio was recorded locally by each team member and isn't shared here.

## Team

Ayesha & Sabeeh — Mechatronics Engineering, NUST Islamabad

## Scope and caveats

This pilot used 2 enrolled speakers, 10 known-speaker test clips, and 7
unknown-speaker clips, all clean audio. The calibrated threshold (0.40) should be
re-validated as more speakers, background noise, distance, and Urdu-language
samples are added in later testing rounds.

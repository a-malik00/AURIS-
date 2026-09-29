# Sprint 1 — Speaker-embedding model comparison

## 1. Setup

- **Data:** 2 enrolled speakers (Ayesha, Sabeeh); enrollment clips of 5/10/15/30 s (Ayesha
  also has 20/40/51 s clips and second takes); 10 known test clips (5 per speaker);
  4 unknown-speaker test clips (U1–U4).
- **Scoring:** cosine similarity between the test clip's embedding and each speaker's
  profile (mean of their enrollment embeddings, L2-normalised).
- **Decision:** best score ≥ threshold → that speaker; otherwise UNKNOWN.
- **Threshold calibration:** odd-numbered clips (dev) set the threshold at the midpoint
  between the lowest genuine score and the highest impostor score (or the equal-error point
  if they overlap); even-numbered clips (eval) are used for threshold-dependent metrics.

## 2. Results on our data (averaged enrollment profile)

| | WeSpeaker ResNet34-LM | TitaNet-Large | WavLM @ 0.86 | WavLM @ 0.927 |
|---|---|---|---|---|
| Threshold | 0.4986 | 0.4854 | 0.86 | 0.927 |
| Known clips correct | 10/10 | 10/10 | 10/10 | 10/10 |
| Unknowns rejected | 4/4 | 4/4 | 0/4 | 4/4 |
| Lowest genuine score | 0.565 | 0.577 | 0.942 | 0.942 |
| Highest impostor score | 0.418 | 0.429 | 0.912 | 0.912 |
| Gap | 0.147 | 0.147 | 0.030 | 0.030 |
| d′ | 4.49 | 4.72 | 3.22 | 3.22 |
| Ayesha–Sabeeh profile similarity | 0.384 | 0.251 | n/a | n/a |
| Errors with 5/10 s enrollment | 5 false rejects | 4 false rejects | not tested | not tested |

Notes:
- Each model has its own score range, so thresholds and raw gaps are not comparable
  across models; d′ is.
- Highest unknown-speaker score: WeSpeaker 0.33, TitaNet 0.43 (U2) — TitaNet leaves less
  headroom against strangers.
- Every error in both WeSpeaker and TitaNet was a false reject of Sabeeh with 5–10 s
  enrollment. Mean genuine score for Sabeeh (0.62–0.63) is well below Ayesha's (0.79–0.82)
  in both models, suggesting recording quality.

## 3. Published results (literature)

| Model | VoxCeleb1-O EER | Parameters | Source |
|---|---|---|---|
| WeSpeaker ResNet34-LM | 0.797% (0.723% with AS-Norm) | 6.63M | WeSpeaker model card / VoxCeleb v2 recipe |
| TitaNet-Large | 0.68% (model card: 0.66%) | ~23M | Koluguri et al., ICASSP 2022 |
| TitaNet-Small | — | ~6M | Koluguri et al., ICASSP 2022 |
| WavLM Base+ with x-vector head | ~4.1% (SUPERB-style setup) | ~94.7M | Chen et al. 2022; CA-MHFA (arXiv 2409.15234), Table |

Caveat: these come from different papers with different trial lists and scoring setups;
use them to rank the models, not as exact head-to-head numbers. WavLM itself can reach
much lower EER with stronger back-ends and full fine-tuning; the problem is the specific
`wavlm-base-plus-sv` checkpoint, which uses a simple x-vector head trained on VoxCeleb1.

## 4. Practical comparison

| | WeSpeaker ResNet34 | TitaNet | WavLM Base+ SV |
|---|---|---|---|
| Accuracy (published) | high | high | clearly lower |
| Size | 6.6M params (~26 MB) | 23M (Large) / 6M (Small) | ~95M (~380 MB) |
| Fine-tuning tooling | WeSpeaker toolkit recipes, LM fine-tuning, AS-Norm | NeMo scripts; heavy install | Hugging Face; custom loop, GPU needed |
| Edge deployment | ONNX; sherpa-onnx ARM builds | ONNX; sherpa-onnx ARM builds (Small and Large) | no ready edge export found |
| Fit for Raspberry Pi 5 | best | good (TitaNet-Small) | unlikely to run in real time alongside ASR (not measured) |

Fine-tuning on 2–5 enrolled speakers would overfit; improvements should come from
enrollment quality, score normalisation (AS-Norm), and voice-activity detection.

## 5. Decision

- Main model: **WeSpeaker ResNet34-LM**.
- Comparison model: **TitaNet** (TitaNet-Small for on-device tests).
- Drop the `wavlm-base-plus-sv` checkpoint.

No Raspberry Pi latency has been measured yet; that is a planned Sprint 2 result.

## References

- Wang et al. (2023). Wespeaker: A research and production oriented speaker embedding learning toolkit. ICASSP.
- Koluguri, Park & Ginsburg (2022). TitaNet: Neural model for speaker representation with 1D depth-wise separable convolutions and global context. ICASSP.
- Chen et al. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech processing. IEEE JSTSP.
- Bredin (2023). pyannote.audio 2.1 speaker diarization pipeline. INTERSPEECH.
- Poddar, Sahidullah & Saha (2018). Speaker verification with short utterances. IET Biometrics.
- Brown, Cai & DasGupta (2001). Interval estimation for a binomial proportion. Statistical Science.

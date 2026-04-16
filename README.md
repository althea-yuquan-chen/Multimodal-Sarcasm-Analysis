# Multimodal Sarcasm Analysis

Sarcasm detection on the [MUStARD](https://github.com/soujanyaporia/MUStARD) dataset using text, audio, and vision modalities. We experiment with text-only baselines, early fusion (concatenated audio+vision fed into a sequential encoder alongside BERT), and late fusion (separate per-modality models combined at the logit level).

## Data Setup

1. Download `sarcasm.pkl` from [MultiBench Google Drive](https://drive.google.com/drive/folders/1JFcX-NF97zu9ZOZGALGU9kp8dwkP7aJ7) and place in `data/`
2. Download the metadata JSON:
   ```bash
   curl -o data/sarcasm_data.json https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json
   ```

Each sample contains:
- **Text**: utterance string + dialogue context string
- **Audio**: `(50, 81)` feature array
- **Vision**: `(50, 371)` feature array
- **Label**: binary (1 = sarcastic)

## Notebooks

| File | Description |
|------|-------------|
| `baseline.ipynb` | Text-only BERT experiments |
| `early_fusion.ipynb` | BERT + RNN/LSTM over concatenated audio+vision |
| `late_fusion_experiments.ipynb` | Separate per-modality models with fusion at logit level |

## Experiments & Results

### Baseline — Text-only (`baseline.ipynb`)

Three progressively richer text encoders, all using `bert-base-uncased` + MLP, trained for 10 epochs.

| Experiment | Input | Best Test Acc |
|---|---|---|
| Exp 1 | BERT(context) | 72.46% |
| Exp 2 | BERT(context + utterance, concatenated) | 68.12% |
| Exp 3 | concat(BERT(context), BERT(utterance)) — dual encoder | **76.81%** |

The dual encoder (Exp 3) encodes context and utterance separately and concatenates the two `[CLS]` embeddings before the MLP, giving the best text-only result.

---

### Early Fusion (`early_fusion.ipynb`)

Audio and vision features are concatenated into a `(50, 452)` sequence, then encoded by a unidirectional RNN or LSTM. The resulting vector is concatenated with the BERT `[CLS]` embedding and passed through an MLP.

Two training schedules were compared:
- **Joint**: all parameters updated every step
- **Alternating**: seq-encoder-only phases interleaved with BERT-only phases (2 epochs each)

| Seq Encoder | Training | Best Test Acc |
|---|---|---|
| LSTM | Joint | 73.19% |
| LSTM | Alternating | 71.74% |
| RNN | Joint | 71.74% |
| RNN | Alternating | **75.36%** |

Alternating training with RNN gives the best early fusion result, suggesting the lighter RNN benefits more from the staged optimization.

---

### Late Fusion (`late_fusion_experiments.ipynb`)

Each modality (text, audio, vision) has its own model trained independently, and their scalar logits are combined by a `LateFusionCombiner`.

- **Text model**: BERT `[CLS]` → MLP → 1 logit
- **Audio/Vision models**: LayerNorm → Bidirectional RNN/LSTM → MLP → 1 logit
- **Combiners**: `average`, `weighted` (learned softmax weights), `mlp` (small 2-layer net)

#### Exp 1 — Fusion strategy × Training mode

| Fusion Strategy | Training Mode | Test Acc | F1 |
|---|---|---|---|
| average | joint | **77.54%** | **0.7480** |
| average | sequential | 60.87% | 0.6582 |
| weighted | joint | 70.29% | 0.7172 |
| weighted | sequential | 76.09% | 0.7179 |
| mlp | joint | 69.57% | 0.7200 |
| mlp | sequential | 69.57% | 0.6613 |

Sequential training (train unimodal branches, then freeze them and train combiner) underperforms joint training across all strategies.

#### Exp 2 — RNN vs LSTM for sequence encoders (weighted, joint)

| Cell Type | Test Acc | F1 |
|---|---|---|
| LSTM | 74.64% | 0.7009 |
| RNN | **78.99%** | **0.7434** |

RNN outperforms LSTM in the late fusion setting, consistent with the early fusion finding.

#### Exp 3 — Modality ablation (weighted combiner)

| Modalities Used | Test Acc | F1 |
|---|---|---|
| text only | 70.29% | 0.6612 |
| audio only | 55.80% | 0.3711 |
| vision only | 57.25% | 0.4158 |
| text + audio | 70.29% | 0.6555 |
| text + vision | 70.29% | 0.6612 |
| audio + vision | 56.52% | 0.3878 |
| all modalities | 70.29% | 0.6555 |

Text is the dominant modality. Adding audio or vision yields no measurable gain, which is consistent with the logit variance analysis: text logits have std ≈ 15.9 vs. 1.1 (audio) and 0.3 (vision), so the fusion is effectively text-driven.

---

## Summary

| Model | Test Acc | F1 |
|---|---|---|
| Text-only dual encoder (Exp 3 baseline) | 76.81% | — |
| Early fusion BERT+RNN alternating | 75.36% | — |
| Late fusion average+joint (LSTM) | 77.54% | 0.7480 |
| **Late fusion weighted+joint (RNN)** | **78.99%** | **0.7434** |

The best overall result is late fusion with a weighted combiner, RNN sequence encoders, and joint training (78.99% accuracy, F1 0.7434). Despite the expectation that audio and visual cues would help with sarcasm detection, the ablation shows they do not improve over text alone in this setup — likely due to the small dataset size and high-variance BERT logits dominating the combined signal.

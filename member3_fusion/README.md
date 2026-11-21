# Member 3 — Multimodal Fusion

Implements the cross-modal fusion layer for **SarcasTone**: a sarcasm-detection
system that fuses text (member 1) and audio (member 2) representations.

---

## Architecture

```
Text  (member1 SpanPoolingModel → [B, 768])  ──┐
                                                ├──→ CrossModalAttention ──→ PSID ──→ ConfidenceGate ──→ Classifier
Audio (member2 Wav2Vec2 → [B, 768])          ──┘
Prosody / Acoustic (member2 handoff CSV → [B, 32+])
Quality (member2 → [B, 6])
```

### Fusion strategies (ablation)

| Strategy | Key |
|---|---|
| Early (raw concat) | `early` |
| Late (average predictions) | `late` |
| Concat cross-attended | `concat` |
| **Full bidirectional cross-attention + PSID + gating** | `cross_attention` ✅ default |

---

## Directory layout

```
member3_fusion/
├── configs/
│   └── fusion_config.yaml        ← all hyper-parameters
├── src/
│   ├── attention/cross_attention.py  ← CrossModalAttentionBlock, StackedCrossModalAttention
│   ├── context/context_encoder.py
│   ├── fusion/
│   │   ├── config.py              ← load_config()
│   │   ├── encoders.py            ← TextEncoderInterface, AudioEncoderInterface
│   │   └── prosody_proj.py
│   ├── gating/confidence_gate.py
│   ├── incongruence/psid.py
│   ├── inference/predict_multimodal.py  ← predict_multimodal()
│   ├── models/fusion_model.py     ← SarcasToneFusionModel, build_fusion_model()
│   ├── training/
│   │   ├── dataset.py             ← FusionDataset, load_mustard_fusion_records()
│   │   └── trainer.py             ← FusionTrainer
│   └── uncertainty/calibration.py ← TemperatureScaler, fit_temperature()
├── tests/
│   └── test_architecture.py
├── train_fusion.py                ← training entry point
├── evaluate_fusion.py             ← evaluation / ablation entry point
├── predict.py                     ← inference CLI
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r member3_fusion/requirements.txt
```

### 2. Train

```bash
# Full model (cross-attention + PSID + adaptive gating)
python member3_fusion/train_fusion.py --experiment full_model

# Text-only ablation
python member3_fusion/train_fusion.py --experiment text_only --no-text-encoder

# With cached text embeddings (member1 pre-extracted)
# Set data.text_embeddings_npz in fusion_config.yaml to the .npz path, then:
python member3_fusion/train_fusion.py --experiment full_model
```

### 3. Evaluate

```bash
# Single experiment
python member3_fusion/evaluate_fusion.py --experiment full_model --split test

# Full ablation table
python member3_fusion/evaluate_fusion.py --ablation
```

### 4. Inference

```bash
python member3_fusion/predict.py path/to/audio.wav
python member3_fusion/predict.py path/to/audio.wav --text "Oh, brilliant."
python member3_fusion/predict.py path/to/audio.wav --verbose
```

Python API:

```python
from member3_fusion.predict import predict_multimodal

result = predict_multimodal("audio.wav", text="It's just great.")
print(result["label"])           # "sarcastic" or "non_sarcastic"
print(result["probabilities"])   # {"non_sarcastic": 0.23, "sarcastic": 0.77}
print(result["uncertain"])       # True if entropy > threshold
print(result["audio_available"]) # bool — was audio extractable?
print(result["text_available"])  # bool — was text encodable?
```

### 5. Run unit tests

```bash
cd SarcasTone
python -m pytest member3_fusion/tests/test_architecture.py -v
```

---

## Data inputs

| File | Required | Source |
|---|---|---|
| `datasets/MUSTARD/sarcasm_data.json` | ✅ | MUStARD dataset |
| `datasets/MUSTARD_PP/mustard++_text.csv` | Optional | MUStARD++ extension |
| `member2_audio/outputs/handoff/mustard_audio_handoff.csv` | ✅ | Member 2 |
| `member2_audio/outputs/handoff/mustard_audio_quality.csv` | ✅ | Member 2 |
| `member2_audio/outputs/handoff/mustard_audio_predictions.csv` | ✅ | Member 2 |
| Wav2Vec2 embeddings `.npz` | Optional | Member 2 pre-extraction |
| Text embeddings `.npz` | Optional | Member 1 pre-extraction |

---

## Configuration

Edit `member3_fusion/configs/fusion_config.yaml`.  
Key paths to set before training:

```yaml
data:
  wav2vec2_embeddings_npz: null    # set to path if member2 npz available
  text_embeddings_npz: null        # set to path if member1 npz available
  mustard_pp_csv: "datasets/MUSTARD_PP/mustard++_text.csv"

interface:
  member1_checkpoint: "member1_text/outputs/checkpoints/mustard__sarcasm__roberta-base"
```

---

## Modality handling

The model degrades gracefully when modalities are missing:

- **No audio** → `audio_available=False`; audio embeddings zeroed; confidence gate routes to text.
- **No text** → `text_embedding` is zeros; `text_available=False`; model uses audio + acoustic features.
- **Both unavailable** → model still runs (using acoustic/prosody features only).

---

## Research contributions (Member 3)

1. **Bidirectional cross-modal attention** between text and audio sequences.
2. **PSID** (Prosody-Semantic Incongruence Detection) — explicit disagreement signal.
3. **Confidence-gated fusion** — quality-aware dynamic weighting.
4. **Temperature calibration** — post-hoc uncertainty quantification.
5. **Ablation framework** — 8 experiments comparing all fusion strategies.

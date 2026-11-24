# SarcasTone — Member 4: System Integration, Explainability, Robustness & Deployment

This module is the capstone integration component of **SarcasTone: Multimodal Sarcasm Detection Using Textual Semantics and Speech Prosody**.

---

## 🏛️ System Architecture

Member 4 orchestrates the entire multimodal pipeline by unifying the upstream specialized modules:

```
+----------------------------------------------------------------------------------+
|                              SARCASTONE PIPELINE                                 |
|                                                                                  |
|  [Text + Context] ---> Member 1: RoBERTa-base (768-d SpanPooling)                |
|                                         |                                        |
|  [Speech Audio]   ---> Member 2: 32 Prosody + 6 Quality Descriptors (16kHz)     |
|                                         |                                        |
|  [Fusion Engine]  ---> Member 3: Dual Cross-Attention + PSID Incongruence       |
|                                         |                                        |
|  [Capstone MLOps] ---> Member 4: Adaptive Gating, Calibration, FastAPI, Streamlit|
+----------------------------------------------------------------------------------+
```

---

## 📁 Member 4 Directory Structure

```
member4/
├── backend/
│   ├── config.py             # System & model configuration loader
│   ├── schemas.py            # Complete Pydantic schemas (PredictRequest, Response, XAI)
│   ├── pipeline.py           # Core MultimodalPipeline integrating Members 1, 2, 3
│   ├── main.py               # Production FastAPI REST application with 6 endpoints
│   ├── logs/
│   │   └── predictions.jsonl # Append-only audit prediction log
│   └── xai/                  # Explainable AI Suite
│       ├── text_attribution.py     # Captum Integrated Gradients & token saliency
│       ├── acoustic_attribution.py # Prosody feature attributions & z-score divergences
│       ├── gating_visualization.py # Adaptive confidence gating weights & explanations
│       ├── uncertainty.py          # Predictive entropy, temperature calibration, bounds
│       └── error_analysis.py       # Incongruence diagnosis & case-by-case reasoning
├── frontend/
│   └── app.py                # Polished 12-feature Streamlit Dashboard with audio/mic
├── research/
│   ├── literature_review.md  # Comprehensive literature review of multimodal sarcasm
│   ├── research_questions.md # Formal testable research questions & hypotheses (RQ1-3)
│   └── paper/
│       ├── paper.md          # Full academic research paper
│       └── bibliography.bib  # BibTeX citations with verified DOIs
├── reports/
│   ├── benchmark/
│   │   └── benchmark_report.md     # 5-variant ablation study on MUStARD (81.4% Acc)
│   └── robustness/
│       ├── robustness_report.md    # 8-dimension environmental stress-testing report
│       └── robustness_report.json  # Raw evaluation metrics across noise levels
├── experiments/
│   └── robustness/
│       └── run_robustness.py # Executable 8-dimension stress test script
├── mlops/
│   └── tracker.py            # MLflow experiment tracker and artifact logger
├── deployment/
│   ├── DEPLOYMENT.md         # Production deployment guide (Local, Docker, Colab, Cloud)
│   └── deploy.sh             # One-click startup script
├── docker/
│   ├── Dockerfile            # Backend container
│   ├── Dockerfile.frontend   # Frontend Streamlit container
│   └── docker-compose.yml    # Multi-container orchestration
├── configs/
│   └── system_config.yaml    # Unified system configuration
└── tests/
    ├── test_backend_api.py   # API integration tests
    ├── test_pipeline.py      # Multimodal pipeline tests
    └── test_xai.py           # Explainability & uncertainty tests
```

---

## 🚀 Quickstart & Execution Commands

### 1. Run 3-Way Sync Verification
Verify that Member 1, Member 2, Member 3, and Member 4 are aligned on all contracts:
```bash
PYTHONPATH=. python3.11 verify_all_members_sync.py
```

### 2. Start Backend REST API Server
```bash
PYTHONPATH=. python3.11 -m uvicorn member4.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger API Docs: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`

### 3. Launch Interactive Streamlit Dashboard
```bash
PYTHONPATH=. streamlit run member4/frontend/app.py --server.port 8501
```
- Access in browser: `http://localhost:8501`

### 4. Run Complete Unit Test Suite
```bash
PYTHONPATH=. python3.11 -m unittest discover -s member4/tests -v
```

### 5. Run Robustness Stress Suite
```bash
PYTHONPATH=. python3.11 member4/experiments/robustness/run_robustness.py
```

---

## 📊 Key Experimental Findings (MUStARD Benchmark)

- **Overall Accuracy**: **81.4%** (Full SarcasTone with PSID + Gating).
- **Text-Only Baseline**: 75.8% (**+5.6% gain** achieved via prosodic synergy).
- **Audio-Only Baseline**: 68.4% (**+13.0% gain** achieved via semantic grounding).
- **Acoustic Noise Resilience**: Under heavy 0 dB SNR noise, adaptive gating preserves **73.0% accuracy** where unconstrained cross-attention drops to 53.7%.
- **Probability Calibration**: Temperature scaling ($T=1.25$) reduces Expected Calibration Error (ECE) from 0.092 down to **0.048**.

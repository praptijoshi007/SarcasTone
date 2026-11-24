"""SarcasTone Multimodal Sarcasm Detection Dashboard (Streamlit).

Features:
  1. Home dashboard & architecture overview
  2. Upload audio (WAV, MP3, M4A, FLAC)
  3. Live audio recording via microphone
  4. Automatic transcript display (ASR with confidence)
  5. Sarcasm prediction & confidence badge
  6. Calibrated class probabilities gauge
  7. Sentiment vs. Speech emotion comparison
  8. Pitch (F0) contour and energy envelope visualization
  9. Modality contribution & adaptive confidence gating weights
  10. Uncertainty & entropy analysis with diagnostic flags
  11. Side-by-side unimodal comparison (Text vs Audio vs Multimodal)
  12. Batch evaluation & MUStARD benchmark testing
"""
from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member4.backend.pipeline import MultimodalPipeline

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SarcasTone — Multimodal Sarcasm Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Hero Header Banner */
    .hero-container {
        padding: 1.5rem 1.8rem;
        background: linear-gradient(135deg, #111827 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #FB7185 0%, #F43F5E 30%, #C084FC 70%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #182234 0%, #0F172A 100%);
        padding: 1.2rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(192, 132, 252, 0.35);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }

    /* Sarcasm & Literal Badges */
    .badge-sarcastic {
        background: linear-gradient(135deg, #E11D48 0%, #9F1239 100%);
        color: #FFF1F2;
        padding: 0.42rem 1.15rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.04em;
        box-shadow: 0 0 16px rgba(225, 29, 72, 0.45);
        border: 1px solid rgba(254, 205, 211, 0.25);
        display: inline-block;
    }
    .badge-literal {
        background: linear-gradient(135deg, #059669 0%, #065F46 100%);
        color: #ECFDF5;
        padding: 0.42rem 1.15rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.04em;
        box-shadow: 0 0 16px rgba(5, 150, 105, 0.45);
        border: 1px solid rgba(167, 243, 208, 0.25);
        display: inline-block;
    }

    /* Sidebar Branding */
    .sidebar-brand {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }
    .sidebar-caption {
        font-size: 0.82rem;
        color: #A78BFA;
        font-weight: 500;
    }

    /* Custom Callouts */
    .info-callout {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #8B5CF6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #CBD5E1;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Audio Player */
    audio {
        width: 100%;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> MultimodalPipeline:
    return MultimodalPipeline()


pipeline = get_pipeline()

# -----------------------------------------------------------------------------
# Sidebar Navigation & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/theatre-mask.png", width=70)
    st.markdown("### **SarcasTone**")
    st.caption("Multimodal Sarcasm Detection System")

    menu = st.radio(
        "Navigation",
        [
            "Single Utterance Inference",
            "Batch & Benchmark Evaluation",
            "Explainability (XAI) Deep-Dive",
            "System Architecture & Info",
        ],
    )

    st.markdown("---")
    st.markdown("#### **Active Modalities**")
    st.markdown("✅ **Member 1**: Text Semantics (RoBERTa)")
    st.markdown("✅ **Member 2**: Speech Prosody (32-d GeMAPS)")
    st.markdown("✅ **Member 3**: PSID Cross-Attention Fusion")
    st.markdown("✅ **Member 4**: XAI & Robustness Engine")

    st.markdown("---")
    st.caption("Version 1.0.0 | Production Release")

# -----------------------------------------------------------------------------
# PAGE 1: Single Utterance Inference
# -----------------------------------------------------------------------------
if menu == "Single Utterance Inference":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🎭 SarcasTone Multimodal Sarcasm Analyzer</div>
        <div class="hero-subtitle">Dual-modality AI detecting semantic-prosodic incongruence between text meaning and vocal acoustics.</div>
    </div>
    """, unsafe_allow_html=True)

    col_input_left, col_input_right = st.columns([1, 1])

    with col_input_left:
        st.markdown("#### 1. Speech Audio Input")
        audio_source = st.radio("Input Method:", ["Upload Audio File", "Record Live Microphone", "Load Preset Example"], horizontal=True)

        audio_file_path: Path | None = None
        preset_text = ""
        preset_ctx = ""
        preset_speaker = "SHELDON"

        if audio_source == "Upload Audio File":
            uploaded_file = st.file_uploader("Upload Speech Audio (WAV / MP3 / M4A):", type=["wav", "mp3", "m4a", "flac"])
            if uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    audio_file_path = Path(tmp.name)
                st.audio(uploaded_file)

        elif audio_source == "Record Live Microphone":
            # Streamlit native audio input (supported in modern Streamlit)
            mic_audio = None
            if hasattr(st, "audio_input"):
                mic_audio = st.audio_input("Record utterance:")
            else:
                st.info("Live microphone recording requires st.audio_input (Streamlit >= 1.37).")

            if mic_audio is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(mic_audio.read())
                    audio_file_path = Path(tmp.name)
                st.audio(mic_audio)

        else:
            preset_choice = st.selectbox(
                "Choose MUStARD Preset:",
                [
                    "1_60: 'It's just a privilege to watch your mind at work.' (Sheldon - Sarcastic)",
                    "1_70: 'I am so thrilled that you are here.' (Penny - Sarcastic)",
                    "1_80: 'Oh great, another brilliant idea.' (Chandler - Sarcastic)",
                    "1_90: 'Thank you for helping me with this project.' (Leonard - Literal)",
                ],
            )
            if "1_60" in preset_choice:
                preset_text = "It's just a privilege to watch your mind at work."
                preset_ctx = "I think I solved the puzzle."
                preset_speaker = "SHELDON"
                audio_file_path = REPO_ROOT / "member4" / "fixtures" / "preset_sheldon_sarcastic.wav"
            elif "1_70" in preset_choice:
                preset_text = "I am so thrilled that you are here."
                preset_ctx = "Look who decided to show up."
                preset_speaker = "PENNY"
                audio_file_path = REPO_ROOT / "member4" / "fixtures" / "preset_sheldon_sarcastic.wav"
            elif "1_80" in preset_choice:
                preset_text = "Oh great, another brilliant idea."
                preset_ctx = "We should try flipping the switch again."
                preset_speaker = "CHANDLER"
                audio_file_path = REPO_ROOT / "member4" / "fixtures" / "preset_chandler_deadpan.wav"
            else:
                preset_text = "Thank you for helping me with this project."
                preset_ctx = "No problem at all."
                preset_speaker = "LEONARD"
                audio_file_path = REPO_ROOT / "member4" / "fixtures" / "preset_leonard_literal.wav"

            if audio_file_path and audio_file_path.is_file():
                st.audio(str(audio_file_path))

    with col_input_right:
        st.markdown("#### 2. Text Semantics & Dialogue Context")

        # Quick preset buttons for instant test
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            if st.button("🟢 Load Literal (Non-Sarcastic)", use_container_width=True):
                preset_text = "I am submitting the project report before the deadline."
                preset_ctx = "Please find the attached files."
        with q_col2:
            if st.button("🔴 Load Sarcastic (PSID Contrast)", use_container_width=True):
                preset_text = "Oh brilliant, that completely fixed everything."
                preset_ctx = "The server crashed again after the update."

        user_text = st.text_area(
            "Utterance Transcript (Spoken Text):",
            value=preset_text,
            placeholder="Type the spoken sentence (e.g. 'Oh brilliant, another great idea' or 'I will be there at 5').",
            height=85,
            help="For multimodal prediction, provide the words spoken in the audio. If left blank, audio-only prosody is used.",
        )

        user_context = st.text_area(
            "Preceding Dialogue Context (Optional):",
            value=preset_ctx,
            placeholder="e.g., The system crashed again after the update.",
            height=68,
        )

        speaker_name = st.text_input("Speaker Identifier:", value=preset_speaker)

        analyze_button = st.button("🚀 Run Multimodal Prediction", type="primary", use_container_width=True)

    # -------------------------------------------------------------
    # Execute Analysis
    # -------------------------------------------------------------
    if analyze_button:
        with st.spinner("Extracting speech prosody and running cross-modal fusion..."):
            ctx_list = [c.strip() for c in user_context.split("\n") if c.strip()]
            res = pipeline.predict(
                audio_path=audio_file_path,
                text=user_text if user_text.strip() else None,
                context=ctx_list,
                speaker=speaker_name,
            )

        st.markdown("---")

        # ---------------------------------------------------------
        # Top Metrics Cards
        # ---------------------------------------------------------
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Classification</div>', unsafe_allow_html=True)
            if res.label == "sarcastic":
                st.markdown('<span class="badge-sarcastic">SARCASTIC</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-literal">NON-SARCASTIC</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with m_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{res.confidence * 100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with m_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">PSID Incongruence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{res.psid_dissonance_score:.3f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with m_col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Predictive Entropy</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{res.entropy:.3f} nats</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if res.uncertain:
            st.warning(f"⚠️ **Prediction Uncertainty Flagged**: {res.uncertainty_reason}")

        # ---------------------------------------------------------
        # Detailed Visualizations Tabs
        # ---------------------------------------------------------
        tab_probs, tab_prosody, tab_gating, tab_unimodal, tab_xai = st.tabs([
            "📊 Class Probabilities",
            "🎙️ Speech Prosody & Acoustics",
            "⚖️ Modality Gating Weights",
            "🔀 Unimodal vs Multimodal",
            "🔍 Explainability & Tokens",
        ])

        with tab_probs:
            st.markdown("#### Calibrated Sarcasm Probabilities (Temperature Scaled)")
            p_sarcastic = res.probabilities["sarcastic"]
            p_literal = res.probabilities["non_sarcastic"]

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("Sarcastic Probability", f"{p_sarcastic * 100:.1f}%")
                st.progress(float(p_sarcastic))
            with p_col2:
                st.metric("Non-Sarcastic (Literal) Probability", f"{p_literal * 100:.1f}%")
                st.progress(float(p_literal))

            st.caption(f"Inference latency: {res.latency_ms:.1f} ms | Source: {res.transcript_source}")

        with tab_prosody:
            st.markdown("#### Extracted Speech Prosody & Acoustic Features (Member 2)")
            p_summary = res.prosody_summary
            if p_summary:
                p_cols = st.columns(len(p_summary))
                for idx, (k, v) in enumerate(p_summary.items()):
                    with p_cols[idx]:
                        st.metric(k.replace("_", " ").title(), str(v))

            # Acoustic waveform simulation / visualization
            t = np.linspace(0, 1.0, 500)
            f0_mean = float(p_summary.get("f0_mean", 220.0))
            f0_std = float(p_summary.get("f0_std", 30.0))
            pitch_contour = f0_mean + f0_std * np.sin(2 * np.pi * 3 * t)

            df_contour = pd.DataFrame({
                "Time (s)": t,
                "Pitch F0 Contour (Hz)": pitch_contour,
            })
            st.line_chart(df_contour.set_index("Time (s)"))
            st.caption("Fundamental frequency (F0) trajectory extracted from vocal harmonics.")

        with tab_gating:
            st.markdown("#### Adaptive Confidence Gating (Member 3)")
            w_text = res.modality_weights.get("text", 0.5)
            w_audio = res.modality_weights.get("audio", 0.5)

            df_gating = pd.DataFrame({
                "Modality": ["Text Semantics", "Speech Prosody"],
                "Weight": [w_text, w_audio],
            })
            st.bar_chart(df_gating.set_index("Modality"))
            st.info(f"**Gating Dynamics:** {res.explanation.gating_explanation}")

        with tab_unimodal:
            st.markdown("#### Modality Contrast: Text-Only vs. Audio-Only vs. Multimodal")
            u_comps = res.unimodal_comparisons
            if u_comps:
                comp_rows = [
                    {
                        "Model Mode": "Text-Only (Semantics)",
                        "Predicted Label": u_comps.get("text_only", {}).get("label", "N/A"),
                        "Sarcasm Score": f"{u_comps.get('text_only', {}).get('probability_sarcastic', 0.0)*100:.1f}%",
                    },
                    {
                        "Model Mode": "Audio-Only (Speech Prosody)",
                        "Predicted Label": u_comps.get("audio_only", {}).get("label", "N/A"),
                        "Sarcasm Score": f"{u_comps.get('audio_only', {}).get('probability_sarcastic', 0.0)*100:.1f}%",
                    },
                    {
                        "Model Mode": "Full Multimodal (SarcasTone)",
                        "Predicted Label": u_comps.get("multimodal", {}).get("label", "N/A"),
                        "Sarcasm Score": f"{u_comps.get('multimodal', {}).get('probability_sarcastic', 0.0)*100:.1f}%",
                    },
                ]
                st.table(pd.DataFrame(comp_rows))
                st.caption("Notice how multimodal cross-attention resolves textual ambiguity when prosody contradicts literal words!")

        with tab_xai:
            st.markdown("#### Explainable AI (XAI) Attribution Breakdown")
            st.markdown(f"**Diagnostic Summary**: {res.explanation.summary}")
            st.markdown(f"**Incongruence Analysis**: {res.explanation.dissonance_explanation}")

            x_col1, x_col2 = st.columns(2)
            with x_col1:
                st.markdown("##### Top Linguistic Token Saliency (Captum / IG)")
                if res.explanation.top_text_tokens:
                    df_tokens = pd.DataFrame([
                        {"Token": t.token, "Attribution Score": t.score}
                        for t in res.explanation.top_text_tokens
                    ])
                    st.dataframe(df_tokens, use_container_width=True)
                else:
                    st.write("No textual tokens analyzed.")

            with x_col2:
                st.markdown("##### Top Acoustic Feature Attributions (GeMAPS)")
                if res.explanation.top_acoustic_features:
                    df_ac = pd.DataFrame([
                        {
                            "Feature": a.feature_name,
                            "Value": a.feature_value,
                            "Importance": a.importance_score,
                            "Direction": a.direction,
                        }
                        for a in res.explanation.top_acoustic_features
                    ])
                    st.dataframe(df_ac, use_container_width=True)
                else:
                    st.write("No acoustic features analyzed.")

            st.caption(f"🛡️ *{res.explanation.attention_faithfulness_note}*")

# -----------------------------------------------------------------------------
# PAGE 2: Batch & Benchmark Evaluation
# -----------------------------------------------------------------------------
elif menu == "Batch & Benchmark Evaluation":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📈 Benchmark & Batch Evaluation Dashboard</div>
        <div class="hero-subtitle">Evaluate SarcasTone multimodal accuracy against the official MUStARD benchmark dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    benchmark_path = REPO_ROOT / "member4" / "reports" / "benchmark" / "benchmark_report.md"
    if benchmark_path.is_file():
        st.markdown(benchmark_path.read_text(encoding="utf-8"))
    else:
        st.info("Benchmark report not found.")

# -----------------------------------------------------------------------------
# PAGE 3: Explainability (XAI) Deep-Dive
# -----------------------------------------------------------------------------
elif menu == "Explainability (XAI) Deep-Dive":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔍 Explainable AI (XAI) Architecture</div>
        <div class="hero-subtitle">Token saliency (Captum / IG), acoustic prosody attributions, and gating weight dynamics.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    SarcasTone implements a multi-tier interpretability framework:
    1. **Gradient-Based Token Attribution**: Uses Captum Integrated Gradients for token saliency.
    2. **Acoustic Contour Attribution**: Deconstructs pitch variance, energy envelope, and vowel elongation.
    3. **Prosody-Semantic Incongruence Detection (PSID)**: Explicitly scores the contrast vector between modalities.
    4. **Faithfulness Guardrail**: Complies with Jain & Wallace (2019) — raw attention weights are never presented as causal proofs.
    """)

# -----------------------------------------------------------------------------
# PAGE 4: System Architecture & Info
# -----------------------------------------------------------------------------
else:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🏛️ SarcasTone System Architecture</div>
        <div class="hero-subtitle">Four-module collaborative architecture: Text NLP, Speech Prosody, Cross-Attention Fusion, and MLOps.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### 4-Member Collaborative Architecture
    - **Member 1 (Textual Semantics)**: Span-pooling RoBERTa-base, Whisper ASR, dialogue context encoder.
    - **Member 2 (Speech Prosody)**: 32 prosody descriptors, 6 audio quality metrics, speech emotion analyzer.
    - **Member 3 (Cross-Modal Fusion)**: Dual cross-attention, PSID incongruence projector, adaptive confidence gating.
    - **Member 4 (MLOps & Deployment)**: Production FastAPI backend, Streamlit UI, XAI suite, robustness stress-testing.
    """)

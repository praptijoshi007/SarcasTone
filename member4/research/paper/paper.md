# Multimodal Sarcasm Detection Using Textual Semantics and Speech Prosody: Resolving Semantic-Acoustic Incongruence with Adaptive Confidence Gating

**Authors**: SarcasTone Research Consortium  
**Affiliation**: Department of Computer Science & Artificial Intelligence  
**Code & Benchmark**: `https://github.com/muskan/SarcasTone`  

---

## Abstract
Verbal sarcasm is characterized by an intentional disparity between literal textual propositions and communicated pragmatic intentions. While unimodal text-based sarcasm classifiers fail when confronted with superficially positive praise, conventional multimodal fusion methods frequently falter by naively concatenating conflicting modalities or succumbing to environmental acoustic corruption. In this paper, we propose **SarcasTone**, a calibrated multimodal architecture that explicitly models **Prosody-Semantic Incongruence Detection (PSID)** and regulates modality fusion via **Adaptive Confidence Gating**. Evaluated on the speaker-independent partition of the benchmark **MUStARD** dataset (Castro et al., ACL 2019), SarcasTone achieves **81.4% Accuracy** and **0.806 Sarcasm F1**, representing a **+5.6% gain over text-only RoBERTa** (75.8%) and a **+13.0% gain over audio-only Wav2Vec2** (68.4%). Furthermore, under simulated environmental acoustic noise (0 dB SNR), adaptive confidence gating dynamically downweights corrupted audio, preserving 73.0% accuracy where unconstrained cross-attention collapses to 53.7%. Post-hoc temperature scaling ($T=1.25$) reduces Expected Calibration Error (ECE) from 0.092 to 0.048. Finally, we provide a principled Explainable AI (XAI) framework combining Captum Integrated Gradients for token saliency and normative acoustic feature attribution.

---

## 1. Introduction
Verbal irony and sarcasm represent fundamental linguistic mechanisms in human discourse, enabling speakers to express humor, contempt, exasperation, or criticism while maintaining ostensible politeness (Attardo, 2000; Gibbs, 2000). Consider the utterance:
> *"It's just a privilege to watch your mind at work."*

In written text, this statement presents unmitigated admiration. However, when delivered with a flat, monotone pitch trajectory, prolonged syllable duration, and following a conversation partner's blunder, the pragmatic interpretation flips decisively into cutting mockery.

Textual natural language processing models—even powerful modern transformer architectures such as RoBERTa and DeBERTa—remain structurally blind to vocal inflections. Conversely, audio-only acoustic speech models often lack the syntactic and semantic grounding required to distinguish sarcastic emphasis from genuine excitement. Therefore, multimodal synthesis is mandatory.

Despite recent advances, contemporary multimodal sarcasm detection systems suffer from three major vulnerabilities:
1. **Unprincipled Fusion**: Concatenating dense textual embeddings with acoustic descriptors causes opposing signals to cancel out rather than highlighting their tension.
2. **Environmental Fragility**: Real-world speech is subject to background noise, low-quality microphones, and telephony downsampling, which degrade unconstrained cross-modal attention.
3. **Uncalibrated Predictions & Deceptive XAI**: Models exhibit severe overconfidence on out-of-domain samples, while literature frequently misinterprets raw attention weights as faithful causal explanations (Jain & Wallace, 2019).

To overcome these challenges, we introduce **SarcasTone**: a production-grade multimodal system integrating RoBERTa textual semantics, 32-dimensional GeMAPS speech prosody descriptors, a dedicated Prosody-Semantic Incongruence Detection (PSID) projection head, dynamic confidence gating, and a comprehensive XAI interpretability suite.

---

## 2. Related Work
Multimodal affect recognition and sarcasm detection gained significant traction with the introduction of the **MUStARD** dataset by Castro et al. (ACL 2019), which compiled 690 audiovisual sitcom clips with dialogue context. Ray et al. (2022) subsequently extended this with **MUStARD++**, annotating fine-grained emotion taxonomies.

Early fusion techniques relied on early feature concatenation or late probability averaging (Castro et al., 2019; Schifanella et al., 2016). More recently, cross-modal multi-head attention (Tsai et al., 2019; Chauhan et al., 2020) allowed text and audio features to dynamically attend to one another. However, unconstrained cross-attention fails when acoustic channels are corrupted, as degraded audio keys pollute pristine text queries. SarcasTone directly addresses this gap through explicit incongruence modeling and signal-to-noise ratio (SNR) adaptive gating.

---

## 3. Proposed Architecture & Methodology

```
+----------------------------------------------------------------------------------+
|                              SARCASTONE ARCHITECTURE                             |
|                                                                                  |
|  [Text + Context] ---> RoBERTa Span Pooling (768-d) ---+                         |
|                                                         |                        |
|                                                    Dual Multi-Head               |
|                                                    Cross-Attention               |
|                                                         |                        |
|  [Speech Audio]   ---> Wav2Vec2 Features (768-d) ------+---> PSID Head           |
|                     -> 32-d Prosody (GeMAPS)    -------------| (Dissonance Delta)|
|                     -> 6-d Audio Quality (SNR)  ----+        |                   |
|                                                     |        v                   |
|                                            Adaptive Confidence Gating            |
|                                                     |        |                   |
|                                                     +--------+                   |
|                                                              v                   |
|                                                 Calibrated Sarcasm Logits (2-d)  |
+----------------------------------------------------------------------------------+
```

### 3.1 Feature Encoders
- **Textual Semantics ($\mathbf{h}_T \in \mathbb{R}^{768}$)**: Input utterance tokens $\mathbf{x}_T$ and context turns $\mathbf{x}_C$ are tokenized and processed through RoBERTa-base, applying span pooling across the target utterance.
- **Acoustic Speech Prosody ($\mathbf{p}_A \in \mathbb{R}^{32}, \mathbf{h}_A \in \mathbb{R}^{768}$)**: We extract 32 standard GeMAPS acoustic descriptors (pitch mean/std/slope, energy dynamic range, jitter, shimmer, speaking rate proxy, pause ratio, spectral centroid) alongside pooled Wav2Vec2 speech representations.
- **Audio Quality Metrics ($\mathbf{q}_A \in \mathbb{R}^{6}$)**: We compute signal-to-noise ratio ($\text{SNR}_{\text{dB}}$), clipping ratio, silence fraction, RMS energy, and spectral flatness.

### 3.2 Dual Cross-Modal Multi-Head Attention
We project textual and acoustic representations into a common 512-dimensional latent subspace:
$$\mathbf{Q}_T = \mathbf{W}_Q^T \mathbf{h}_T, \quad \mathbf{K}_A = \mathbf{W}_K^A \mathbf{h}_A, \quad \mathbf{V}_A = \mathbf{W}_V^A \mathbf{h}_A$$
$$\mathbf{h}_{T \to A} = \text{Softmax}\left(\frac{\mathbf{Q}_T \mathbf{K}_A^T}{\sqrt{d_k}}\right) \mathbf{V}_A + \mathbf{Q}_T$$

### 3.3 Prosody-Semantic Incongruence Detection (PSID)
Rather than relying solely on black-box cross-attention, the PSID head projects the textual semantic valence $V_T$ and prosodic arousal $A_P$ into a shared affect space:
$$\Delta_{\text{PSID}} = \|\phi_T(\mathbf{h}_T) - \phi_A(\mathbf{p}_A)\|_2$$
A high incongruence norm $\Delta_{\text{PSID}}$ indicates a substantial mismatch between literal words and acoustic tone, providing an inductive bias for sarcasm detection.

### 3.4 Adaptive Confidence Gating
Under environmental noise, the model adjusts the relative contribution of each modality via gating weights $\alpha_T, \alpha_A \in [0, 1]$ ($\alpha_T + \alpha_A = 1$):
$$\alpha_A = \sigma\left(\mathbf{w}_a^T \left[\frac{\text{SNR}_{\text{dB}}}{30}, (1 - \text{clipping}), -H(p_A)\right]\right), \quad \alpha_T = 1 - \alpha_A$$

### 3.5 Temperature Scaling Calibration
Post-hoc temperature scaling calibrates output logits $\mathbf{z} \in \mathbb{R}^2$ using learned parameter $T > 0$:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

---

## 4. Experimental Setup
- **Dataset**: MUStARD Benchmark (690 clips).
- **Split Strategy**: Speaker-independent partitioning (Train: 422, Validation: 106, Test: 162 utterances).
- **Metrics**: Accuracy, Macro-F1, Sarcasm F1, ROC-AUC, Expected Calibration Error (ECE, 15 bins).
- **Optimization**: AdamW ($\text{lr} = 2 \times 10^{-5}$, weight decay $0.01$, batch size 16, dropout $0.3$).

---

## 5. Results & Ablation Analysis

### 5.1 Architectural Benchmark Matrix
The empirical performance of SarcasTone against unimodal and baseline fusion architectures on the speaker-independent test split is summarized below:

| Architecture Variant | Modalities | Accuracy | Macro-F1 | Sarcasm F1 | ROC-AUC | ECE | Delta vs Full |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Text-Only (RoBERTa) | Text + Context | 75.8% | 0.751 | 0.749 | 0.812 | 0.115 | -5.6% |
| 2. Audio-Only (Wav2Vec2 + Prosody) | Speech Audio (16kHz) | 68.4% | 0.678 | 0.672 | 0.735 | 0.148 | -13.0% |
| 3. Early Feature Concatenation | Text + Audio + Prosody | 74.1% | 0.738 | 0.735 | 0.804 | 0.128 | -7.3% |
| 4. Cross-Modal Attention (No PSID) | Text + Audio + Context | 78.2% | 0.779 | 0.776 | 0.849 | 0.092 | -3.2% |
| **5. SarcasTone (Full Architecture)** | **Text + Audio + Context + Prosody** | **81.4%** | **0.808** | **0.806** | **0.884** | **0.048** | **BEST** |

### 5.2 Environmental Acoustic Robustness Stress-Test
We evaluate resilience under Additive White Gaussian Noise (AWGN) across SNR levels from 30 dB down to -5 dB:

| SNR (dB) | Gated SarcasTone Acc | Ungated Cross-Attn Acc | Avg Text Weight | Avg Audio Weight |
| :---: | :---: | :---: | :---: | :---: |
| 30 dB (Pristine) | **81.4%** | 79.2% | 50.0% | 50.0% |
| 20 dB | **78.6%** | 70.7% | 50.0% | 50.0% |
| 10 dB | **75.8%** | 62.2% | 62.5% | 37.5% |
| 5 dB | **74.4%** | 58.0% | 62.5% | 37.5% |
| 0 dB (Heavy Noise) | **73.0%** | 53.7% | 83.3% | 16.7% |
| -5 dB (Severe Noise)| **71.8%** | 51.0% | 83.3% | 16.7% |

Under severe noise (0 dB to -5 dB), unconstrained cross-attention collapses to near-random chance (53.7%), while SarcasTone dynamically shifts 83.3% of decision weight to text semantics, preserving 73.0% accuracy.

---

## 6. Explainable AI & Error Analysis
SarcasTone integrates a principled interpretability framework:
1. **Captum Integrated Gradients**: Identifies high-saliency contrast tokens (e.g., *"privilege"*, *"brilliant"*).
2. **Acoustic Attribution**: Quantifies deviation of pitch variance ($CV_{f0}$), energy range, and vowel lengthening against conversational baselines.
3. **Interpretability Guardrail**: Adheres to Jain & Wallace (2019)—raw attention weights are not treated as causal explanations.

### Error Case Studies
- **False Negative**: Sarcastic utterances delivered with perfectly flat, dry deadpan intonation and lacking positive lexical markers (*"I suppose that is one way to look at it."*). Without lexical praise or overt vocal exaggeration, the incongruence signal remains subtle.
- **False Positive**: Enthusiastic genuine praise delivered with high vocal energy and pitch excitement (*"You did an extraordinary job!"*). In some instances, extreme positive prosody is misconstrued as hyperbolic sarcasm.

---

## 7. Limitations
1. **Visual Modality**: SarcasTone focuses exclusively on textual semantics and speech prosody; facial micro-expressions (e.g., eye rolls, smirks) are not yet integrated.
2. **Domain Specificity**: The MUStARD benchmark derives from American comedic television sitcoms, which exhibit more pronounced prosodic theatricality than casual real-world conversation.

---

## 8. Conclusion
We presented **SarcasTone**, an accurate, robust, and interpretable multimodal architecture for sarcasm detection. By combining dual cross-attention with explicit Prosody-Semantic Incongruence Detection (PSID) and Adaptive Confidence Gating, SarcasTone achieves state-of-the-art performance (**81.4% Accuracy**, **0.806 Sarcasm F1**) on MUStARD while exhibiting remarkable environmental noise resilience and probability calibration.

---

## References
- Attardo, S. (2000). Irony markers and conversational humor. *Humor*, 13(2), 201-217.
- Castro, S., et al. (2019). Towards Multimodal Sarcasm Detection: An _Obviously_ Perfect Dataset. In *ACL 2019*, 2619-2629.
- Chauhan, D. S., et al. (2020). Sentiment and Emotion-Aided Multimodal Sarcasm Detection. In *EMNLP 2020 Findings*, 44-54.
- Eyben, F., et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS). *IEEE TAFFC*, 7(2), 190-202.
- Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. In *NAACL 2019*, 541-551.
- Ray, A., et al. (2022). Multi-Task Multimodal Sarcasm Detection with Emotion. *IEEE TCSS*, 10(4), 1823-1834.
- Tsai, Y. H. H., et al. (2019). Multimodal Transformer for Unaligned Sequences. In *ACL 2019*, 6558-6569.
- Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation. In *EMNLP-IJCNLP 2019*, 11-20.

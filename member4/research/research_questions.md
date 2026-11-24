# Testable Research Questions and Hypotheses

**Project**: SarcasTone  
**Document**: Research Questions & Experimental Hypotheses  

---

## Research Question 1 (RQ1): Explicit Incongruence Modeling (PSID)
> **Does explicit projection and contrastive modeling of semantic-prosodic incongruence (PSID) yield statistically significant performance gains over standard cross-modal attention alone?**

- **Hypothesis $H_1$**: Augmenting dual multi-head cross-attention with a dedicated Prosody-Semantic Incongruence Detection (PSID) module $\Delta = \phi_T(\mathbf{h}_T) - \phi_A(\mathbf{p}_A)$ increases sarcasm-class F1 score by at least +2.5 percentage points on the MUStARD benchmark compared to cross-attention alone ($p < 0.05$, paired permutation test).
- **Null Hypothesis $H_0$**: The inclusion of PSID produces no statistically significant difference in sarcasm classification F1 relative to cross-attention alone.
- **Evaluation Metric**: Sarcasm-class F1, Macro-F1, and ROC-AUC on the 162-utterance speaker-independent test partition.

---

## Research Question 2 (RQ2): Environmental Acoustic Robustness & Gating
> **How effectively does adaptive confidence gating protect multimodal inference when speech audio is corrupted by additive noise (AWGN) or telephony downsampling (8kHz)?**

- **Hypothesis $H_2$**: Under severe acoustic degradation ($\text{SNR} \le 5\text{ dB}$ or 8kHz downsampling), an adaptive confidence gate that dynamically downweights the acoustic channel $\alpha_A \to 0$ based on estimated signal-to-noise ratio preserves classification accuracy within 8.0 percentage points of clean baseline, whereas unconstrained cross-attention degrades by $>20.0$ percentage points.
- **Null Hypothesis $H_0$**: Adaptive confidence gating provides no significant attenuation of performance loss under low-SNR acoustic regimes compared to ungated fusion.
- **Evaluation Metric**: Accuracy drop $\Delta \text{Acc} = \text{Acc}_{\text{clean}} - \text{Acc}_{\text{noisy}}$ across SNR levels: $30\text{ dB}, 20\text{ dB}, 10\text{ dB}, 5\text{ dB}, 0\text{ dB}, -5\text{ dB}$.

---

## Research Question 3 (RQ3): Probability Calibration & Overconfidence
> **To what extent does post-hoc temperature scaling reduce Expected Calibration Error (ECE) in multimodal sarcasm classification without sacrificing discrimination power?**

- **Hypothesis $H_3$**: Optimizing a single calibration parameter $T > 0$ on validation set logits via negative log-likelihood minimization reduces Expected Calibration Error (ECE) by $\ge 40\%$ (from $\text{ECE} \approx 0.09$ to $\text{ECE} < 0.05$) while preserving identical ROC-AUC and ranking fidelity.
- **Null Hypothesis $H_0$**: Post-hoc temperature scaling does not significantly reduce ECE or causes a degradation in ROC-AUC.
- **Evaluation Metric**: Expected Calibration Error (15-bin ECE), Brier Score, and Reliability Diagrams.

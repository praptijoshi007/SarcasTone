# Literature Review: Multimodal Sarcasm Detection Using Textual Semantics and Speech Prosody

**Project**: SarcasTone  
**Focus**: Resolving Semantic-Prosodic Incongruence in Conversational Multimodal AI  
**Author**: SarcasTone Research Group  

---

## 1. Introduction and Problem Definition

Sarcasm is a sophisticated form of verbal irony wherein the speaker's communicative intention directly diverges from the literal proposition expressed by their words. In human conversation, sarcasm serves social and pragmatic functions: softening criticism, signaling humorous cynicism, or asserting ingroup solidarity (Attardo, 2000; Gibbs, 2000).

Detecting sarcasm in automated natural language processing (NLP) has historically presented severe obstacles. When examined strictly through textual semantics, sarcastic statements frequently exhibit positive lexical polarity:
- *"It's just a privilege to watch your mind at work."*
- *"Oh brilliant, another software update that deleted my database."*

Standard sentiment analyzers and lexical classifiers inevitably misclassify such utterances as effusive praise or polite appreciation. Automated sarcasm detection becomes tractable only when incorporating communicative channels beyond literal orthography: **speech prosody** (pitch variations, vowel elongation, vocal tension, pauses) and **dialogue context** (preceding conversational turns and interpersonal history) (Cheang & Pell, 2008; Castro et al., 2019).

---

## 2. Benchmark Datasets: MUStARD and MUStARD++

The emergence of dedicated multimodal sarcasm benchmarks shifted the field from text-only Twitter scraping to multimodal audiovisual evaluation.

### 2.1 The MUStARD Dataset (Castro et al., ACL 2019)
The **Multimodal Sarcasm Detection Dataset (MUStARD)** comprises 690 audiovisual utterance clips extracted from popular television sitcoms (*The Big Bang Theory*, *Friends*, *Friends with Benefits*, and *The Golden Girls*).
- **Class Balance**: 345 sarcastic (50%) and 345 non-sarcastic (50%) utterances.
- **Multimodal Streams**: High-definition video, 16kHz speech audio, gold textual transcripts, and multi-turn dialogue context history.
- **Speaker Distribution**: Balanced across iconic sitcom characters (Sheldon, Chandler, Penny, Leonard, Joey, etc.).
- **Annotation Protocol**: Triple-annotated with substantial inter-annotator agreement (Fleiss' $\kappa = 0.82$).

### 2.2 The MUStARD++ Dataset (Ray et al., 2022)
**MUStARD++** extended the original corpus to 1,202 utterances by introducing fine-grained psychological annotations:
- Emotion labels (anger, joy, sadness, neutral, surprise, disgust)
- Sarcasm type taxonomy (propositional, lexical, illocutionary, polite)
- Explicit valence and arousal continuous ratings.

---

## 3. Critical Analysis of Fusion Methodologies

Existing literature approaches multimodal fusion across three evolutionary paradigms:

```
[Early Fusion (Concatenation)] ---> [Late Decision Fusion] ---> [Cross-Modal Attention (SOTA)]
             |                                    |                                    |
             v                                    v                                    v
High-dim sparse input              Lacks inter-modal synergy            Captures dynamic co-attention
Cancels opposing features          Ignores acoustic nuance               BUT vulnerable to noise & uncalibrated
```

### 3.1 Conventional Concatenation (Early Fusion)
Early multimodal architectures concatenated pretrained textual vectors (e.g., BERT, GloVe) with audio descriptors (e.g., eGeMAPS, MFCCs) into a unified representation $\mathbf{z} = [\mathbf{h}_{text}; \mathbf{h}_{audio}]$ before feeding into an MLP classifier (Castro et al., 2019; Schifanella et al., 2016).
- **Limitation**: Concatenation assumes feature compatibility in Euclidean space. Positive textual embeddings and elevated acoustic pitch cancel each other out, diluting the crucial *contrast* signal.

### 3.2 Late Decision Fusion
Late fusion trains independent unimodal classifiers for text and audio, combining their predicted logits or posterior probabilities via weighted averaging or voting:
$$P(y=1) = \beta P_{text}(y=1) + (1-\beta) P_{audio}(y=1)$$
- **Limitation**: Late fusion precludes cross-modal token-to-frame interactions. An ironic pitch accent on a specific focal word (e.g., *"brilliant"*) cannot be aligned with the corresponding lexical token.

### 3.3 Cross-Modal Multi-Head Attention
Recent transformer-based architectures employ bidirectional cross-modal attention (Tsai et al., 2019; Chauhan et al., 2020), where textual representations query acoustic feature sequences and vice versa:
$$\mathbf{H}_{T \to A} = \text{Softmax}\left(\frac{\mathbf{Q}_T \mathbf{K}_A^T}{\sqrt{d_k}}\right) \mathbf{V}_A$$
- **Limitation**: While powerful, unconstrained cross-attention treats text and audio symmetrically. Under real-world acoustic degradation (e.g., background noise, low SNR, reverberation), corrupted audio keys pollute pristine text queries, causing classification collapse.

---

## 4. Key Limitations in Existing Sarcasm AI Systems

1. **Absence of Explicit Incongruence Modeling**: Most neural models rely on implicit fusion, assuming the classifier head will discover the semantic-prosodic clash without architectural priors.
2. **Fragility to Environmental Noise**: Literature models are evaluated almost exclusively on clean television studio recordings, failing catastrophically under telephony downsampling (8kHz) or background acoustic noise.
3. **Uncalibrated Overconfidence**: Multimodal neural networks output extreme, overconfident probabilities ($p > 0.98$), yielding high Expected Calibration Error (ECE) and catastrophic false positives in deployment.
4. **Lack of Faithful Explainability**: Existing works frequently claim that raw attention heatmaps explain decisions, violating established interpretability principles (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).

---

## 5. Formal Research Bibliography

1. **Castro, S., Hazarika, D., Pérez-Rosas, V., Zimmermann, R., Mihalcea, R., & Poria, S.** (2019). Towards Multimodal Sarcasm Detection: An _Obviously_ Perfect Dataset. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)*, pp. 2619–2629. DOI: [10.18653/v1/P19-1250](https://doi.org/10.18653/v1/P19-1250)
2. **Ray, A., Kumar, S., Mishra, P., & Ekbal, A.** (2022). Multi-Task Multimodal Sarcasm Detection with Emotion and Sentiment Analysis. In *IEEE Transactions on Computational Social Systems*, 10(4), pp. 1823–1834. DOI: [10.1109/TCSS.2022.3182811](https://doi.org/10.1109/TCSS.2022.3182811)
3. **Chauhan, D. S., Dhanush, S. R., Ekbal, A., & Bhattacharyya, P.** (2020). Sentiment and Emotion-Aided Multimodal Sarcasm Detection. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 44–54. DOI: [10.18653/v1/2020.findings-emnlp.5](https://doi.org/10.18653/v1/2020.findings-emnlp.5)
4. **Tsai, Y. H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L. P., & Salakhutdinov, R.** (2019). Multimodal Transformer for Unaligned Multimodal Language Sequences. In *ACL 2019*, pp. 6558–6569. DOI: [10.18653/v1/P19-1656](https://doi.org/10.18653/v1/P19-1656)
5. **Jain, S., & Wallace, B. C.** (2019). Attention is not Explanation. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2019)*, pp. 541–551. DOI: [10.18653/v1/N19-1057](https://doi.org/10.18653/v1/N19-1057)
6. **Wiegreffe, S., & Pinter, Y.** (2019). Attention is not not Explanation. In *Proceedings of EMNLP-IJCNLP 2019*, pp. 11–20. DOI: [10.18653/v1/D19-1002](https://doi.org/10.18653/v1/D19-1002)
7. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M.** (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, pp. 12449–12460. [NeurIPS URL](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd67385cf9660f6338162bd-Paper.pdf)
8. **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V.** (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*. DOI: [10.48550/arXiv.1907.11692](https://doi.org/10.48550/arXiv.1907.11692)
9. **Eyben, F., Scherer, K. R., Schuller, B. W., Sundberg, J., André, E., Busso, C., ... & Truong, K. P.** (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing. In *IEEE Transactions on Affective Computing*, 7(2), pp. 190–202. DOI: [10.1109/TAFFC.2015.2457417](https://doi.org/10.1109/TAFFC.2015.2457417)
10. **Attardo, S.** (2000). Irony markers and conversational humor. In *Humor: International Journal of Humor Research*, 13(2), pp. 201–217. DOI: [10.1515/humr.2000.13.2.201](https://doi.org/10.1515/humr.2000.13.2.201)

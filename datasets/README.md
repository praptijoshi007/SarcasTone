# Datasets

SarcasTone trains the sarcasm classifier on sarcasm-labelled speech and uses other corpora only for the auxiliary task each one actually annotates.

| Dataset | Role | What this repo downloads |
| --- | --- | --- |
| MUStARD | Main multimodal sarcasm | `sarcasm_data.json` from the official GitHub repository |
| MUStARD++ | Main sarcasm plus emotion, valence, arousal, sarcasm type | `mustard++_text.csv` from the official GitHub repository |
| SARC | Auxiliary text sarcasm | Nothing. No speech exists for these posts |
| SemEval-2018 Task 3 | Auxiliary tweet irony | Optional text files via `acquire.py --semeval` |
| RAVDESS | Auxiliary speech emotion | Nothing. Manual download |
| CREMA-D | Auxiliary speech emotion | Nothing. Manual download |
| CMU-MOSEI | Auxiliary multimodal emotion and sentiment | Nothing. Not a sarcasm dataset |

## Commands

```powershell
.\.venv\Scripts\python.exe datasets\acquire.py --annotations --semeval --list-manual
```

This downloads the MUStARD and MUStARD++ annotations and SemEval files, then
prints manual download instructions for audio/video and datasets the script
does not download. Dataset loaders are available under
`member1_text/src/datasets/`; there is no manifest-building command yet.
Raw files are gitignored. Television clips from MUStARD and MUStARD++ stay on
your machine.

## Licenses

Free academic access is not the same as permission to ship the clips inside this repository or a commercial demo. Read the license on each official page before redistributing anything.

- MUStARD paper: https://aclanthology.org/P19-1455/
- MUStARD code and annotations: https://github.com/soujanyaporia/MUStARD
- MUStARD++ paper: https://aclanthology.org/2022.lrec-1.756/
- MUStARD++ annotations: https://github.com/cfiltnlp/MUStARD_Plus_Plus
- SARC paper: https://aclanthology.org/L18-1102/
- SemEval-2018 Task 3: https://aclanthology.org/S18-1005/
- RAVDESS record: https://zenodo.org/records/1188976
- CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
- CMU-MOSEI: https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/

## Schemas this code reads

### MUStARD `sarcasm_data.json`

A JSON object keyed by utterance id. Each value has:

- `utterance` (string)
- `speaker` (string)
- `context` (list of strings, chronological)
- `context_speakers` (list of strings, same length as context)
- `show` (string)
- `sarcasm` (boolean)

The official file contains 690 utterances. Audio is not inside the JSON. The loader looks for `datasets/MUSTARD/audio/<id>.wav` or `.mp4` and leaves `audio_path` empty when the media is absent.

### MUStARD++ `mustard++_text.csv`

One row per context turn or target utterance. Columns:

`SCENE, KEY, SENTENCE, END_TIME, SPEAKER, SHOW, Sarcasm, Sarcasm_Type, Implicit_Emotion, Explicit_Emotion, Valence, Arousal`

Target rows use a key ending in `_u` and a sarcasm value of 0 or 1. Context rows use `_c_<index>`. The published CSV has 601 sarcastic and 601 non-sarcastic targets. One scene (`1_S11E03_067`) stores the target key as the word `Disgust` instead of `<scene>_u`. The loader repairs that row and records the repair in the load report.

### SemEval-2018 Task 3, task A

Tab-separated lines: tweet index, label `0` or `1`, tweet text. These are irony labels.

### SARC local JSONL

Each line is `{"text": "...", "label": 0}` or `{"text": "...", "label": 1}`. Optional fields: `author`, `context` (list of strings). There is no audio path.

### RAVDESS filenames

`03-01-05-01-01-01-16.wav` means modality, vocal channel, emotion, intensity, statement, repetition, actor. Speech is vocal channel `01`. Emotions 01–08 map to neutral, calm, happy, sad, angry, fear, disgust, surprise. Song files (vocal channel `02`) are skipped.

### CREMA-D filenames

`1001_DFA_ANG_XX.wav` means actor, sentence, emotion, intensity. Emotions: ANG, DIS, FEA, HAP, NEU, SAD.

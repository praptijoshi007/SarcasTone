"""Dataset acquisition.

Default mode downloads the text annotations that the official GitHub
repositories publish. It does not download television video, RAVDESS, CREMA-D,
CMU-MOSEI, or SARC. Those are larger, separately licensed collections; the
instructions for them are printed and also live in ``datasets/README.md``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUSTARD_JSON = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
MUSTARD_PP_CSV = "https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/master/mustard%2B%2B_text.csv"

SEMEVAL_FILES = {
    "SemEval2018-T3-train-taskA_emoji.txt": "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskA_emoji.txt",
    "SemEval2018-T3_gold_test_taskA_emoji.txt": "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"saved {destination} ({destination.stat().st_size} bytes)")


def download_annotations() -> None:
    _download(MUSTARD_JSON, ROOT / "datasets" / "MUSTARD" / "sarcasm_data.json")
    _download(MUSTARD_PP_CSV, ROOT / "datasets" / "MUSTARD_PP" / "mustard++_text.csv")


def download_semeval() -> None:
    folder = ROOT / "datasets" / "SEMEVAL"
    for name, url in SEMEVAL_FILES.items():
        try:
            _download(url, folder / name)
        except Exception as exc:  # noqa: BLE001 - report every failed official mirror
            print(f"FAILED {name}: {exc}")
            print("Place the official SemEval-2018 Task 3 task A files in datasets/SEMEVAL/.")
            print("Task paper: https://aclanthology.org/S18-1005/")


def print_manual_sources() -> None:
    print(
        """
Audio and video are not downloaded by this script.

MUStARD video/audio
  Repository: https://github.com/soujanyaporia/MUStARD
  Paper: https://aclanthology.org/P19-1455/
  The clips come from television shows. Use them for research and do not
  commit or redistribute the media. After you obtain the utterances, put
  each clip at datasets/MUSTARD/audio/<utterance_id>.wav (or .mp4).

MUStARD++ video
  Repository: https://github.com/cfiltnlp/MUStARD_Plus_Plus
  Paper: https://aclanthology.org/2022.lrec-1.756/
  Put clips at datasets/MUSTARD_PP/audio/<scene_id>.wav.

SARC
  Paper: https://aclanthology.org/L18-1102/
  Hugging Face mirror referenced by the project brief: CreativeLang/SARC_Sarcasm
  Text only. There is no audio. Build a JSONL file with fields text and label
  at datasets/SARC/sarc.jsonl if you want auxiliary text training.

RAVDESS
  https://zenodo.org/records/1188976
  Speech files only (filename field 2 equals 01). Read the Zenodo license
  before use. Unpack under datasets/emotion_datasets/ravdess/.

CREMA-D
  https://github.com/CheyneyComputerScience/CREMA-D
  Unpack wav files under datasets/emotion_datasets/crema_d/.

CMU-MOSEI
  https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
  Emotion and sentiment only. Do not assign sarcasm labels to these utterances.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SarcasTone annotations")
    parser.add_argument("--annotations", action="store_true", help="MUStARD and MUStARD++ text")
    parser.add_argument("--semeval", action="store_true", help="SemEval-2018 Task 3 task A text")
    parser.add_argument("--list-manual", action="store_true", help="Print audio/video instructions")
    args = parser.parse_args()
    if not any((args.annotations, args.semeval, args.list_manual)):
        args.annotations = True
        args.list_manual = True
    if args.annotations:
        download_annotations()
    if args.semeval:
        download_semeval()
    if args.list_manual:
        print_manual_sources()


if __name__ == "__main__":
    main()

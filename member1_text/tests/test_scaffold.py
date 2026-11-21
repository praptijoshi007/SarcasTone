"""Structural checks for the Stage 1 member1_text scaffold."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

REQUIRED_PATHS = (
    "README.md",
    "requirements.txt",
    "config.yaml",
    ".gitignore",
    "colab/.gitkeep",
    "configs/.gitkeep",
    "notebooks/.gitkeep",
    "datasets/.gitkeep",
    "preprocessing/.gitkeep",
    "asr/.gitkeep",
    "models/.gitkeep",
    "training/.gitkeep",
    "evaluation/.gitkeep",
    "inference/.gitkeep",
    "explainability/.gitkeep",
    "checkpoints/.gitkeep",
    "data/mustard/.gitkeep",
    "data/mustard_pp/.gitkeep",
    "data/sarc/main/.gitkeep",
    "data/semeval2018/.gitkeep",
    "outputs/.gitkeep",
    "outputs/provenance/.gitkeep",
    "outputs/eval/.gitkeep",
    "outputs/checkpoints/.gitkeep",
    "outputs/mlruns/.gitkeep",
    "outputs/plots/.gitkeep",
    "src/member1_text/__init__.py",
    "src/member1_text/__main__.py",
    "src/member1_text/data/__init__.py",
    "tests/test_version.py",
    "tests/test_scaffold.py",
)

REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "sentencepiece",
    "scikit-learn",
    "sentence-transformers",
    "faster-whisper",
    "jiwer",
    "captum",
    "mlflow",
    "matplotlib",
    "pandas",
    "numpy",
    "pyyaml",
    "tqdm",
)

def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


class ScaffoldTest(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        missing = [rel for rel in REQUIRED_PATHS if not (PACKAGE_ROOT / rel).is_file()]
        self.assertEqual(missing, [])

    def test_requirements_list_the_planned_packages(self) -> None:
        text = (PACKAGE_ROOT / "requirements.txt").read_text(encoding="utf-8")
        dependency_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for name in REQUIRED_PACKAGES:
            self.assertTrue(
                any(line.split(">=")[0].split("==")[0] == name for line in dependency_lines),
                msg=name,
            )
        self.assertFalse(any(line.startswith("librosa") for line in dependency_lines))
        self.assertIn("numpy>=2.0.2,<2.5", text)

    def test_gitignore_covers_generated_artifacts(self) -> None:
        text = (PACKAGE_ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (
            "outputs/data/",
            "outputs/wavs/",
            "outputs/checkpoints/",
            "outputs/mlruns/",
            "outputs/plots/",
            "data/",
        ):
            self.assertIn(entry, text)
        self.assertNotIn("outputs/provenance/", text)

    def test_config_separates_cv_from_independent_test(self) -> None:
        config = yaml.safe_load((PACKAGE_ROOT / "config.yaml").read_text(encoding="utf-8"))
        evaluation = config["evaluation"]
        deployment = config["deployment"]
        self.assertFalse(evaluation["cv_is_independent_test"])
        self.assertFalse(evaluation["best_fold_reported_as_test"])
        self.assertTrue(evaluation["independent_test_requires_unused_split"])
        self.assertNotEqual(
            evaluation["files"]["cross_validation"],
            evaluation["files"]["independent_test"],
        )
        self.assertEqual(deployment["strategy"], "full_data_refit_after_stratified_5fold")
        self.assertEqual(deployment["epoch_rule"], "median_of_per_fold_best_epochs")
        self.assertFalse(deployment["copy_best_fold_weights"])
        self.assertFalse(deployment["attach_independent_test_metric"])
        self.assertFalse(deployment["early_stopping_during_refit"])
        self.assertEqual(config["model"]["primary"], "roberta-base")
        self.assertEqual(config["model"]["optional_comparison"], "microsoft/deberta-v3-base")
        self.assertEqual(config["model"]["fused_input_dim"], 2305)
        self.assertEqual(config["project"]["python"], "3.11")
        self.assertEqual(config["project"]["seed"], 42)
        self.assertEqual(config["training"]["primary"]["split"], "stratified_5fold")
        for key in ("mustard", "mustard_pp", "sarc", "semeval2018"):
            self.assertIn(key, config["datasets"])

    def test_readme_documents_cv_and_deployment(self) -> None:
        text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("cv_validation.json", text)
        self.assertIn("independent_test.json", text)
        self.assertIn("is_independent_test: false", text)
        self.assertIn("not an independent test result", text)
        self.assertIn("full-data refit", text)
        self.assertIn("median", text)
        self.assertIn("deployment_epochs", text)
        self.assertIn("ffmpeg", text)
        self.assertIn("roberta-base", text)
        self.assertIn("(768,)", text)
        self.assertIn("(2305,)", text)

    def test_root_files_are_unchanged(self) -> None:
        expected = {
            "README.md": "f5e33f6e2e8763f4bd5c7442d999c12cc0a548e6",
            "hello.py": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
            "SarcasTone/.gitkeep": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
        }
        for rel, digest in expected.items():
            target = REPO_ROOT / rel
            if target.is_file():
                self.assertEqual(_sha1(target), digest, msg=rel)


if __name__ == "__main__":
    unittest.main()

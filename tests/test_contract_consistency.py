from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from shared.constants import (
    AUDIO_HIDDEN,
    CONTEXT_HIDDEN,
    FUSION_LOGITS,
    PROSODY_DIM,
    QUALITY_DIM,
    SARCASM_LABELS,
    TARGET_SAMPLE_RATE,
    TEXT_HIDDEN,
)
from shared.interfaces import INTERFACE_SUMMARY

import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ContractConsistencyTest(unittest.TestCase):

    def test_interface_contract_matches_shared_constants(self) -> None:
        contract = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "interface_contract.yaml").read_text(
                encoding="utf-8"
            )
        )

        expected_shapes = {
            "text_embeddings": ["B", "L", TEXT_HIDDEN],
            "audio_embeddings": ["B", "F", AUDIO_HIDDEN],
            "context_embeddings": ["B", "C", CONTEXT_HIDDEN],
            "prosody": ["B", PROSODY_DIM],
            "quality": ["B", QUALITY_DIM],
            "fusion_logits": ["B", FUSION_LOGITS],
        }
        for name, shape in expected_shapes.items():
            self.assertEqual(contract[name], shape)
            self.assertEqual(INTERFACE_SUMMARY[name], str(shape).replace("'", ""))

        self.assertEqual(contract["sample_rate_hz"], TARGET_SAMPLE_RATE)
        self.assertEqual(int(INTERFACE_SUMMARY["sample_rate_hz"]), TARGET_SAMPLE_RATE)
        self.assertEqual(contract["label_index"], dict(enumerate(SARCASM_LABELS)))
        self.assertEqual(contract["mask_convention"], "true_means_valid")

    def test_project_names_are_consistent(self) -> None:
        metadata = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        project = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(metadata["project"]["name"], "sarcastone")
        self.assertEqual(project["project_name"], "SarcasTone")


# Backward compatibility for standalone pytest functions
def test_interface_contract_matches_shared_constants() -> None:
    ContractConsistencyTest().test_interface_contract_matches_shared_constants()


def test_project_names_are_consistent() -> None:
    ContractConsistencyTest().test_project_names_are_consistent()


if __name__ == "__main__":
    unittest.main()

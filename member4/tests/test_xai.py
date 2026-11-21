"""Unit tests for Member 4 Explainable AI (XAI) modules."""
from __future__ import annotations

import unittest

from member4.backend.xai.acoustic_attribution import compute_acoustic_attributions
from member4.backend.xai.error_analysis import diagnose_multimodal_incongruence
from member4.backend.xai.gating_visualization import compute_gating_weights
from member4.backend.xai.text_attribution import compute_text_attributions
from member4.backend.xai.uncertainty import calibrate_probabilities, compute_entropy, evaluate_uncertainty


class TestXAIModules(unittest.TestCase):

    def test_text_attribution(self) -> None:
        """Verify token saliency scoring."""
        text = "Oh brilliant, that completely fixed everything!"
        tokens = compute_text_attributions(text)
        self.assertGreater(len(tokens), 0)
        top_tokens = [t["token"].lower() for t in tokens[:3]]
        self.assertIn("brilliant", top_tokens)

    def test_acoustic_attribution(self) -> None:
        """Verify prosody feature attributions."""
        prosody = {
            "f0_std": 55.0,
            "pitch_variation_coeff": 0.32,
            "energy_dynamic_range_db": 22.5,
            "speaking_rate_proxy": 2.8,
            "pause_ratio": 0.35,
        }
        attributions = compute_acoustic_attributions(prosody, top_k=5)
        self.assertGreater(len(attributions), 0)
        self.assertEqual(attributions[0]["direction"], "promotes_sarcasm")

    def test_gating_visualization(self) -> None:
        """Verify adaptive confidence gating calculation."""
        gating = compute_gating_weights(
            has_text=True,
            has_audio=True,
            audio_quality={"snr_db_estimate": 30.0, "clipping_ratio": 0.0},
        )
        self.assertAlmostEqual(gating["text"] + gating["audio"], 1.0, places=3)
        self.assertIn("explanation", gating)

    def test_uncertainty_and_calibration(self) -> None:
        """Verify probability calibration and entropy bounds."""
        raw_logits = [0.2, 0.4]
        probs = calibrate_probabilities(raw_logits, temperature=1.25)
        self.assertAlmostEqual(probs["non_sarcastic"] + probs["sarcastic"], 1.0, places=3)

        entropy = compute_entropy(probs)
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, 0.694)  # ln(2) = 0.6931


if __name__ == "__main__":
    unittest.main()

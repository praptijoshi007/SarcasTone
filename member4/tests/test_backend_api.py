"""Unit and integration tests for Member 4 REST API endpoints."""
from __future__ import annotations

import unittest
from pathlib import Path

from member4.backend.main import app
from member4.backend.pipeline import MultimodalPipeline


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_wav = Path(__file__).resolve().parents[2] / "member2_audio" / "tests" / "fixtures" / "test_sample.wav"
        cls.pipeline = MultimodalPipeline()

    def test_pipeline_integration(self) -> None:
        """Verify pipeline execution from API context."""
        res = self.pipeline.predict(
            audio_path=self.fixture_wav if self.fixture_wav.is_file() else None,
            text="Privilege to watch your mind at work.",
            speaker="SHELDON",
        )
        self.assertIn(res.label, ("sarcastic", "non_sarcastic"))
        self.assertGreaterEqual(res.confidence, 0.5)

    def test_app_routes(self) -> None:
        """Verify FastAPI / Starlette app has required routes configured."""
        route_paths = [r.path for r in getattr(app, "routes", [])]
        self.assertIn("/health", route_paths)
        self.assertIn("/model-info", route_paths)


if __name__ == "__main__":
    unittest.main()

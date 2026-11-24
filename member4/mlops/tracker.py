"""MLOps experiment tracking and model artifact versioning module for SarcasTone.

Supports MLflow logging when available, with an automatic local JSON/CSV
tracking fallback.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mlops_tracker")


class ExperimentTracker:
    """Manages experiment runs, metric logging, and artifact tracking."""

    def __init__(
        self,
        experiment_name: str = "SarcasTone_Multimodal",
        tracking_dir: Path | str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir) if tracking_dir else REPO_ROOT / "member4" / "experiments" / "runs"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.active_run: Optional[Dict[str, Any]] = None

        # Check for MLflow
        self.has_mlflow = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.has_mlflow = True
            self.mlflow.set_experiment(experiment_name)
        except Exception:
            pass

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start an experiment tracking session."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{run_name or 'experiment'}_{timestamp}"

        self.active_run = {
            "run_id": run_id,
            "run_name": run_name or run_id,
            "experiment": self.experiment_name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "params": {},
            "metrics": {},
            "tags": tags or {},
            "status": "RUNNING",
        }

        if self.has_mlflow:
            try:
                self.mlflow.start_run(run_name=run_name, tags=tags)
            except Exception as e:
                log.warning("MLflow start_run warning: %s", e)

        return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyper-parameters and architectural configurations."""
        if self.active_run:
            self.active_run["params"].update(params)
        if self.has_mlflow:
            try:
                self.mlflow.log_params(params)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics (accuracy, macro-F1, ROC-AUC, ECE, latency)."""
        if self.active_run:
            self.active_run["metrics"].update(metrics)
        if self.has_mlflow:
            try:
                self.mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

    def end_run(self, status: str = "FINISHED") -> None:
        """Finalize run and persist tracking manifest."""
        if not self.active_run:
            return

        self.active_run["status"] = status
        self.active_run["end_time"] = datetime.now(timezone.utc).isoformat()

        # Save run manifest locally
        run_file = self.tracking_dir / f"{self.active_run['run_id']}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(self.active_run, f, indent=2)

        if self.has_mlflow:
            try:
                self.mlflow.end_run(status=status)
            except Exception:
                pass

        log.info("Persisted run %s to %s", self.active_run["run_id"], run_file)
        self.active_run = None

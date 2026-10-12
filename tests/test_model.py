"""
Tests for model loading, persistence, and evaluation metrics consistency.
"""

from pathlib import Path
import json
import joblib
import pytest
from sklearn.pipeline import Pipeline


def test_saved_model_artifact_exists():
    model_path = Path("models/churn_model.joblib")
    assert model_path.exists(), "models/churn_model.joblib must exist."


def test_saved_model_loads_cleanly():
    model_path = Path("models/churn_model.joblib")
    pipeline = joblib.load(model_path)
    assert isinstance(pipeline, Pipeline), "Loaded model must be a sklearn Pipeline."
    assert "engineer" in pipeline.named_steps
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_saved_metrics_file_validity():
    metrics_path = Path("models/metrics.json")
    assert metrics_path.exists(), "models/metrics.json must exist."

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "best_model" in metrics
    assert "best_metrics" in metrics
    assert "models" in metrics

    best_m = metrics["best_metrics"]
    for key in ["roc_auc", "precision", "recall", "f1", "accuracy"]:
        assert key in best_m
        assert 0.0 <= best_m[key] <= 1.0

    # Honest metric threshold checks
    assert best_m["roc_auc"] >= 0.75, "Tuned model ROC-AUC should meet or exceed 0.75"
    assert best_m["recall"] >= 0.70, "Imbalance handling should achieve high recall (>=0.70)"

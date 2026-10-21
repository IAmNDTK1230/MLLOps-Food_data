import json
import re

import mlflow
from mlflow import MlflowClient

from src.ml.config import MLConfig, TrackingConfig
from src.ml.data import require_trainable
from src.storage.manifest import atomic_json


def register_selected(config: MLConfig, tracking: TrackingConfig, catalog: str) -> dict:
    _, report = require_trainable(config)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", catalog):
        raise ValueError("Use a simple Unity Catalog identifier")
    selected_path = config.models_dir / "selection.json"
    if not selected_path.exists():
        raise ValueError("No trained model; import real reviews and train first")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["dataset_hash"] != report["dataset_hash"]:
        raise ValueError("Selected model belongs to another dataset")
    metrics_path = config.reports_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise ValueError("Evaluate the selected model once before registration")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state_path = config.models_dir / "evaluation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if not state.get("completed") or state.get("training_run_id") != selected["run_id"]:
        raise ValueError("Complete test evaluation and MLflow logging before registration")
    if (
        metrics.get("training_run_id") != selected["run_id"]
        or metrics.get("dataset_hash") != selected["dataset_hash"]
    ):
        raise ValueError("Test report must match the selected training run and dataset")
    if (
        not tracking.tracking_uri.startswith("databricks")
        or tracking.registry_uri != "databricks-uc"
    ):
        raise ValueError(
            "Unity Catalog registration requires Databricks tracking and MLFLOW_REGISTRY_URI=databricks-uc"
        )
    if selected["tracking_uri"] != tracking.tracking_uri:
        raise ValueError(
            "Configure Databricks tracking before training; local model runs cannot be registered directly in UC"
        )
    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_registry_uri(tracking.registry_uri)
    name = f"{catalog}.ml.sentiment_classifier"
    version = mlflow.register_model(selected["model_uri"], name)
    client = MlflowClient()
    selected["test_macro_f1"] = metrics["macro_f1"]
    for key in ("dataset_hash", "run_id", "test_macro_f1", "validation_macro_f1"):
        client.set_model_version_tag(name, version.version, key, str(selected[key]))
    result = {
        "model_name": name,
        "model_version": str(version.version),
        "training_run_id": selected["run_id"],
        "dataset_hash": selected["dataset_hash"],
        "test_macro_f1": selected["test_macro_f1"],
        "promotion": "NONE — registration never changes champion aliases",
    }
    atomic_json(config.models_dir / "registry.json", result)
    return result

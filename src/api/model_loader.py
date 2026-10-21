import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
from dotenv import load_dotenv

from src.utils.hashing import file_checksum


@dataclass
class LoadedModel:
    pipeline: object
    version: str


def load_model() -> LoadedModel:
    load_dotenv()
    uri = os.getenv("MLFLOW_MODEL_URI", "")
    if uri:
        import mlflow
        import mlflow.sklearn

        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        if os.getenv("MLFLOW_REGISTRY_URI"):
            mlflow.set_registry_uri(os.environ["MLFLOW_REGISTRY_URI"])
        version = uri
        # Resolve an alias once so responses identify the actual immutable version.
        if uri.startswith("models:/") and "@" in uri:
            name, alias = uri.removeprefix("models:/").rsplit("@", 1)
            registered = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
            version = str(registered.version)
            uri = f"models:/{name}/{version}"
        elif uri.startswith("models:/"):
            version = uri.rsplit("/", 1)[-1]
        return LoadedModel(mlflow.sklearn.load_model(uri), version)
    if os.getenv("MLFLOW_TRACKING_URI"):
        raise ValueError(
            "Set MLFLOW_MODEL_URI for configured MLflow serving; local fallback is disabled"
        )
    local = os.getenv("LOCAL_MODEL_PATH")
    if not local:
        raise ValueError(
            "No model configured; set MLFLOW_MODEL_URI or explicit LOCAL_MODEL_PATH after real-data training"
        )
    path = Path(local)
    metadata_path = Path(os.getenv("LOCAL_MODEL_METADATA") or str(path.parent / "selection.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_checksum(path) != metadata["model_sha256"]:
        raise ValueError("Local model checksum mismatch")
    return LoadedModel(joblib.load(path), metadata["model_version"])

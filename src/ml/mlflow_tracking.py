from pathlib import Path

import mlflow

from src.ml.config import TrackingConfig


def configure_tracking(config: TrackingConfig):
    if config.tracking_uri.startswith("sqlite:///"):
        Path(config.tracking_uri.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    return mlflow

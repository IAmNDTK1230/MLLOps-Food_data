import json

import joblib
from filelock import FileLock
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.ml.config import LABELS, MLConfig, TrackingConfig
from src.ml.split import load_splits
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def score_predictions(y_true, y_pred) -> dict:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def prediction_artifacts(y_true, y_pred) -> dict:
    return {
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def evaluate_selected(config: MLConfig, tracking: TrackingConfig) -> dict:
    from src.ml.mlflow_tracking import configure_tracking

    frame, splits, report = load_splits(config)
    selected_path = config.models_dir / "selection.json"
    if not selected_path.exists():
        raise ValueError("No selected baseline; run ml-train")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if (
        selected["dataset_hash"] != report["dataset_hash"]
        or selected["split_file_sha256"] != splits["split_file_sha256"]
    ):
        raise ValueError("Model selection and test dataset do not match")
    model_path = config.models_dir / "baseline.joblib"
    if file_checksum(model_path) != selected["model_sha256"]:
        raise ValueError("Selected model bytes changed")
    config.models_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(config.models_dir / ".evaluate.lock"), timeout=1):
        state_path = config.models_dir / "evaluation_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if state.get("training_run_id") == selected["run_id"] and state.get("attempted"):
            existing = config.reports_dir / "test_metrics.json"
            if not existing.exists():
                raise ValueError(
                    "Test evaluation was already attempted. Inspect the interrupted run; no automatic second evaluation is allowed."
                )
            metrics = json.loads(existing.read_text(encoding="utf-8"))
            if metrics["training_run_id"] != selected["run_id"]:
                raise ValueError("Test report belongs to another model selection")
            return metrics
        if selected["tracking_uri"] != tracking.tracking_uri:
            raise ValueError(
                "MLflow tracking URI differs from training; restore the training configuration"
            )
        model = joblib.load(model_path)  # trusted locally generated artifact only
        test = frame[frame["split"] == "test"]
        mlflow = configure_tracking(tracking)
        # Persist before prediction: a retry must never evaluate the same selection twice.
        atomic_json(
            state_path,
            {"training_run_id": selected["run_id"], "attempted": True, "completed": False},
        )
        prediction = model.predict(test["comment"].tolist())
        metrics = {
            **score_predictions(test["sentiment"], prediction),
            "training_run_id": selected["run_id"],
            "dataset_hash": report["dataset_hash"],
            "test_size": len(test),
        }
        artifacts = prediction_artifacts(test["sentiment"], prediction)
        atomic_json(config.reports_dir / "test_metrics.json", metrics)
        atomic_json(
            config.reports_dir / "test_confusion_matrix.json", artifacts["confusion_matrix"]
        )
        atomic_json(
            config.reports_dir / "test_classification_report.json",
            artifacts["classification_report"],
        )
        with mlflow.start_run(run_id=selected["run_id"]):
            mlflow.log_metrics(
                {
                    f"test_{key}": value
                    for key, value in score_predictions(test["sentiment"], prediction).items()
                }
            )
            for name in (
                "test_metrics.json",
                "test_confusion_matrix.json",
                "test_classification_report.json",
            ):
                mlflow.log_artifact(str(config.reports_dir / name), artifact_path="test_evaluation")
        atomic_json(
            state_path,
            {
                "training_run_id": selected["run_id"],
                "attempted": True,
                "completed": True,
                "test_macro_f1": metrics["macro_f1"],
            },
        )
        return metrics

import importlib.metadata
import json

import joblib
import pandas as pd
from filelock import FileLock
from mlflow import sklearn as sklearn_tracking
from mlflow.models import infer_signature

from src.ml.config import MLConfig, TrackingConfig
from src.ml.evaluate import prediction_artifacts, score_predictions
from src.ml.features import build_pipeline
from src.ml.mlflow_tracking import configure_tracking
from src.ml.split import load_splits
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def train_experiments(config: MLConfig, tracking: TrackingConfig) -> dict:
    frame, splits, report = load_splits(config)  # fails before MLflow/model fit if no real data
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(config.models_dir / ".train.lock"), timeout=1):
        mlflow = configure_tracking(tracking)
        rows, candidates = [], []
        for experiment in config.experiments:
            with mlflow.start_run(run_name=experiment.name) as run:
                parameters = {
                    "dataset_source": report["source"],
                    "dataset_hash": report["dataset_hash"],
                    "random_seed": config.random_seed,
                    **{
                        f"{name}_size": splits["counts"][name]
                        for name in ("train", "validation", "test")
                    },
                    "tfidf_max_features": experiment.max_features,
                    "ngram_range": str(experiment.ngram_range),
                    "classifier_C": experiment.c,
                    "classifier_max_iter": experiment.max_iter,
                    "classifier_solver": "lbfgs",
                    "classifier_class_weight": "balanced",
                    "selection_metric": "validation_macro_f1",
                    "label_rules": json.dumps(config.label_rules, sort_keys=True),
                }
                mlflow.log_params(parameters)
                mlflow.set_tags(
                    {"data_kind": "external_public", "source_license": report["license"]}
                )
                model = build_pipeline(experiment, config.random_seed)
                model.fit(train["comment"].tolist(), train["sentiment"].tolist())
                prediction = model.predict(validation["comment"].tolist())
                metrics = score_predictions(validation["sentiment"], prediction)
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
                directory = config.reports_dir / "experiments" / experiment.name
                artifacts = prediction_artifacts(validation["sentiment"], prediction)
                for name, artifact in artifacts.items():
                    atomic_json(directory / f"{name}.json", artifact)
                atomic_json(directory / "label_distribution.json", report["label_distribution"])
                mlflow.log_artifacts(str(directory), artifact_path="validation")
                mlflow.log_artifact(str(config.reports_dir / "ml_data_report.json"))
                # Use training rows for signature; test data is not read for selection.
                example = train["comment"].head(3).tolist()
                info = sklearn_tracking.log_model(
                    model,
                    name="model",
                    signature=infer_signature(example, model.predict(example)),
                    code_paths=["src"],
                    pip_requirements=[
                        f"scikit-learn=={importlib.metadata.version('scikit-learn')}",
                        f"cloudpickle=={importlib.metadata.version('cloudpickle')}",
                    ],
                )
                rows.append(
                    {
                        "experiment": experiment.name,
                        "run_id": run.info.run_id,
                        "model_uri": info.model_uri,
                        **{f"validation_{key}": value for key, value in metrics.items()},
                    }
                )
                candidates.append(model)
        comparison = pd.DataFrame(rows)
        comparison.to_csv(config.reports_dir / "experiment_comparison.csv", index=False)
        # Stable experiment order resolves ties; the test set never selects the winner.
        index = max(range(len(rows)), key=lambda i: rows[i]["validation_macro_f1"])
        winner = rows[index]
        model_path = config.models_dir / "baseline.joblib"
        joblib.dump(candidates[index], model_path)
        selected = {
            **winner,
            "dataset_hash": report["dataset_hash"],
            "split_file_sha256": splits["split_file_sha256"],
            "source": report["source"],
            "model_sha256": file_checksum(model_path),
            "tracking_uri": tracking.tracking_uri,
            "model_version": f"local-{winner['run_id']}",
            "test_evaluation_attempted": False,
            "selection_metric": "validation_macro_f1",
        }
        atomic_json(config.models_dir / "selection.json", selected)
        with mlflow.start_run(run_id=winner["run_id"]):
            mlflow.set_tag("selected_by_validation_macro_f1", "true")
            mlflow.log_artifact(str(config.reports_dir / "experiment_comparison.csv"))
        return selected

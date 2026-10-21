import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from src.baselines.metrics import (
    LABELS,
    class_report,
    comparison_plot,
    confusion_plot,
    measure_predictions,
    scores,
    weight_plot,
)
from src.baselines.rule_based import VietnameseRuleBased, rule_definition
from src.baselines.tfidf_logreg import EXPERIMENTS, build_pipeline, select_validation_trial
from src.food.split import load_splits
from src.ml.config import TrackingConfig
from src.ml.mlflow_tracking import configure_tracking
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def load_protocol(config):
    frame, splits, preparation = load_splits(config)  # Read-only; never split/download.
    if (
        len(frame) != 5869
        or splits["counts"] != {"train": 4140, "validation": 842, "test": 887}
        or splits["random_seed"] != 42
    ):
        raise ValueError(
            "Baseline protocol requires the verified 5869-row canonical split (4140/842/887, seed 42)"
        )
    if set(frame.sentiment) != set(LABELS) or not frame.dataset_name.eq("vn_msasar_v3").all():
        raise ValueError("Baseline labels/dataset do not match the verified food contract")
    source_hashes = {
        name: file_checksum(Path(__file__).parent / name)
        for name in ("runner.py", "rule_based.py", "tfidf_logreg.py", "metrics.py")
    }
    protocol = {
        "dataset_hash": preparation["processed_sha256"],
        "split_hash": splits["split_file_sha256"],
        "random_seed": 42,
        "input_features": ["text"],
        "target": "sentiment",
        "labels": LABELS,
        "counts": splits["counts"],
        "source_hashes": source_hashes,
        "experiments": EXPERIMENTS,
        "rule_definition": rule_definition(),
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["protocol_hash"] = digest
    root = config.reports_dir / "baselines"
    root.mkdir(parents=True, exist_ok=True)
    return frame, protocol, root


def environment(protocol, root):
    path = root / "environment.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError(
                "Experiment protocol changed; existing evidence must be reviewed before another run"
            )
        return result
    cpu, ram = platform.processor(), None
    try:
        import psutil

        ram = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    result = {
        **protocol,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": cpu or "unavailable",
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": ram,
        "execution_timestamp": datetime.now(UTC).isoformat(),
        "package_versions": {
            name: version(name) for name in ("scikit-learn", "numpy", "pandas", "mlflow")
        },
        "latency_note": "Sequential batch size 1 on this machine; not a production benchmark",
    }
    atomic_json(path, result)
    return result


def cached(path, protocol):
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError(
            "Baseline evidence belongs to a different protocol; no automatic retuning after held-out evaluation"
        )
    return result


def run_rule(config):
    frame, protocol, root = load_protocol(config)
    environment(protocol, root)
    path = root / "rule_based_validation_metrics.json"
    existing = cached(path, protocol)
    if existing:
        return existing
    if (root / "evaluation_state.json").exists():
        raise ValueError("Test evaluation already attempted; cannot change the rule baseline")
    validation = frame.loc[frame.split == "validation"]
    predictions, latency = measure_predictions(
        VietnameseRuleBased(), validation.text, frame.loc[frame.split == "train"].text
    )
    result = {
        **scores(validation.sentiment, predictions),
        **latency,
        **protocol,
        "model": "Rule-Based",
        "evaluation_split": "validation",
        "model_size_mb": 0,
        "size_note": "No serialized learned model; auditable rule definition stored separately",
    }
    atomic_json(root / "rule_definition.json", rule_definition())
    atomic_json(path, result)
    return result


def tracking_config():
    config = TrackingConfig.from_env()
    if not os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        config = config.model_copy(update={"experiment_name": "food-restaurant-baselines"})
    return config


def run_tfidf(config):
    frame, protocol, root = load_protocol(config)
    env = environment(protocol, root)
    model_path = Path("models/tfidf_logreg.joblib")
    selection_path = root / "tfidf_selection.json"
    existing = cached(selection_path, protocol)
    if existing:
        if not model_path.exists() or file_checksum(model_path) != existing["model_sha256"]:
            raise ValueError(
                "Selected model missing/changed; no automatic refit of evaluated model"
            )
        return existing
    if (root / "evaluation_state.json").exists():
        raise ValueError("Held-out evaluation already attempted; model selection is frozen")
    train, validation = (frame.loc[frame.split == name] for name in ("train", "validation"))
    mlflow = configure_tracking(tracking_config())
    from mlflow import sklearn as sklearn_tracking
    from mlflow.models import infer_signature

    experiments, fitted = [], []
    for experiment in EXPERIMENTS:
        model = build_pipeline(experiment, seed=protocol["random_seed"])
        with mlflow.start_run(run_name=experiment["name"]) as run:
            model.fit(train.text.tolist(), train.sentiment.tolist())
            prediction, latency = measure_predictions(model, validation.text, train.text)
            metrics = scores(validation.sentiment, prediction)
            params = {
                "dataset_hash": protocol["dataset_hash"],
                "split_hash": protocol["split_hash"],
                "protocol_hash": protocol["protocol_hash"],
                "random_seed": 42,
                "ngram_range": str(experiment["ngram_range"]),
                "C": experiment["c"],
                "class_weight": "balanced",
                "max_features": 20000,
                "min_df": 2,
                "max_iter": 1500,
                "features": "text_only",
                "normalization": "NFC + whitespace; case/accents preserved",
                "solver": "lbfgs",
                "sublinear_tf": True,
            }
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    f"validation_{key}": value
                    for key, value in (
                        metrics
                        | {key: latency[key] for key in ("mean_latency_ms", "p95_latency_ms")}
                    ).items()
                }
            )
            mlflow.log_dict(env, "environment.json")
            report = class_report(validation.sentiment, prediction)
            mlflow.log_dict(report.to_dict("records"), "validation_classification_report.json")
            artifact = root / "experiments" / experiment["name"]
            artifact.mkdir(parents=True, exist_ok=True)
            confusion_plot(
                validation.sentiment,
                prediction,
                artifact / "validation_confusion_matrix.png",
                experiment["name"] + " — validation",
            )
            mlflow.log_artifact(str(artifact / "validation_confusion_matrix.png"))
            sklearn_tracking.log_model(
                model,
                name="model",
                signature=infer_signature(
                    train.text.head(3).tolist(), model.predict(train.text.head(3).tolist())
                ),
                code_paths=["src"],
                pip_requirements=[
                    f"scikit-learn=={version('scikit-learn')}",
                    f"cloudpickle=={version('cloudpickle')}",
                ],
            )
            experiments.append(
                {
                    "experiment": experiment["name"],
                    "ngram_range": str(experiment["ngram_range"]),
                    "C": experiment["c"],
                    "class_weight": "balanced",
                    **metrics,
                    **{key: latency[key] for key in ("mean_latency_ms", "p95_latency_ms")},
                    "run_id": run.info.run_id,
                }
            )
            fitted.append((model, experiment, metrics, latency, run.info.run_id, prediction))
    # Fixed candidate order resolves ties; test performance is never used here.
    selected = select_validation_trial(experiments)
    model, experiment, metrics, latency, run_id, balanced_prediction = fitted[selected]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    pd.DataFrame(experiments).to_csv(root / "tfidf_experiments.csv", index=False)
    # Controlled imbalance ablation on VALIDATION only, with selected hyperparameters fixed.
    unweighted = build_pipeline(experiment, class_weight=None)
    with mlflow.start_run(run_name=experiment["name"] + "_unweighted_ablation"):
        unweighted.fit(train.text.tolist(), train.sentiment.tolist())
        prediction = unweighted.predict(validation.text.tolist())
        mlflow.log_params(
            {
                "dataset_hash": protocol["dataset_hash"],
                "split_hash": protocol["split_hash"],
                "random_seed": 42,
                "C": experiment["c"],
                "ngram_range": str(experiment["ngram_range"]),
                "class_weight": "None",
                "purpose": "validation_only_imbalance_ablation",
            }
        )
        mlflow.log_metrics(
            {
                f"validation_{key}": value
                for key, value in scores(validation.sentiment, prediction).items()
            }
        )
        weighted_rows = []
        for weight, preds in (("None", prediction), ("balanced", balanced_prediction)):
            report = class_report(validation.sentiment, preds)
            for record in report.loc[report.label.isin(LABELS)].to_dict("records"):
                weighted_rows.append(
                    {
                        "class_weight": weight,
                        "evaluation_split": "validation",
                        "experiment": experiment["name"],
                        **record,
                    }
                )
        weights = pd.DataFrame(weighted_rows)
        weights.to_csv(root / "class_weight_comparison.csv", index=False)
        weight_plot(weights, root / "class_weight_comparison.png")
        mlflow.log_artifact(str(root / "class_weight_comparison.csv"))
        mlflow.log_artifact(str(root / "class_weight_comparison.png"))
    result = {
        **protocol,
        "selected_experiment": experiment,
        "class_weight": "balanced",
        "selection_metric": "validation_macro_f1",
        "validation_metrics": metrics | latency,
        "run_id": run_id,
        "model_sha256": file_checksum(model_path),
        "model_size_mb": model_path.stat().st_size / 1_000_000,
        "test_evaluated": False,
        "ablation": "validation only; not used to replace balanced baseline",
    }
    atomic_json(selection_path, result)
    atomic_json(root / "tfidf_validation_metrics.json", metrics | latency | protocol)
    return result


def failure_analysis(frame, rule, tfidf, root):
    rows = (
        frame[["review_id", "text", "aspect", "sentiment"]]
        .rename(columns={"sentiment": "true_label"})
        .copy()
    )
    rows["rule_prediction"], rows["tfidf_prediction"] = rule, tfidf
    errors = (rows.true_label != rows.rule_prediction) | (rows.true_label != rows.tfidf_prediction)
    failures = rows.loc[errors].copy()
    for name, column in (("rule", "rule_prediction"), ("tfidf", "tfidf_prediction")):
        failures[name + "_error_type"] = [
            "correct"
            if truth == pred
            else "neutral_as_positive"
            if truth == "neutral" and pred == "positive"
            else "neutral_as_negative"
            if truth == "neutral" and pred == "negative"
            else "false_positive_positive_class"
            if pred == "positive"
            else "false_negative_positive_class"
            if truth == "positive"
            else "negative_as_neutral"
            for truth, pred in zip(failures.true_label, failures[column], strict=True)
        ]
    failures.to_csv(root / "failure_cases.csv", index=False, encoding="utf-8")
    by_aspect = []
    for name, column in (
        ("Rule-Based", "rule_prediction"),
        ("TF-IDF + Logistic Regression", "tfidf_prediction"),
    ):
        for aspect, group in rows.groupby("aspect"):
            error_count = int((group.true_label != group[column]).sum())
            by_aspect.append(
                {
                    "model": name,
                    "aspect": aspect,
                    "samples": len(group),
                    "errors": error_count,
                    "error_rate": error_count / len(group),
                    **scores(group.true_label, group[column]),
                }
            )
    pd.DataFrame(by_aspect).to_csv(root / "error_by_aspect.csv", index=False)
    return {
        name: failures[column].value_counts().to_dict()
        for name, column in (("rule", "rule_error_type"), ("tfidf", "tfidf_error_type"))
    }


def evaluate(config):
    frame, protocol, root = load_protocol(config)
    selection = cached(root / "tfidf_selection.json", protocol)
    rule_validation = cached(root / "rule_based_validation_metrics.json", protocol)
    if not selection or not rule_validation:
        raise ValueError("Run baseline-rule and baseline-tfidf before baseline-evaluate")
    state_path = root / "evaluation_state.json"
    if state_path.exists():
        state = cached(state_path, protocol)
        if state["status"] != "completed":
            raise ValueError(
                "Test evaluation already attempted but incomplete; inspect saved state, do not silently reevaluate"
            )
        for name, checksum in state["artifacts"].items():
            if not Path(name).exists() or file_checksum(Path(name)) != checksum:
                raise ValueError(
                    "Saved test evidence missing/changed; no automatic second evaluation"
                )
        return pd.read_csv(root / "baseline_comparison.csv").to_dict("records")
    model_path = Path("models/tfidf_logreg.joblib")
    if file_checksum(model_path) != selection["model_sha256"]:
        raise ValueError("Selected model checksum changed; refusing test evaluation")
    model = joblib.load(model_path)
    mlflow = configure_tracking(tracking_config())
    selected_run = mlflow.get_run(selection["run_id"])
    if selected_run.data.params.get("split_hash") != protocol["split_hash"]:
        raise ValueError("MLflow selected run does not match this split/backend")
    # Durable marker BEFORE any held-out prediction, including rule baseline.
    atomic_json(
        state_path,
        {
            **protocol,
            "status": "attempted",
            "selected_run_id": selection["run_id"],
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    test, validation = (frame.loc[frame.split == name] for name in ("test", "validation"))
    rule_prediction, rule_latency = measure_predictions(
        VietnameseRuleBased(), test.text, validation.text
    )
    tfidf_prediction, tfidf_latency = measure_predictions(model, test.text, validation.text)
    comparison = []
    for key, name, prediction, latency, size in (
        ("rule_based", "Rule-Based", rule_prediction, rule_latency, 0),
        (
            "tfidf",
            "TF-IDF + Logistic Regression",
            tfidf_prediction,
            tfidf_latency,
            selection["model_size_mb"],
        ),
    ):
        metrics = scores(test.sentiment, prediction) | latency | {"model_size_mb": size}
        matrix = confusion_plot(
            test.sentiment, prediction, root / f"{key}_confusion_matrix.png", name + " — test"
        )
        report = class_report(test.sentiment, prediction)
        report.to_csv(root / f"{key}_classification_report.csv", index=False)
        atomic_json(
            root / f"{key}_test_metrics.json",
            metrics
            | protocol
            | {"model": name, "evaluation_split": "test", "confusion_matrix": matrix},
        )
        comparison.append(
            {
                "model": name,
                **{
                    field: metrics[field]
                    for field in (
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "weighted_f1",
                        "mean_latency_ms",
                        "p95_latency_ms",
                        "model_size_mb",
                    )
                },
            }
        )
        if key == "tfidf":
            with mlflow.start_run(run_id=selection["run_id"]):
                mlflow.log_metrics(
                    {
                        f"test_{field}": value
                        for field, value in metrics.items()
                        if isinstance(value, (float, int))
                    }
                )
                mlflow.log_artifact(str(root / f"{key}_confusion_matrix.png"))
                mlflow.log_artifact(str(root / f"{key}_classification_report.csv"))
                mlflow.set_tag("held_out_evaluation", "once_after_validation_selection")
    table = pd.DataFrame(comparison)
    table.to_csv(root / "baseline_comparison.csv", index=False)
    comparison_plot(table, root / "baseline_comparison.png")
    failure_counts = failure_analysis(test, rule_prediction, tfidf_prediction, root)
    preparation_evidence = json.loads(
        (config.reports_dir / "task2_slide_evidence.json").read_text(encoding="utf-8")
    )
    evidence = {
        "data_preparation": preparation_evidence,
        "slide_7": environment(protocol, root),
        "slide_8": {"test_comparison": comparison, "selected_tfidf": selection},
        "slide_9": {
            "macro_f1_difference_tfidf_minus_rule": comparison[1]["macro_f1"]
            - comparison[0]["macro_f1"],
            "error_counts": failure_counts,
            "error_by_aspect": "baselines/error_by_aspect.csv",
            "limitations": [
                "Text-only target can differ across aspect annotations of the same sentence",
                "Rule baseline has only local negation and no sarcasm model",
                "No production, Qwen, or external-evaluation claim",
            ],
        },
        "slide_6_update": {
            "comparison_split": "validation",
            "class_weight_comparison": pd.read_csv(root / "class_weight_comparison.csv").to_dict(
                "records"
            ),
            "smote_used": False,
        },
        "proposed_v1_next_step": "Define a frozen text-only Qwen inference protocol using these exact splits and output labels; tune prompts on train/validation only; no Qwen execution yet",
    }
    atomic_json(config.reports_dir / "task2_baseline_evidence.json", evidence)
    output_names = [
        "baseline_comparison.csv",
        "baseline_comparison.png",
        "failure_cases.csv",
        "error_by_aspect.csv",
    ]
    output_names += [
        f"{key}_{suffix}"
        for key in ("rule_based", "tfidf")
        for suffix in ("test_metrics.json", "confusion_matrix.png", "classification_report.csv")
    ]
    artifacts = [root / name for name in output_names] + [
        config.reports_dir / "task2_baseline_evidence.json",
        model_path,
    ]
    atomic_json(
        state_path,
        {
            **protocol,
            "status": "completed",
            "selected_run_id": selection["run_id"],
            "timestamp": datetime.now(UTC).isoformat(),
            "artifacts": {str(path): file_checksum(path) for path in artifacts},
        },
    )
    return comparison


def run_all(config):
    run_rule(config)
    run_tfidf(config)
    return evaluate(config)

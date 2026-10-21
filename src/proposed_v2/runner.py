import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baselines.metrics import LABELS, confusion_plot
from src.baselines.runner import load_protocol, tracking_config
from src.ml.mlflow_tracking import configure_tracking
from src.proposed_v1.runner import digest, predict_partition, score_predictions, usage
from src.proposed_v1.runtime import LocalQwen
from src.proposed_v2.examples import freeze_examples_and_prompts
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


class TrackedRuntime(LocalQwen):
    """Same inference implementation/settings as V1; durable V2 telemetry on exit."""

    def __init__(self, config, provenance, root):
        super().__init__(config, provenance)
        self.root = root

    def snapshot(self):
        return {
            "sampled_peak_server_rss_bytes": self.peak_rss,
            "mean_server_cpu_percent_one_core_100": float(np.mean(self.cpu_samples))
            if self.cpu_samples
            else None,
            "sampled_peak_gpu_memory_mib_device_wide": max(
                [r[1] for r in self.gpu_samples], default=None
            ),
            "mean_gpu_utilization_percent_device_wide": float(
                np.mean([r[0] for r in self.gpu_samples])
            )
            if self.gpu_samples
            else None,
            "sampling_interval_seconds": 1,
            "scope": "model session including train warmups; device-wide GPU values may include other applications",
        }

    def __exit__(self, *args):
        try:
            super().__exit__(*args)
        finally:
            atomic_json(self.root / "resource_session.json", self.snapshot())


def choose_candidate(rows):
    # Frozen before experiments: macro F1 first, precision diagnoses over-neutral bias.
    return max(
        rows,
        key=lambda row: (
            row["macro_f1"],
            row["neutral_precision"],
            row["json_validity_rate"],
            -row["mean_latency_ms"],
        ),
    )


def historical_validation(config, validation):
    """Read V1 validation outputs only; no V1 test material during development."""
    path = config.reports_dir / "proposed_v1/v1_c_validation_predictions.parquet"
    records = pd.read_parquet(path)
    column = "sample_index" if "sample_index" in records else "index"
    if records[column].duplicated().any() or set(records[column]) != set(validation.index):
        raise ValueError("V1 validation membership mismatch")
    records = records.set_index(column).loc[validation.index]
    if records.text_sha256.tolist() != [digest(text) for text in validation.text]:
        raise ValueError("V1 validation text checksum mismatch")
    metrics, report = score_predictions(validation.sentiment, records.prediction)
    recorded = json.loads(
        (config.reports_dir / "proposed_v1/v1_c_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if any(abs(recorded[key] - value) > 1e-12 for key, value in metrics.items()):
        raise ValueError("Historical V1 validation scores changed")
    return records, metrics, report


def frozen_inputs(config):
    """Read-only audit: canonical data, historical outputs, model, Phase 1 source."""
    paths = [
        config.data_dir / "splits/reviews.parquet",
        config.data_dir / "splits/metadata.json",
        config.data_dir / "processed/reviews.parquet",
        Path("models/tfidf_logreg.joblib"),
        Path("databricks.yml"),
    ]
    paths += [
        p
        for folder in (
            config.reports_dir / "baselines",
            config.reports_dir / "proposed_v1",
            Path("databricks"),
        )
        for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    return {str(path): file_checksum(path) for path in paths}


def assert_frozen(audit):
    if any(
        not Path(name).exists() or file_checksum(Path(name)) != sha for name, sha in audit.items()
    ):
        raise ValueError("A protected dataset/baseline/V1/Phase 1 input changed")


def join_predictions(rows, records):
    """Explicit stable index name; no reliance on pandas preserving .loc index names."""
    metadata = rows.copy()
    metadata.insert(0, "sample_index", rows.index.to_numpy())
    metadata = metadata.rename(columns={"sentiment": "true_label"})
    return metadata.merge(records, on="sample_index", how="inner", validate="one_to_one")


def verified_records(path, rows, protocol_hash):
    records = pd.DataFrame(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    )
    if (
        len(records) != len(rows)
        or records.sample_index.duplicated().any()
        or set(records.sample_index) != set(rows.index)
        or not records.protocol_hash.eq(protocol_hash).all()
    ):
        raise ValueError("Saved checkpoint membership/protocol mismatch")
    records = (
        records.set_index("sample_index").loc[rows.index].rename_axis("sample_index").reset_index()
    )
    if records.text_sha256.tolist() != [digest(text) for text in rows.text]:
        raise ValueError("Saved checkpoint text mismatch")
    return records


def finish(config, frame, protocol, root, candidates, selected, mlflow):
    """Reporting only: no runtime object, no model prediction, no prompt selection."""
    state_path = root / "test_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("pending_sample") is not None:
        raise ValueError("Unresolved in-flight test request; cannot finish or predict again")
    test = frame.loc[frame.split == "test"]
    records = verified_records(root / "test_checkpoint.jsonl", test, protocol["protocol_hash"])
    measurement = json.loads((root / "test_measurements.json").read_text(encoding="utf-8"))
    metrics, report = score_predictions(test.sentiment, records.prediction)
    result = {
        **metrics,
        **usage(records, measurement["duration_seconds"]),
        "model_size_mb": protocol["model_provenance"]["model_size_bytes"] / 1_000_000,
        "selected_prompt": selected["prompt"],
        "dataset_hash": protocol["dataset_hash"],
        "split_hash": protocol["split_hash"],
        "test_hygiene": protocol["test_hygiene"],
        "sample_count": len(test),
        "evaluation_split": "test",
    }
    atomic_json(root / "test_metrics.json", result)
    predictions = join_predictions(test, records)
    predictions.to_parquet(root / "predictions.parquet", index=False)
    pd.DataFrame(
        [{"label": label, **report[label]} for label in (*LABELS, "macro avg", "weighted avg")]
    ).to_csv(root / "classification_report.csv", index=False)
    if records.json_valid.all():
        confusion_plot(
            test.sentiment,
            records.prediction,
            root / "confusion_matrix.png",
            "Qwen V2-A — exploratory reused test",
        )
    else:
        from src.proposed_v2.analysis import abstention_plot

        abstention_plot(test.sentiment, records.prediction, root / "confusion_matrix.png")
    from src.proposed_v2.analysis import final_analysis

    final_analysis(config, frame, predictions, result, candidates, selected, protocol, root)
    with mlflow.start_run(run_id=selected["run_id"]):
        mlflow.log_metrics(
            {
                "test_" + key: value
                for key, value in result.items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.set_tag("test_hygiene", "exploratory_reuse_v1_test_previously_inspected")
        for name in (
            "test_metrics.json",
            "confusion_matrix.png",
            "classification_report.csv",
            "comparison.csv",
            "failure_analysis.json",
            "resource_report.json",
        ):
            mlflow.log_artifact(str(root / name))
    assert_frozen(protocol["protected_inputs"])
    paths = [path for path in root.rglob("*") if path.is_file() and path != state_path]
    paths.append(config.reports_dir / "task2_proposed_v2_evidence.json")
    atomic_json(
        state_path,
        {
            "status": "completed",
            "protocol_hash": protocol["protocol_hash"],
            "pending_sample": None,
            "completed": datetime.now(UTC).isoformat(),
            "test_hygiene": protocol["test_hygiene"],
            "artifacts": {str(path): file_checksum(path) for path in paths},
        },
    )
    return result


def run(config, settings_path="qwen_v2.yaml", prepare_only=False):
    frame, baseline, _ = load_protocol(config)  # Read-only checks, no baseline inference.
    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    runtime_config = yaml.safe_load(
        Path(settings["base_runtime_config"]).read_text(encoding="utf-8")
    )
    root = config.reports_dir / "proposed_v2"
    root.mkdir(parents=True, exist_ok=True)
    prompts, manifest = freeze_examples_and_prompts(
        frame.loc[frame.split == "train"], settings, root
    )
    provenance = json.loads(
        (Path(runtime_config["model_dir"]) / "provenance.json").read_text(encoding="utf-8")
    )
    if (
        file_checksum(Path(runtime_config["model_dir"]) / runtime_config["model_file"])
        != provenance["model_sha256"]
        or file_checksum(Path(provenance["server_executable"])) != provenance["server_sha256"]
    ):
        raise ValueError("Pinned V1 model/runtime checksum changed")
    # Verify settings are identical to V1 without reading any V1 TEST artifacts.
    v1_validation = json.loads(
        (config.reports_dir / "proposed_v1/v1_c_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    v1_runtime = json.loads(
        (config.reports_dir / "proposed_v1/selection.json").read_text(encoding="utf-8")
    )["frozen_protocol"]["config"]
    if runtime_config != v1_runtime:
        raise ValueError("V2-A must keep all V1 model/runtime/generation settings unchanged")
    if runtime_config["temperature"] != 0 or runtime_config["seed"] != 42:
        raise ValueError("V2 protocol requires temperature 0 and seed 42")
    protocol = {
        "dataset_hash": baseline["dataset_hash"],
        "split_hash": baseline["split_hash"],
        "runtime_config": runtime_config,
        "model_provenance": provenance,
        "settings": settings,
        "model_hash": provenance["model_sha256"],
        "prompt_hashes": {name: p["sha256"] for name, p in prompts.items()},
        "train_manifest_hash": digest(manifest),
        "historical_v1_validation_macro_f1": v1_validation["macro_f1"],
        "test_hygiene": settings["test_hygiene"],
        "input_features": ["text"],
        "protected_inputs": frozen_inputs(config),
        "source_hashes": {
            name: file_checksum(Path(__file__).parent / name)
            for name in ("examples.py", "runner.py", "analysis.py")
        },
        "shared_runtime_hash": file_checksum(Path("src/proposed_v1/runtime.py")),
        "shared_checkpoint_hash": file_checksum(Path("src/proposed_v1/runner.py")),
    }
    protocol["protocol_hash"] = digest(protocol)
    frozen_path = root / "protocol.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Frozen V2 experiment changed; refusing automatic post-test retuning")
    atomic_json(frozen_path, protocol)
    if prepare_only:
        return {
            "status": "prompts_frozen_no_inference",
            "prompts": protocol["prompt_hashes"],
            "example_categories": [row["category"] for row in manifest["examples"]],
        }
    state_path = root / "test_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("V2 test protocol differs")
        if state["status"] == "completed":
            for name, sha in state["artifacts"].items():
                if file_checksum(Path(name)) != sha:
                    raise ValueError("Completed V2 artifact changed; no new test inference")
            return json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
    mlflow = configure_tracking(tracking_config())
    validation = frame.loc[frame.split == "validation"]
    candidates = []
    # Complete checkpoints can finish reports without starting the model again.
    if (root / "test_measurements.json").exists():
        candidates = [
            json.loads((root / f"{name}_validation_metrics.json").read_text(encoding="utf-8"))
            for name in settings["candidate_names"]
        ]
        selected = json.loads((root / "selection.json").read_text(encoding="utf-8"))
        return finish(config, frame, protocol, root, candidates, selected, mlflow)
    with TrackedRuntime(runtime_config, provenance, root) as client:
        for name, prompt in prompts.items():
            metric_path = root / f"{name}_validation_metrics.json"
            if metric_path.exists():
                result = json.loads(metric_path.read_text(encoding="utf-8"))
                if result["protocol_hash"] != protocol["protocol_hash"]:
                    raise ValueError("V2 validation checkpoint protocol differs")
                candidates.append(result)
                continue
            with mlflow.start_run(run_name="qwen_" + name) as tracking:
                warmup = client.predict(
                    prompt["text"], frame.loc[frame.split == "train"].iloc[0].text
                )
                records, duration = predict_partition(
                    client,
                    prompt["text"],
                    validation,
                    root / f"{name}_validation_checkpoint.jsonl",
                    protocol["protocol_hash"],
                )
                metrics, report = score_predictions(validation.sentiment, records.prediction)
                result = {
                    "prompt": name,
                    "prompt_hash": prompt["sha256"],
                    "example_count": prompt["example_count"],
                    "protocol_hash": protocol["protocol_hash"],
                    "run_id": tracking.info.run_id,
                    **metrics,
                    **usage(records, duration),
                    "neutral_prediction_count": int(records.prediction.eq("neutral").sum()),
                    "warmup_usage": warmup,
                    "resource_snapshot": client.snapshot(),
                }
                atomic_json(metric_path, result)
                pd.DataFrame([{"label": label, **report[label]} for label in LABELS]).to_csv(
                    root / f"{name}_validation_classification_report.csv", index=False
                )
                records.to_parquet(root / f"{name}_validation_predictions.parquet", index=False)
                mlflow.log_params(
                    {
                        "model": runtime_config["model_repo"],
                        "model_hash": provenance["model_sha256"],
                        "revision": runtime_config["model_revision"],
                        "runtime": runtime_config["runtime_tag"],
                        "quantization": runtime_config["quantization"],
                        "prompt": name,
                        "prompt_hash": prompt["sha256"],
                        "dataset_hash": protocol["dataset_hash"],
                        "split_hash": protocol["split_hash"],
                        "temperature": 0,
                        "seed": 42,
                        "max_new_tokens": runtime_config["max_new_tokens"],
                        "context_size": runtime_config["context_size"],
                        "example_count": prompt["example_count"],
                        "features": "text_only",
                    }
                )
                mlflow.log_metrics(
                    {
                        "validation_" + key: value
                        for key, value in result.items()
                        if isinstance(value, (int, float))
                    }
                )
                for artifact in (
                    Path(settings["prompt_dir"]) / f"{name}.txt",
                    root / "train_example_manifest.json",
                    root / f"{name}_validation_classification_report.csv",
                ):
                    mlflow.log_artifact(str(artifact))
                candidates.append(result)
        table = [
            {
                key: value
                for key, value in row.items()
                if key not in {"warmup_usage", "resource_snapshot"}
            }
            for row in candidates
        ]
        pd.DataFrame(table).to_csv(root / "prompt_comparison.csv", index=False)
        from src.proposed_v2.analysis import validation_analysis

        validation_analysis(config, frame, candidates, root)
        selected = choose_candidate(candidates)
        selection = {
            key: selected[key]
            for key in (
                "prompt",
                "prompt_hash",
                "run_id",
                "macro_f1",
                "neutral_precision",
                "neutral_recall",
                "neutral_f1",
            )
        }
        selection.update(
            protocol_hash=protocol["protocol_hash"],
            model_hash=protocol["model_hash"],
            dataset_hash=protocol["dataset_hash"],
            split_hash=protocol["split_hash"],
            generation_settings=runtime_config,
            selection_criteria="validation macro F1, neutral precision, JSON validity, lower latency",
            test_hygiene=protocol["test_hygiene"],
        )
        if (root / "selection.json").exists() and json.loads(
            (root / "selection.json").read_text(encoding="utf-8")
        ) != selection:
            raise ValueError("Frozen V2 selection changed")
        atomic_json(root / "selection.json", selection)
        mlflow.get_run(selected["run_id"])
        if not state_path.exists():
            atomic_json(
                state_path,
                {
                    "status": "in_progress",
                    "protocol_hash": protocol["protocol_hash"],
                    "pending_sample": None,
                    "started": datetime.now(UTC).isoformat(),
                },
            )
        # First access to final TEST features/predictions is after immutable selection.
        test = frame.loc[frame.split == "test"]
        _, duration = predict_partition(
            client,
            prompts[selected["prompt"]]["text"],
            test,
            root / "test_checkpoint.jsonl",
            protocol["protocol_hash"],
            state_path,
        )
        atomic_json(
            root / "test_measurements.json",
            {"duration_seconds": duration, "resource_snapshot": client.snapshot()},
        )
    return finish(config, frame, protocol, root, candidates, selection, mlflow)

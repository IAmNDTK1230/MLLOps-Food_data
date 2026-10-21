"""Recover saved predictions without calling any classifier."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from src.baselines.metrics import LABELS, confusion_plot
from src.baselines.runner import load_protocol, tracking_config
from src.ml.mlflow_tracking import configure_tracking
from src.proposed_v1.analysis import analyze
from src.proposed_v1.runner import digest, score_predictions
from src.proposed_v1.runtime import LocalQwen
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def complete_saved(config, config_path="qwen_v1.yaml", probe_memory=False):
    frame, baseline, _ = load_protocol(config)
    root = config.reports_dir / "proposed_v1"
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    saved = selection["frozen_protocol"]
    qwen = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (
        saved["config"] != qwen
        or saved["dataset_hash"] != baseline["dataset_hash"]
        or saved["split_hash"] != baseline["split_hash"]
    ):
        raise ValueError("Frozen Qwen model/settings/data changed")
    for prompt, sha in saved["prompt_manifest"]["prompts"].items():
        if file_checksum(Path(qwen["prompt_dir"]) / f"{prompt}.txt") != sha:
            raise ValueError("Frozen prompt changed")
    provenance = saved["provenance"]
    if file_checksum(Path(qwen["model_dir"]) / qwen["model_file"]) != provenance["model_sha256"]:
        raise ValueError("Frozen model changed")
    state_path = root / "test_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("pending_sample") is not None:
        raise ValueError("Unknown in-flight test request; cannot recover reporting")
    test = frame.loc[frame.split == "test"]
    records = pd.DataFrame(
        [
            json.loads(line)
            for line in (root / "test_checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    )
    if (
        len(records) != len(test)
        or records.sample_index.duplicated().any()
        or set(records.sample_index) != set(test.index)
        or not records.protocol_hash.eq(selection["protocol_hash"]).all()
    ):
        raise ValueError("Complete unique frozen test responses required; no new inference allowed")
    records = (
        records.set_index("sample_index").loc[test.index].rename_axis("sample_index").reset_index()
    )
    if records.text_sha256.tolist() != [digest(text) for text in test.text]:
        raise ValueError("Checkpoint text changed")
    metrics, report = score_predictions(test.sentiment, records.prediction)
    result = json.loads((root / "qwen_test_metrics.json").read_text(encoding="utf-8"))
    if any(abs(result[key] - value) > 1e-12 for key, value in metrics.items()):
        raise ValueError("Saved metrics do not match checkpoint predictions")
    predictions = (
        test.reset_index()
        .rename(columns={"index": "sample_index", "sentiment": "true_label"})
        .merge(records, on="sample_index", validate="one_to_one")
    )
    predictions.to_parquet(root / "qwen_predictions.parquet", index=False)
    pd.DataFrame(
        [{"label": label, **report[label]} for label in (*LABELS, "macro avg", "weighted avg")]
    ).to_csv(root / "qwen_classification_report.csv", index=False)
    if not records.json_valid.all():
        raise ValueError(
            "Recovery requires explicit abstention plotting; no invalid output silently mapped"
        )
    confusion_plot(
        test.sentiment,
        records.prediction,
        root / "qwen_confusion_matrix.png",
        "Qwen Proposed V1 — saved test outputs",
    )
    candidates = [
        json.loads((root / f"{name}_validation_metrics.json").read_text(encoding="utf-8"))
        for name in ("v1_a", "v1_b", "v1_c")
    ]
    capacity = None
    if probe_memory:
        with LocalQwen(qwen, provenance) as client:
            time.sleep(5)  # Loaded-model idle measurement; ZERO predict calls.
            capacity = {
                "scope": "separate loaded-model idle probe, not evaluation peak",
                "sampled_server_rss_bytes": client.peak_rss,
                "sampled_gpu_memory_mib_device_wide": max(
                    [r[1] for r in client.gpu_samples], default=None
                ),
                "prediction_calls": 0,
            }
    resources = {
        "test": result,
        "validation": candidates,
        "total_evaluation_input_tokens": result["input_tokens"]
        + sum(c["input_tokens"] for c in candidates),
        "total_evaluation_output_tokens": result["output_tokens"]
        + sum(c["output_tokens"] for c in candidates),
        "warmup_input_tokens": sum(c["warmup_usage"]["input_tokens"] for c in candidates),
        "warmup_output_tokens": sum(c["warmup_usage"]["output_tokens"] for c in candidates),
        "sampled_peak_server_rss_bytes": None,
        "mean_server_cpu_percent_one_core_100": None,
        "gpu_mean_utilization_percent_device_wide": None,
        "sampled_peak_gpu_memory_mib_device_wide": None,
        "resource_limitation": "Live sampler values were lost in post-inference reporting failure; inference peak memory and CPU/GPU utilization are unavailable, not estimated",
        "loaded_model_capacity_probe": capacity,
        "gpu_layers_requested": qwen["gpu_layers"],
        "threads": qwen["threads"],
        "recovery": "887 saved responses reconstructed; no additional review predictions",
    }
    atomic_json(root / "qwen_resource_report.json", resources)
    analyze(config, frame, predictions, result, candidates, selection, resources)
    mlflow = configure_tracking(tracking_config())
    with mlflow.start_run(run_id=selection["run_id"]):
        mlflow.log_metrics(
            {
                "test_" + key: value
                for key, value in result.items()
                if isinstance(value, (float, int))
            }
        )
        for name in (
            "qwen_test_metrics.json",
            "qwen_confusion_matrix.png",
            "qwen_classification_report.csv",
            "qwen_resource_report.json",
        ):
            mlflow.log_artifact(str(root / name))
    paths = (
        list(root.glob("*.json"))
        + list(root.glob("*.csv"))
        + list(root.glob("*.png"))
        + [
            root / "qwen_predictions.parquet",
            config.reports_dir / "task2_proposed_v1_evidence.json",
        ]
    )
    atomic_json(
        state_path,
        {
            "protocol_hash": selection["protocol_hash"],
            "status": "completed",
            "pending_sample": None,
            "completed": datetime.now(UTC).isoformat(),
            "reporting_recovery_only": True,
            "execution_source_hashes": saved["source_hashes"],
            "artifacts": {str(path): file_checksum(path) for path in paths if path != state_path},
        },
    )
    return result

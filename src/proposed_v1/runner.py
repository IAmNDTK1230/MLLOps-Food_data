import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.baselines.metrics import LABELS, confusion_plot
from src.baselines.runner import load_protocol, tracking_config
from src.ml.mlflow_tracking import configure_tracking
from src.proposed_v1.runtime import LocalQwen
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def freeze_prompts(frame, directory, root):
    train = frame.loc[frame.split == "train"]
    unambiguous = train.groupby("text").sentiment.nunique().eq(1)
    examples = []
    for label in LABELS:
        candidates = train.loc[
            (train.sentiment == label)
            & train.text.isin(unambiguous[unambiguous].index)
            & train.text.str.len().between(30, 120)
        ].sort_values(["review_id", "text"])
        if candidates.empty:
            raise ValueError("No eligible train-only few-shot example for a class")
        row = candidates.iloc[0]
        examples.append(
            {"review_id": row.review_id, "text": row.text, "sentiment": label, "origin": "train"}
        )
    prompts = {}
    for name in ("v1_a", "v1_b", "v1_c"):
        path = directory / f"{name}.txt"
        text = path.read_text(encoding="utf-8")
        if "{{TRAIN_EXAMPLES}}" in text:
            text = text.replace(
                "{{TRAIN_EXAMPLES}}",
                json.dumps(
                    [{key: e[key] for key in ("text", "sentiment")} for e in examples],
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            path.write_text(text, encoding="utf-8")
        prompts[name] = {"text": text, "sha256": file_checksum(path)}
    manifest_path = root / "prompt_manifest.json"
    manifest = {
        "prompts": {name: value["sha256"] for name, value in prompts.items()},
        "few_shot_examples": examples,
        "example_policy": "one per class, train only, nonconflicting text, length 30..120; sorted IDs/text; no validation/test inspection",
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Frozen prompts/examples changed; refusing to retune after selection")
    atomic_json(manifest_path, manifest)
    return prompts, manifest


def select_prompt(rows):
    return max(
        rows, key=lambda row: (row["macro_f1"], row["json_validity_rate"], -row["mean_latency_ms"])
    )


def score_predictions(truth, predictions):
    predictions = [value if value is not None else "__invalid__" for value in predictions]
    p, r, f, _ = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, average="macro", zero_division=0
    )
    report = classification_report(
        truth, predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "neutral_precision": float(report["neutral"]["precision"]),
        "neutral_recall": float(report["neutral"]["recall"]),
        "neutral_f1": float(report["neutral"]["f1-score"]),
    }, report


def usage(records, duration):
    return {
        "json_validity_rate": float(records.json_valid.mean()),
        "first_attempt_json_validity_rate": float(records.first_attempt_valid.mean()),
        "invalid_output_count": int(records.invalid_outputs.sum()),
        "terminal_invalid_samples": int((~records.json_valid).sum()),
        "mean_latency_ms": float(records.latency_ms.mean()),
        "p95_latency_ms": float(records.latency_ms.quantile(0.95)),
        "input_tokens": int(records.input_tokens.sum()),
        "output_tokens": int(records.output_tokens.sum()),
        "total_tokens": int(records.input_tokens.sum() + records.output_tokens.sum()),
        "mean_tokens_per_sample": float((records.input_tokens + records.output_tokens).mean()),
        "p95_input_tokens": float(records.input_tokens.quantile(0.95)),
        "maximum_input_tokens": int(records.input_tokens.max()),
        "evaluation_duration_seconds": duration,
        "samples_per_second": len(records) / duration if duration else None,
        "output_tokens_per_second": int(records.output_tokens.sum()) / duration
        if duration
        else None,
    }


def predict_partition(client, instruction, rows, path, protocol_hash, state_path=None):
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if any(row["protocol_hash"] != protocol_hash for row in records):
            raise ValueError("Prediction checkpoint belongs to a different protocol")
    done = {record["sample_index"]: record for record in records}
    if state_path and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pending = state.get("pending_sample")
        if pending is not None and pending not in done:
            raise ValueError(
                "An in-flight test request was interrupted; do not silently predict it twice"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with path.open("a", encoding="utf-8") as stream:
        for index, row in rows.iterrows():
            if index in done:
                if done[index]["text_sha256"] != digest(row.text):
                    raise ValueError("Checkpoint text changed")
                continue
            if state_path:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["pending_sample"] = int(index)
                atomic_json(state_path, state)
            result = client.predict(instruction, row.text)
            record = {
                "sample_index": int(index),
                "text_sha256": digest(row.text),
                "protocol_hash": protocol_hash,
                **result,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            if state_path:
                state["pending_sample"] = None
                atomic_json(state_path, state)
            if len(records) % 100 == 0:
                print(f"{path.stem}: {len(records)}/{len(rows)}", flush=True)
    if len(records) != len(rows) or set(record["sample_index"] for record in records) != set(
        rows.index
    ):
        raise ValueError("Prediction checkpoint membership differs from frozen partition")
    return pd.DataFrame(records).set_index("sample_index").loc[rows.index].rename_axis(
        "sample_index"
    ).reset_index(), time.perf_counter() - start


def run(config, config_path="qwen_v1.yaml"):
    frame, baseline_protocol, _ = load_protocol(config)  # No baseline fit/evaluation calls.
    qwen = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = config.reports_dir / "proposed_v1"
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(qwen["model_dir"]) / "provenance.json"
    if not provenance_path.exists():
        raise ValueError(
            "Qwen runtime/model missing. Run python -m scripts.setup_qwen_v1; no inference has been claimed"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance["model"] != qwen["model_repo"]
        or provenance["revision"] != qwen["model_revision"]
        or provenance["runtime_tag"] != qwen["runtime_tag"]
    ):
        raise ValueError("Model/runtime provenance does not match frozen config")
    if (
        file_checksum(Path(qwen["model_dir"]) / qwen["model_file"]) != provenance["model_sha256"]
        or file_checksum(Path(provenance["server_executable"])) != provenance["server_sha256"]
    ):
        raise ValueError("Qwen model/runtime checksum changed")
    prompts, manifest = freeze_prompts(frame, Path(qwen["prompt_dir"]), root)
    protocol = {
        "dataset_hash": baseline_protocol["dataset_hash"],
        "split_hash": baseline_protocol["split_hash"],
        "config": qwen,
        "provenance": provenance,
        "prompt_manifest": manifest,
        "source_hashes": {
            name: file_checksum(Path(__file__).parent / name)
            for name in ("runner.py", "runtime.py", "analysis.py")
        },
    }
    protocol_hash = digest(protocol)
    state_path = root / "test_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["status"] == "completed":
            saved = json.loads((root / "selection.json").read_text(encoding="utf-8"))[
                "frozen_protocol"
            ]
            if {k: v for k, v in protocol.items() if k != "source_hashes"} != {
                k: v for k, v in saved.items() if k != "source_hashes"
            }:
                raise ValueError(
                    "Frozen model/prompt/data/settings changed; no cached substitution"
                )
            for filename, sha in state["artifacts"].items():
                if file_checksum(Path(filename)) != sha:
                    raise ValueError("Saved Qwen results changed; no second test evaluation")
            return json.loads((root / "qwen_test_metrics.json").read_text(encoding="utf-8"))
        if state["protocol_hash"] != protocol_hash:
            raise ValueError(
                "Frozen Qwen test protocol changed; recover saved reports without inference"
            )
    mlflow = configure_tracking(tracking_config())
    selection_path = root / "selection.json"
    candidates = []
    with LocalQwen(qwen, provenance) as client:
        validation = frame.loc[frame.split == "validation"]
        for name, prompt in prompts.items():
            metric_path = root / f"{name}_validation_metrics.json"
            if metric_path.exists():
                result = json.loads(metric_path.read_text(encoding="utf-8"))
                if result["protocol_hash"] != protocol_hash:
                    raise ValueError("Validation evidence/protocol mismatch")
                candidates.append(result)
                continue
            with mlflow.start_run(run_name="qwen_" + name) as tracking:
                # One train-only warmup; never use test input as warmup or examples.
                warmup = client.predict(
                    prompt["text"], frame.loc[frame.split == "train"].iloc[0].text
                )
                records, duration = predict_partition(
                    client,
                    prompt["text"],
                    validation,
                    root / f"{name}_validation_checkpoint.jsonl",
                    protocol_hash,
                )
                metrics, _ = score_predictions(validation.sentiment, records.prediction)
                result = {
                    "prompt": name,
                    "prompt_hash": prompt["sha256"],
                    "protocol_hash": protocol_hash,
                    "run_id": tracking.info.run_id,
                    **metrics,
                    **usage(records, duration),
                    "warmup_usage": warmup,
                }
                mlflow.log_params(
                    {
                        "model": qwen["model_repo"],
                        "revision": qwen["model_revision"],
                        "quantization": qwen["quantization"],
                        "runtime": qwen["runtime_tag"],
                        "prompt_version": name,
                        "prompt_hash": prompt["sha256"],
                        "temperature": qwen["temperature"],
                        "max_new_tokens": qwen["max_new_tokens"],
                        "dataset_hash": protocol["dataset_hash"],
                        "split_hash": protocol["split_hash"],
                        "features": "text_only",
                        "structured_output": "json_schema",
                    }
                )
                mlflow.log_metrics(
                    {
                        "validation_" + key: value
                        for key, value in result.items()
                        if isinstance(value, (int, float))
                    }
                )
                mlflow.log_artifact(str(Path(qwen["prompt_dir"]) / f"{name}.txt"))
                atomic_json(metric_path, result)
                records.to_parquet(root / f"{name}_validation_predictions.parquet", index=False)
                candidates.append(result)
        pd.DataFrame(
            [
                {key: value for key, value in row.items() if key != "warmup_usage"}
                for row in candidates
            ]
        ).to_csv(root / "qwen_prompt_comparison.csv", index=False)
        selected = select_prompt(candidates)
        selection = {
            "protocol_hash": protocol_hash,
            "selected_prompt": selected["prompt"],
            "validation_macro_f1": selected["macro_f1"],
            "run_id": selected["run_id"],
            "prompt_hash": selected["prompt_hash"],
            "frozen_protocol": protocol,
            "selection": "validation_macro_f1, then JSON validity, then lower mean latency",
        }
        if (
            selection_path.exists()
            and json.loads(selection_path.read_text(encoding="utf-8")) != selection
        ):
            raise ValueError("Frozen selected prompt changed")
        atomic_json(selection_path, selection)
        # Confirm tracking backend before any held-out request.
        mlflow.get_run(selected["run_id"])
        if not state_path.exists():
            atomic_json(
                state_path,
                {
                    "protocol_hash": protocol_hash,
                    "status": "in_progress",
                    "pending_sample": None,
                    "started": datetime.now(UTC).isoformat(),
                },
            )
        test = frame.loc[frame.split == "test"]
        records, duration = predict_partition(
            client,
            prompts[selected["prompt"]]["text"],
            test,
            root / "test_checkpoint.jsonl",
            protocol_hash,
            state_path,
        )
        metrics, report = score_predictions(test.sentiment, records.prediction)
        result = {
            **metrics,
            **usage(records, duration),
            "dataset_hash": protocol["dataset_hash"],
            "split_hash": protocol["split_hash"],
            "selected_prompt": selected["prompt"],
            "model": qwen["model_repo"],
            "revision": qwen["model_revision"],
            "model_size_mb": provenance["model_size_bytes"] / 1_000_000,
            "evaluation_split": "test",
            "sample_count": len(test),
        }
        atomic_json(root / "qwen_test_metrics.json", result)
        predictions = (
            test.reset_index()
            .rename(columns={"index": "sample_index", "sentiment": "true_label"})
            .merge(records, on="sample_index", validate="one_to_one")
        )
        predictions.to_parquet(root / "qwen_predictions.parquet", index=False)
        pd.DataFrame(
            [{"label": label, **report[label]} for label in (*LABELS, "macro avg", "weighted avg")]
        ).to_csv(root / "qwen_classification_report.csv", index=False)
        if records.json_valid.all():
            confusion_plot(
                test.sentiment,
                records.prediction,
                root / "qwen_confusion_matrix.png",
                "Qwen Proposed V1 — test",
            )
        else:
            # Abstentions are errors; never silently map malformed output to a label.
            import matplotlib.pyplot as plt

            labels = [*LABELS, "__invalid__"]
            matrix = confusion_matrix(
                test.sentiment, records.prediction.fillna("__invalid__"), labels=labels
            )
            fig, ax = plt.subplots()
            ax.imshow(matrix)
            ax.set_xticks(range(4), labels)
            ax.set_yticks(range(4), labels)
            fig.tight_layout()
            fig.savefig(root / "qwen_confusion_matrix.png")
            plt.close(fig)
        from src.proposed_v1.analysis import analyze

        resources = {
            "sampled_peak_server_rss_bytes": client.peak_rss,
            "mean_server_cpu_percent_one_core_100": float(np.mean(client.cpu_samples))
            if client.cpu_samples
            else None,
            "gpu_mean_utilization_percent_device_wide": float(
                np.mean([r[0] for r in client.gpu_samples])
            )
            if client.gpu_samples
            else None,
            "sampled_peak_gpu_memory_mib_device_wide": max(
                [r[1] for r in client.gpu_samples], default=None
            ),
            "sampling_interval_seconds": 1,
            "test": result,
            "validation": candidates,
            "gpu_layers_requested": qwen["gpu_layers"],
            "threads": qwen["threads"],
            "context_size": qwen["context_size"],
            "measurement_scope": "resource samples cover model session including validation/warmups; token counts are per partition, warmups separate",
        }
        atomic_json(root / "qwen_resource_report.json", resources)
        analyze(config, frame, predictions, result, candidates, selection, resources)
        with mlflow.start_run(run_id=selected["run_id"]):
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
                "protocol_hash": protocol_hash,
                "status": "completed",
                "pending_sample": None,
                "completed": datetime.now(UTC).isoformat(),
                "artifacts": {
                    str(path): file_checksum(path) for path in paths if path != state_path
                },
            },
        )
        return result

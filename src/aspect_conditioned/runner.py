import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.aspect_conditioned.prompts import classical_input, freeze_prompts, immutable_json
from src.aspect_conditioned.runtime import AspectRuntime
from src.baselines.metrics import class_report, confusion_plot, measure_predictions
from src.baselines.runner import load_protocol, tracking_config
from src.baselines.tfidf_logreg import build_pipeline
from src.ml.mlflow_tracking import configure_tracking
from src.proposed_v1.runner import digest, score_predictions, select_prompt, usage
from src.proposed_v2.runner import assert_frozen, frozen_inputs, join_predictions
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def log_numbers(mlflow, prefix, metrics):
    mlflow.log_metrics(
        {prefix + k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
    )


def log_params(mlflow, protocol, **extra):
    cfg = protocol["runtime_config"]
    mlflow.log_params(
        {
            "input_protocol": "text+aspect",
            "seed": 42,
            "dataset_hash": protocol["dataset_hash"],
            "split_hash": protocol["split_hash"],
            **extra,
        }
    )
    if extra.get("family") == "qwen":
        mlflow.log_params(
            {
                k: cfg[k]
                for k in (
                    "model_repo",
                    "model_revision",
                    "quantization",
                    "runtime_tag",
                    "temperature",
                    "max_new_tokens",
                    "context_size",
                    "threads",
                    "gpu_layers",
                )
            }
        )
        mlflow.log_param("model_hash", protocol["model_hash"])


def validate_records(records, rows, protocol_hash):
    if (
        len(records) != len(rows)
        or records.sample_index.duplicated().any()
        or set(records.sample_index) != set(rows.index)
        or not records.protocol_hash.eq(protocol_hash).all()
    ):
        raise ValueError("Aspect checkpoint membership/protocol mismatch")
    records = (
        records.set_index("sample_index").loc[rows.index].rename_axis("sample_index").reset_index()
    )
    if records.input_sha256.tolist() != [
        digest({"text": r.text, "aspect": r.aspect}) for r in rows.itertuples()
    ]:
        raise ValueError("Aspect checkpoint input mismatch")
    return records


def predict_partition(client, prompt, rows, path, protocol_hash, state_path=None):
    records = (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )
    done = {r["sample_index"]: r for r in records}
    if len(done) != len(records) or any(r["protocol_hash"] != protocol_hash for r in records):
        raise ValueError("Duplicate/mismatched aspect checkpoint")
    if not set(done) <= set(rows.index):
        raise ValueError("Checkpoint includes unexpected rows")
    if state_path and state_path.exists():
        pending = json.loads(state_path.read_text(encoding="utf-8")).get("pending_sample")
        if pending is not None and pending not in done:
            raise ValueError("Unresolved in-flight TEST request; refusing repeat inference")
    start = time.perf_counter()
    with path.open("a", encoding="utf-8") as stream:
        for index, row in rows.iterrows():
            sha = digest({"text": row.text, "aspect": row.aspect})
            if index in done:
                if done[index]["input_sha256"] != sha:
                    raise ValueError("Checkpoint text/aspect changed")
                continue
            if state_path:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["pending_sample"] = int(index)
                atomic_json(state_path, state)
            result = client.predict_aspect(prompt, row.text, row.aspect)
            record = {
                "sample_index": int(index),
                "input_sha256": sha,
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
    return validate_records(pd.DataFrame(records), rows, protocol_hash), time.perf_counter() - start


def classical_experiment(frame, settings, protocol, root, mlflow):
    train, validation = (frame.loc[frame.split.eq(name)] for name in ("train", "validation"))

    def features(rows):
        return [classical_input(r.text, r.aspect) for r in rows.itertuples()]

    xtrain, xval = features(train), features(validation)
    trials = []
    for trial in settings["tfidf_trials"]:
        path = root / f"tfidf_{trial['name']}_validation.json"
        if path.exists():
            trials.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        with mlflow.start_run(run_name="aspect_tfidf_" + trial["name"]) as run:
            model = build_pipeline(trial, class_weight="balanced", seed=42)
            model.fit(xtrain, train.sentiment)
            predictions, latency = measure_predictions(model, xval, xtrain)
            scores, _ = score_predictions(validation.sentiment, predictions)
            model_path = root / f"tfidf_{trial['name']}.joblib"
            joblib.dump(model, model_path, compress=3)
            result = {
                "prompt": trial["name"],
                **scores,
                **latency,
                "settings": trial,
                "run_id": run.info.run_id,
                "model_path": str(model_path),
                "model_sha256": file_checksum(model_path),
            }
            log_params(mlflow, protocol, family="tfidf", class_weight="balanced", **trial)
            log_numbers(mlflow, "validation_", result)
            atomic_json(path, result)
            pd.DataFrame({"sample_index": validation.index, "prediction": predictions}).to_parquet(
                root / f"tfidf_{trial['name']}_validation_predictions.parquet", index=False
            )
            trials.append(result)
    pd.DataFrame(trials).to_csv(root / "tfidf_aspect_validation.csv", index=False)
    selected = max(trials, key=lambda r: r["macro_f1"])
    immutable_json(root / "tfidf_selection.json", selected)
    return selected


def classical_test(frame, selected, protocol, root, mlflow):
    path, state_path = root / "tfidf_aspect_test_metrics.json", root / "tfidf_test_state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if state_path.exists():
        raise ValueError(
            "Aspect TF-IDF test was attempted without saved metrics; no automatic rerun"
        )
    if file_checksum(Path(selected["model_path"])) != selected["model_sha256"]:
        raise ValueError("Selected aspect TF-IDF model changed")
    test, train = (frame.loc[frame.split.eq(name)] for name in ("test", "train"))
    model = joblib.load(selected["model_path"])
    immutable_json(state_path, {"status": "started", "protocol_hash": protocol["protocol_hash"]})
    inputs = [classical_input(r.text, r.aspect) for r in test.itertuples()]
    warmups = [classical_input(r.text, r.aspect) for r in train.head(3).itertuples()]
    prediction, latency = measure_predictions(model, inputs, warmups)
    records = pd.DataFrame({"sample_index": test.index, "prediction": prediction})
    join_predictions(test, records).to_parquet(
        root / "tfidf_aspect_predictions.parquet", index=False
    )
    metrics, _ = score_predictions(test.sentiment, prediction)
    result = {
        **metrics,
        **latency,
        "sample_count": len(test),
        "input_protocol": "text+aspect",
        "model_size_mb": Path(selected["model_path"]).stat().st_size / 1e6,
        "test_hygiene": protocol["test_hygiene"],
    }
    atomic_json(path, result)
    class_report(test.sentiment, prediction).to_csv(
        root / "tfidf_aspect_classification_report.csv", index=False
    )
    with mlflow.start_run(run_id=selected["run_id"]):
        log_numbers(mlflow, "test_", result)
        mlflow.log_artifact(str(path))
    atomic_json(state_path, {"status": "completed", "protocol_hash": protocol["protocol_hash"]})
    return result


def finish(config, frame, protocol, root, mlflow):
    selection = json.loads((root / "qwen_selection.json").read_text(encoding="utf-8"))
    test = frame.loc[frame.split.eq("test")]
    records = validate_records(
        pd.DataFrame(
            [
                json.loads(line)
                for line in (root / "qwen_test_checkpoint.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        ),
        test,
        protocol["protocol_hash"],
    )
    measured = json.loads((root / "test_measurements.json").read_text(encoding="utf-8"))
    scores, _ = score_predictions(test.sentiment, records.prediction)
    metrics = {
        **scores,
        **usage(records, measured["duration_seconds"]),
        "selected_prompt": selection["prompt"],
        "input_protocol": "text+aspect",
        "test_hygiene": protocol["test_hygiene"],
        "sample_count": len(test),
        "dataset_hash": protocol["dataset_hash"],
        "split_hash": protocol["split_hash"],
    }
    atomic_json(root / "qwen_test_metrics.json", metrics)
    predictions = join_predictions(test, records)
    predictions.to_parquet(root / "qwen_predictions.parquet", index=False)
    class_report(test.sentiment, records.prediction.fillna("__invalid__")).to_csv(
        root / "qwen_classification_report.csv", index=False
    )
    if records.json_valid.all():
        confusion_plot(
            test.sentiment,
            records.prediction,
            root / "qwen_confusion_matrix.png",
            "Qwen text+aspect — exploratory reused test",
        )
    else:
        from src.proposed_v2.analysis import abstention_plot

        abstention_plot(test.sentiment, records.prediction, root / "qwen_confusion_matrix.png")
    from src.aspect_conditioned.analysis import analyze

    analyze(config, frame, root, protocol)
    with mlflow.start_run(run_id=selection["run_id"]):
        log_numbers(mlflow, "test_", metrics)
        for name in (
            "qwen_test_metrics.json",
            "conflicting_text_comparison.csv",
            "comparison.csv",
            "qwen_confusion_matrix.png",
            "resource_report.json",
        ):
            mlflow.log_artifact(str(root / name))
    assert_frozen(protocol["protected_inputs"])
    paths = [p for p in root.rglob("*") if p.is_file() and p.name != "test_state.json"]
    paths.append(config.reports_dir / "task2_aspect_v2_evidence.json")
    atomic_json(
        root / "test_state.json",
        {
            "status": "completed",
            "pending_sample": None,
            "protocol_hash": protocol["protocol_hash"],
            "artifacts": {str(p): file_checksum(p) for p in paths},
        },
    )
    return metrics


def preserve_inference_protocol(root, computed):
    """Permit explicitly audited reporting fixes only after durable full TEST responses.

    Original inference source/settings snapshot remains immutable. A recovery must
    pin exact new sources; it cannot authorize prompt/runtime/data changes or calls.
    """
    path = root / "protocol.json"
    if not path.exists():
        return computed
    original = json.loads(path.read_text(encoding="utf-8"))
    if original == computed:
        return original
    audit_path = root / "posttest_reporting_recovery.json"
    if not audit_path.exists() or not (root / "test_measurements.json").exists():
        raise ValueError("Frozen experiment changed without complete reporting-only recovery")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ignored = {"source_hashes", "protocol_hash"}
    if {k: v for k, v in computed.items() if k not in ignored} != {
        k: v for k, v in original.items() if k not in ignored
    }:
        raise ValueError("Reporting recovery cannot change inference inputs/settings")
    changed = {
        Path(k).name
        for k, v in computed["source_hashes"].items()
        if original["source_hashes"].get(k) != v
    }
    if (
        not changed <= {"runner.py", "analysis.py"}
        or audit["source_hashes"] != computed["source_hashes"]
        or audit["inference_protocol_hash"] != original["protocol_hash"]
        or audit["new_inference_calls"] != 0
    ):
        raise ValueError("Unapproved reporting source changes")
    return original


def run(config, config_path="qwen_aspect.yaml", prepare_only=False):
    frame, base, _ = load_protocol(config)
    settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(settings["base_runtime_config"]).read_text(encoding="utf-8"))
    old = json.loads(
        (config.reports_dir / "proposed_v1/selection.json").read_text(encoding="utf-8")
    )
    if cfg != old["frozen_protocol"]["config"] or cfg["temperature"] != 0 or cfg["seed"] != 42:
        raise ValueError("Aspect experiment must preserve V1 runtime/generation settings")
    root = config.reports_dir / "aspect_conditioned"
    root.mkdir(parents=True, exist_ok=True)
    prompts, manifest = freeze_prompts(
        frame.loc[frame.split.eq("train")], settings["prompt_dir"], root
    )
    provenance = json.loads(
        (Path(cfg["model_dir"]) / "provenance.json").read_text(encoding="utf-8")
    )
    for path, sha in (
        (Path(cfg["model_dir"]) / cfg["model_file"], provenance["model_sha256"]),
        (Path(provenance["server_executable"]), provenance["server_sha256"]),
    ):
        if file_checksum(path) != sha:
            raise ValueError("Pinned model/runtime changed")
    protected = frozen_inputs(config)
    protected.update(
        {
            str(p): file_checksum(p)
            for p in (config.reports_dir / "proposed_v2").rglob("*")
            if p.is_file()
        }
    )
    protected.update(
        {
            str(config.reports_dir / name): file_checksum(config.reports_dir / name)
            for name in ("task2_proposed_v1_evidence.json", "task2_proposed_v2_evidence.json")
        }
    )
    protocol = {
        "dataset_hash": base["dataset_hash"],
        "split_hash": base["split_hash"],
        "runtime_config": cfg,
        "model_hash": provenance["model_sha256"],
        "settings": settings,
        "input_features": ["text", "aspect"],
        "prompt_manifest": manifest,
        "protected_inputs": protected,
        "source_hashes": {str(p): file_checksum(p) for p in Path(__file__).parent.glob("*.py")},
        "test_hygiene": settings["test_hygiene"],
    }
    protocol["protocol_hash"] = digest(protocol)
    protocol = preserve_inference_protocol(root, protocol)
    immutable_json(root / "protocol.json", protocol)
    if prepare_only:
        return {"status": "prepared_no_inference", "prompts": manifest["prompt_hashes"]}
    state_path = root / "test_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("TEST protocol changed")
        if state["status"] == "completed":
            assert_frozen(state["artifacts"])
            return json.loads((root / "qwen_test_metrics.json").read_text(encoding="utf-8"))
    mlflow = configure_tracking(tracking_config())
    if (root / "test_measurements.json").exists():
        return finish(config, frame, protocol, root, mlflow)
    classical = classical_experiment(frame, settings, protocol, root, mlflow)
    validation, train = (frame.loc[frame.split.eq(name)] for name in ("validation", "train"))
    candidates = []
    with AspectRuntime(cfg, provenance, root) as client:
        for name, prompt in prompts.items():
            path = root / f"{name}_validation_metrics.json"
            if path.exists():
                candidates.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            with mlflow.start_run(run_name="qwen_aspect_" + name) as run:
                warmup = client.predict_aspect(prompt, train.iloc[0].text, train.iloc[0].aspect)
                records, duration = predict_partition(
                    client,
                    prompt,
                    validation,
                    root / f"{name}_validation_checkpoint.jsonl",
                    protocol["protocol_hash"],
                )
                scores, _ = score_predictions(validation.sentiment, records.prediction)
                result = {
                    "prompt": name,
                    "prompt_hash": digest(prompt),
                    **scores,
                    **usage(records, duration),
                    "run_id": run.info.run_id,
                    "warmup_usage": warmup,
                    "neutral_prediction_count": int(records.prediction.eq("neutral").sum()),
                }
                log_params(mlflow, protocol, family="qwen", prompt=name, prompt_hash=digest(prompt))
                log_numbers(mlflow, "validation_", result)
                mlflow.log_artifact(str(Path(settings["prompt_dir"]) / f"{name}.txt"))
                mlflow.log_artifact(str(root / "train_example_manifest.json"))
                join_predictions(validation, records).to_parquet(
                    root / f"{name}_validation_predictions.parquet", index=False
                )
                class_report(validation.sentiment, records.prediction.fillna("__invalid__")).to_csv(
                    root / f"{name}_validation_classification_report.csv", index=False
                )
                atomic_json(path, result)
                candidates.append(result)
        pd.DataFrame(
            [{k: v for k, v in r.items() if k != "warmup_usage"} for r in candidates]
        ).to_csv(root / "qwen_prompt_comparison.csv", index=False)
        selected = select_prompt(candidates)
        immutable_json(
            root / "qwen_selection.json",
            {
                **selected,
                "protocol_hash": protocol["protocol_hash"],
                "model_hash": protocol["model_hash"],
                "dataset_hash": protocol["dataset_hash"],
                "split_hash": protocol["split_hash"],
                "generation_settings": cfg,
                "selection_criterion": "validation macro F1, JSON validity, lower latency",
            },
        )
        # Both selections are frozen before either new TEST evaluation.
        classical_test(frame, classical, protocol, root, mlflow)
        if not state_path.exists():
            atomic_json(
                state_path,
                {
                    "status": "in_progress",
                    "pending_sample": None,
                    "protocol_hash": protocol["protocol_hash"],
                },
            )
        _, duration = predict_partition(
            client,
            prompts[selected["prompt"]],
            frame.loc[frame.split.eq("test")],
            root / "qwen_test_checkpoint.jsonl",
            protocol["protocol_hash"],
            state_path,
        )
        atomic_json(root / "test_measurements.json", {"duration_seconds": duration})
    return finish(config, frame, protocol, root, mlflow)

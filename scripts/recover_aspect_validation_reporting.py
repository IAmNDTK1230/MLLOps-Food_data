"""One-time, pre-TEST recovery of completed B1 validation after an index-name bug.

No model is loaded/called. Original protocol/checkpoint are preserved verbatim.
Only source/protocol hashes change; prompt/model/input/settings remain identical.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from src.aspect_conditioned.runner import validate_records
from src.baselines.metrics import class_report
from src.baselines.runner import load_protocol, tracking_config
from src.food.config import FoodConfig
from src.ml.mlflow_tracking import configure_tracking
from src.proposed_v1.runner import digest, score_predictions
from src.proposed_v2.runner import assert_frozen, join_predictions
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def main():
    config = FoodConfig.load("params.yaml")
    root = config.reports_dir / "aspect_conditioned"
    if any((root / name).exists() for name in (
        "test_state.json", "tfidf_test_state.json", "test_measurements.json",
        "v2_b1_validation_metrics.json", "pretest_reporting_recovery.json",
    )):
        raise ValueError("Recovery only allowed once before any TEST attempt or completed B1 report")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    assert_frozen(protocol["protected_inputs"])
    frame, _, _ = load_protocol(config)
    validation = frame.loc[frame.split.eq("validation")]
    path = root / "v2_b1_validation_checkpoint.jsonl"
    records = validate_records(pd.DataFrame([json.loads(line) for line in
        path.read_text(encoding="utf-8").splitlines()]), validation, protocol["protocol_hash"])
    mlflow = configure_tracking(tracking_config())
    runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'qwen_aspect_v2_b1'",
                             output_format="list")
    matches = [r for r in runs if r.data.params.get("prompt_hash") ==
               protocol["prompt_manifest"]["prompt_hashes"]["v2_b1"]]
    if len(matches) != 1:
        raise ValueError("Expected one matching failed B1 run")
    run = matches[0]
    measured = {k.removeprefix("validation_"): v for k, v in run.data.metrics.items()
                if k.startswith("validation_")}
    scores, _ = score_predictions(validation.sentiment, records.prediction)
    if any(abs(measured[k] - v) > 1e-12 for k, v in scores.items()):
        raise ValueError("Saved responses disagree with original logged validation scores")
    backup = Path("logs/aspect_validation_reporting_recovery")
    backup.mkdir(parents=True, exist_ok=False)
    for name in ("protocol.json", "v2_b1_validation_checkpoint.jsonl", "resource_session.json"):
        shutil.copy2(root / name, backup / name)
    prior_protocol_hash = protocol.pop("protocol_hash")
    protocol["source_hashes"] = {str(p): file_checksum(p)
                                for p in Path("src/aspect_conditioned").resolve().glob("*.py")}
    protocol["protocol_hash"] = digest(protocol)
    records["protocol_hash"] = protocol["protocol_hash"]
    checkpoint = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records.to_dict("records"))
    path.write_text(checkpoint, encoding="utf-8")
    atomic_json(root / "protocol.json", protocol)
    result = {**measured, "prompt": "v2_b1", "run_id": run.info.run_id,
              "prompt_hash": run.data.params["prompt_hash"],
              "warmup_usage": {"input_tokens": None, "output_tokens": None},
              "recovery_note": "Original MLflow measurements and responses retained; original TRAIN warmup tokens unavailable"}
    join_predictions(validation, records).to_parquet(root / "v2_b1_validation_predictions.parquet", index=False)
    report = root / "v2_b1_validation_classification_report.csv"
    class_report(validation.sentiment, records.prediction).to_csv(report, index=False)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tag("reporting_recovery", "pretest_index_name_fix_no_new_inference")
        mlflow.log_artifact(str(report))
    atomic_json(root / "v2_b1_validation_metrics.json", result)
    atomic_json(root / "pretest_reporting_recovery.json", {
        "reason": "pandas .loc dropped index name; fixed explicit sample_index before join",
        "prior_protocol_hash": prior_protocol_hash, "new_protocol_hash": protocol["protocol_hash"],
        "backup_directory": str(backup), "validation_responses_reused": len(records),
        "new_inference_calls": 0, "test_attempts_before_recovery": 0,
        "first_session_resources": json.loads((backup / "resource_session.json").read_text(encoding="utf-8")),
        "limitation": "first TRAIN warmup token measurements unavailable; evaluation token counts complete"})
    print("Recovered 842 B1 validation responses/metrics without inference; TEST has not started.")


if __name__ == "__main__":
    main()

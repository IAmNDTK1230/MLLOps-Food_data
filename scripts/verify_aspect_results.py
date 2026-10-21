"""Read-only verification of completed aspect experiment; no model inference."""

import json

import pandas as pd

from src.baselines.runner import load_protocol, tracking_config
from src.food.config import FoodConfig
from src.ml.mlflow_tracking import configure_tracking
from src.proposed_v1.runner import score_predictions
from src.proposed_v2.runner import assert_frozen


def main():
    config = FoodConfig.load("params.yaml")
    root = config.reports_dir / "aspect_conditioned"
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    state = json.loads((root / "test_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed" and state["pending_sample"] is None
    assert_frozen(protocol["protected_inputs"])
    assert_frozen(state["artifacts"])
    frame, _, _ = load_protocol(config)
    test = frame.loc[frame.split.eq("test")]
    for predictions, metrics in (("qwen_predictions.parquet", "qwen_test_metrics.json"),
                                  ("tfidf_aspect_predictions.parquet", "tfidf_aspect_test_metrics.json")):
        records = pd.read_parquet(root / predictions)
        assert len(records) == 887 and records.sample_index.nunique() == 887
        records = records.set_index("sample_index").loc[test.index]
        assert records.text.tolist() == test.text.tolist()
        assert records.aspect.tolist() == test.aspect.tolist()
        assert records.true_label.tolist() == test.sentiment.tolist()
        scores, _ = score_predictions(test.sentiment, records.prediction)
        saved = json.loads((root / metrics).read_text(encoding="utf-8"))
        assert all(abs(saved[k] - v) < 1e-12 for k, v in scores.items())
    mlflow = configure_tracking(tracking_config())
    runs = [run for run in mlflow.search_runs(output_format="list")
            if run.data.params.get("input_protocol") == "text+aspect"]
    selected_qwen = json.loads((root / "qwen_selection.json").read_text(encoding="utf-8"))["run_id"]
    selected_tfidf = json.loads((root / "tfidf_selection.json").read_text(encoding="utf-8"))["run_id"]
    assert len(runs) == 6
    for run in runs:
        assert run.info.status == "FINISHED"
        has_test = "test_macro_f1" in run.data.metrics
        assert has_test == (run.info.run_id in {selected_qwen, selected_tfidf})
    print(f"PASS: {len(protocol['protected_inputs'])} protected files unchanged; both 887-row test outputs verified; 6 MLflow runs FINISHED; test metrics only on selected Qwen/control.")


if __name__ == "__main__":
    main()

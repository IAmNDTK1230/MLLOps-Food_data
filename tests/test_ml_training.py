from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from src.ml.config import MLConfig, TrackingConfig
from src.ml.evaluate import score_predictions
from src.ml.features import build_pipeline
from src.ml.train import train_experiments


def test_feature_pipeline_configuration_without_fitting():
    config = MLConfig()
    assert len(config.experiments) == 3
    pipeline = build_pipeline(config.experiments[1], 42)
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipeline.named_steps["classifier"].random_state == 42


def test_selection_uses_validation_only_and_tracks_three_experiments(monkeypatch, tmp_path):
    config = MLConfig(models_dir=tmp_path / "models", reports_dir=tmp_path / "reports")
    # Model doubles only: do not fit a real model on manufactured unit inputs.
    frame = pd.DataFrame(
        {
            "comment": ["train-a", "train-b", "val-a", "val-b", "never-read-test"],
            "sentiment": ["negative", "positive", "negative", "positive", "negative"],
            "split": ["train", "train", "validation", "validation", "test"],
        }
    )
    metadata = {
        "counts": {"train": 2, "validation": 2, "test": 1},
        "split_file_sha256": "unit-split",
    }
    report = {
        "source": "unit-double",
        "dataset_hash": "unit-hash",
        "license": "UNIT",
        "label_distribution": {"negative": 3, "positive": 2},
    }
    monkeypatch.setattr("src.ml.train.load_splits", lambda c: (frame, metadata, report))
    models = []
    for predictions in (
        ["positive", "positive"],
        ["negative", "positive"],
        ["negative", "negative"],
    ):
        model = Mock()
        model.predict.side_effect = [predictions, ["negative", "positive"]]
        models.append(model)
    factory = Mock(side_effect=models)
    monkeypatch.setattr("src.ml.train.build_pipeline", factory)
    tracking = MagicMock()
    tracking.start_run.return_value.__enter__.return_value.info.run_id = "unit-run"
    monkeypatch.setattr("src.ml.train.configure_tracking", lambda c: tracking)
    logged = Mock(return_value=SimpleNamespace(model_uri="runs:/unit-run/model"))
    monkeypatch.setattr("src.ml.train.sklearn_tracking.log_model", logged)
    monkeypatch.setattr(
        "src.ml.train.joblib.dump", lambda model, path: path.write_bytes(b"unit-model-double")
    )
    selection = train_experiments(config, TrackingConfig())
    assert selection["experiment"] == "B_bigram"
    assert logged.call_count == 3 and tracking.log_params.call_count == 3
    for model in models:
        assert model.fit.call_args.args[0] == ["train-a", "train-b"]
        assert model.predict.call_args_list[0].args[0] == ["val-a", "val-b"]
    assert "test_macro_f1" not in selection
    assert (config.reports_dir / "experiment_comparison.csv").exists()


def test_training_without_imported_dataset_fails_before_fit(tmp_path, monkeypatch):
    factory = Mock()
    monkeypatch.setattr("src.ml.train.build_pipeline", factory)
    with pytest.raises(ValueError, match="No real review dataset"):
        train_experiments(MLConfig(data_dir=tmp_path / "empty"), TrackingConfig())
    factory.assert_not_called()


def test_binary_macro_metric_uses_present_classes():
    assert score_predictions(["negative", "positive"], ["negative", "positive"])["macro_f1"] == 1

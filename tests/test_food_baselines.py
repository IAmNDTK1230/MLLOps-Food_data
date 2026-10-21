import json

import pandas as pd
import pytest

from src.baselines.metrics import measure_predictions, scores
from src.baselines.rule_based import VietnameseRuleBased, rule_definition
from src.baselines.runner import evaluate, failure_analysis, load_protocol
from src.baselines.tfidf_logreg import EXPERIMENTS, build_pipeline, select_validation_trial
from src.food.config import FoodConfig
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Món ăn ngon", "positive"),
        ("phục vụ thân thiện", "positive"),
        ("đồ ăn dở", "negative"),
        ("không ngon", "negative"),
        ("không tốt", "negative"),
        ("chưa tốt", "negative"),
        ("không tệ", "positive"),
        ("không hề ngon", "negative"),
        ("không không ngon", "positive"),
        ("không, món ngon", "positive"),
        ("không,, ngon", "positive"),
        ("ngon nhưng chậm", "neutral"),
        ("quán nằm ở đường số 5", "neutral"),
        ("", "neutral"),
        ("TƯƠI NGON", "positive"),
        ("tồi tệ", "negative"),
    ],
)
def test_fixed_rule_polarity_negation_and_boundaries(text, expected):
    assert VietnameseRuleBased().predict([text]) == [expected]


def test_longest_phrase_counted_once_and_intensifiers_documented():
    model = VietnameseRuleBased()
    assert model.score("tươi ngon") == 1
    assert model.score("rất ngon") == 1.5
    assert model.score("không rất ngon") == -1.5
    assert rule_definition()["features"] == ["text"]


def test_tfidf_fit_vocabulary_only_from_training_text():
    pipeline = build_pipeline(EXPERIMENTS[1])
    train = ["ngon tốt", "ngon sạch", "dở tệ", "dở chậm", "quán đường", "quán phố"]
    pipeline.fit(train, ["positive", "positive", "negative", "negative", "neutral", "neutral"])
    pipeline.predict(["heldout_only_secret chưa từng xuất hiện"])
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    assert "heldout_only_secret" not in vocabulary
    assert "original_sentiment" not in vocabulary and "aspect" not in vocabulary
    assert pipeline.named_steps["classifier"].class_weight == "balanced"
    assert pipeline.named_steps["classifier"].random_state == 42
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipeline.named_steps["tfidf"].min_df == 2
    assert pipeline.named_steps["classifier"].max_iter >= 1000
    unweighted = build_pipeline(EXPERIMENTS[1], class_weight=None)
    for key in ("ngram_range", "max_features", "min_df", "preprocessor"):
        assert (
            pipeline.named_steps["tfidf"].get_params()[key]
            == unweighted.named_steps["tfidf"].get_params()[key]
        )


def test_selection_uses_validation_only_and_stable_ties():
    trials = [
        {"macro_f1": 0.3, "test_macro_f1": 1},
        {"macro_f1": 0.6, "test_macro_f1": 0},
        {"macro_f1": 0.6, "test_macro_f1": 1},
    ]
    assert select_validation_trial(trials) == 1


def test_latency_does_not_predict_test_samples_twice():
    class Spy:
        def __init__(self):
            self.seen = []

        def predict(self, texts):
            self.seen.extend(texts)
            return ["neutral"] * len(texts)

    model = Spy()
    predictions, latency = measure_predictions(model, ["test1", "test2"], ["validation"])
    assert model.seen == ["validation", "test1", "test2"]
    assert predictions == ["neutral", "neutral"] and latency["p95_latency_ms"] >= 0


def test_macro_metrics_include_all_three_classes():
    report = scores(["positive", "negative", "neutral"], ["positive"] * 3)
    assert report["accuracy"] == pytest.approx(1 / 3)
    assert report["macro_f1"] == pytest.approx(1 / 6)


def test_protocol_rejects_different_split_without_resplitting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.baselines.runner.load_splits",
        lambda config: (pd.DataFrame(), {"counts": {}, "random_seed": 42}, {}),
    )
    with pytest.raises(ValueError, match="verified 5869-row"):
        load_protocol(FoodConfig(reports_dir=tmp_path))


def cached_setup(tmp_path, monkeypatch):
    protocol = {"protocol_hash": "unit"}
    monkeypatch.setattr(
        "src.baselines.runner.load_protocol", lambda config: (pd.DataFrame(), protocol, tmp_path)
    )
    atomic_json(tmp_path / "tfidf_selection.json", protocol)
    atomic_json(tmp_path / "rule_based_validation_metrics.json", protocol)
    return protocol


def test_attempted_test_evaluation_never_retries(tmp_path, monkeypatch):
    protocol = cached_setup(tmp_path, monkeypatch)
    atomic_json(tmp_path / "evaluation_state.json", protocol | {"status": "attempted"})
    with pytest.raises(ValueError, match="already attempted"):
        evaluate(FoodConfig())


def test_completed_evaluation_returns_evidence_without_predictions(tmp_path, monkeypatch):
    protocol = cached_setup(tmp_path, monkeypatch)
    path = tmp_path / "baseline_comparison.csv"
    pd.DataFrame([{"model": "unit", "macro_f1": 0.5}]).to_csv(path, index=False)
    atomic_json(
        tmp_path / "evaluation_state.json",
        protocol | {"status": "completed", "artifacts": {str(path): file_checksum(path)}},
    )
    assert evaluate(FoodConfig())[0]["macro_f1"] == 0.5
    path.write_text("edited")
    with pytest.raises(ValueError, match="no automatic second"):
        evaluate(FoodConfig())


def test_failure_outputs_preserve_predictions_and_actual_aspects(tmp_path):
    frame = pd.DataFrame(
        {
            "review_id": ["a", "b", "c"],
            "text": ["unit"] * 3,
            "aspect": ["food", "service", "room"],
            "sentiment": ["neutral", "positive", "negative"],
        }
    )
    counts = failure_analysis(
        frame, ["positive", "negative", "neutral"], ["negative", "positive", "negative"], tmp_path
    )
    rows = pd.read_csv(tmp_path / "failure_cases.csv")
    assert rows.rule_prediction.tolist() == ["positive", "negative", "neutral"]
    assert counts["rule"]["neutral_as_positive"] == 1
    assert counts["tfidf"]["neutral_as_negative"] == 1
    assert "room" in pd.read_csv(tmp_path / "error_by_aspect.csv").aspect.tolist()
    assert json.dumps(counts)


def test_report_refresh_preserves_none_weight_and_never_predicts(tmp_path, monkeypatch):
    from src.baselines.reporting import finalize_evidence

    config = FoodConfig(reports_dir=tmp_path)
    root = tmp_path / "baselines"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "review_id": ["a"],
            "text": ["unit"],
            "aspect": ["food"],
            "sentiment": ["neutral"],
            "split": ["test"],
        }
    )
    failure_analysis(frame, ["positive"], ["negative"], root)
    pd.DataFrame([{"class_weight": "None", "label": "neutral", "f1-score": 0.2}]).to_csv(
        root / "class_weight_comparison.csv", index=False
    )
    evidence = tmp_path / "task2_baseline_evidence.json"
    atomic_json(
        evidence,
        {
            "slide_6_update": {"class_weight_comparison": [{"class_weight": float("nan")}]},
            "slide_9": {},
        },
    )
    atomic_json(
        root / "evaluation_state.json",
        {
            "status": "completed",
            "split_hash": "split",
            "dataset_hash": "data",
            "artifacts": {str(evidence): file_checksum(evidence)},
        },
    )
    monkeypatch.setattr(
        "src.baselines.reporting.load_splits",
        lambda config: (frame, {"split_file_sha256": "split"}, {"processed_sha256": "data"}),
    )

    def no_prediction(*args):
        raise AssertionError("report refresh must never predict")

    monkeypatch.setattr(VietnameseRuleBased, "predict", no_prediction)
    assert finalize_evidence(config)["models"]["tfidf"]["neutral_as_negative"] == 1
    output = json.loads(evidence.read_text(), parse_constant=no_prediction)
    assert output["slide_6_update"]["class_weight_comparison"][0]["class_weight"] == "None"
    assert finalize_evidence(config)["models"]["rule"]["errors"] == 1

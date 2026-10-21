import json
from unittest.mock import Mock

import pandas as pd
import pytest

from src.aspect_conditioned.analysis import conflict_mask, recover_classical
from src.aspect_conditioned.prompts import choose_examples, classical_input, input_payload
from src.aspect_conditioned.runner import (
    predict_partition,
    preserve_inference_protocol,
    validate_records,
)
from src.aspect_conditioned.runtime import AspectRuntime
from src.baselines.tfidf_logreg import build_pipeline
from src.proposed_v1.runner import digest, score_predictions, select_prompt


def rows():
    return pd.DataFrame(
        [
            {
                "review_id": "a",
                "text": "Đồ ăn ngon nhưng phục vụ chậm",
                "aspect": "food",
                "sentiment": "positive",
                "split": "train",
            },
            {
                "review_id": "a",
                "text": "Đồ ăn ngon nhưng phục vụ chậm",
                "aspect": "service",
                "sentiment": "negative",
                "split": "train",
            },
            {
                "review_id": "b",
                "text": "Giá khoảng hai mươi nghìn",
                "aspect": "price",
                "sentiment": "neutral",
                "split": "train",
            },
        ],
        index=[10, 11, 12],
    )


def test_only_whitelisted_aspect_input():
    assert input_payload("ngon", "food") == {"text": "ngon", "aspect": "food"}
    with pytest.raises(ValueError):
        input_payload("ngon", "gold_positive")
    assert classical_input("ngon", "food") == "__aspect_food__ ngon"


def test_train_only_selector_deterministic_conflicting_pair():
    train = rows()
    examples = choose_examples(train)
    assert examples == choose_examples(train.iloc[::-1])
    assert examples[0]["text"] == examples[1]["text"]
    assert examples[0]["aspect"] != examples[1]["aspect"]
    assert examples[0]["sentiment"] != examples[1]["sentiment"]
    train.loc[10, "split"] = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        choose_examples(train)


def test_aspect_token_preserved_in_tfidf_vocabulary():
    model = build_pipeline({"ngram_range": [1, 2], "c": 1})
    model.fit(
        [classical_input("ngon ngon", "food")] * 2 + [classical_input("chậm chậm", "service")] * 2,
        ["positive"] * 2 + ["negative"] * 2,
    )
    assert "__aspect_food__" in model.named_steps["tfidf"].vocabulary_
    assert model.named_steps["classifier"].class_weight == "balanced"
    assert model.named_steps["classifier"].random_state == 42


def test_runtime_transport_supplies_text_and_aspect_only_and_restores_session(tmp_path):
    runtime = AspectRuntime(
        {"port": 1, "max_attempts": 2, "temperature": 0, "seed": 42, "max_new_tokens": 32},
        {},
        tmp_path,
    )
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"sentiment":"positive"}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7},
    }
    session = Mock()
    session.post.return_value = response
    runtime.session = session
    result = runtime.predict_aspect("instructions", "review", "food")
    payload = session.post.call_args.kwargs["json"]
    assert json.loads(payload["messages"][1]["content"]) == {"text": "review", "aspect": "food"}
    assert runtime.session is session
    assert result["prediction"] == "positive"
    assert result["input_tokens"] == 10


def test_checkpoint_binds_aspect_not_just_text():
    data = rows().iloc[:1]
    record = pd.DataFrame(
        [
            {
                "sample_index": 10,
                "protocol_hash": "p",
                "input_sha256": digest({"text": data.iloc[0].text, "aspect": "food"}),
            }
        ]
    )
    assert "sample_index" in validate_records(record, data, "p").columns
    data.loc[10, "aspect"] = "service"
    with pytest.raises(ValueError, match="input mismatch"):
        validate_records(record, data, "p")


def test_inflight_test_request_is_not_repeated(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"pending_sample": 10}), encoding="utf-8")
    client = Mock()
    with pytest.raises(ValueError, match="in-flight"):
        predict_partition(client, "prompt", rows(), tmp_path / "pred.jsonl", "p", state)
    client.predict_aspect.assert_not_called()


def test_conflict_membership_and_historical_recovery_without_inference():
    test = rows()
    assert conflict_mask(test).tolist() == [True, True, False]
    failures = test.rename(columns={"sentiment": "true_label"}).iloc[:1].copy()
    failures["tfidf_prediction"] = "negative"
    expected = ["negative", "negative", "neutral"]
    recorded, _ = score_predictions(test.sentiment, expected)
    assert recover_classical(test, failures, recorded) == expected
    with pytest.raises(ValueError, match="disagree"):
        recover_classical(test, failures, {"accuracy": 1.0})


def test_selection_uses_validation_macro_f1():
    candidates = [
        {
            "prompt": "b1",
            "macro_f1": 0.7,
            "json_validity_rate": 1,
            "mean_latency_ms": 10,
            "test_macro_f1": 0.1,
        },
        {
            "prompt": "b2",
            "macro_f1": 0.6,
            "json_validity_rate": 1,
            "mean_latency_ms": 1,
            "test_macro_f1": 0.99,
        },
    ]
    assert select_prompt(candidates)["prompt"] == "b1"


def test_report_only_recovery_pins_sources_and_preserves_inference(tmp_path):
    original = {"source_hashes": {"analysis.py": "old"}, "protocol_hash": "inference", "temperature": 0}
    computed = {**original, "source_hashes": {"analysis.py": "new"}, "protocol_hash": "new"}
    (tmp_path / "protocol.json").write_text(json.dumps(original))
    with pytest.raises(ValueError):
        preserve_inference_protocol(tmp_path, computed)
    (tmp_path / "test_measurements.json").write_text("{}")
    audit = {"source_hashes": computed["source_hashes"], "inference_protocol_hash": "inference", "new_inference_calls": 0}
    (tmp_path / "posttest_reporting_recovery.json").write_text(json.dumps(audit))
    assert preserve_inference_protocol(tmp_path, computed) == original
    with pytest.raises(ValueError, match="inputs/settings"):
        preserve_inference_protocol(tmp_path, {**computed, "temperature": 1})
    with pytest.raises(ValueError, match="source changes"):
        preserve_inference_protocol(tmp_path, {**computed, "source_hashes": {"runtime.py": "new"}})


def test_posthoc_reindex_preserves_sample_identity():
    source = pd.DataFrame({"sample_index": [10, 11], "prediction": ["positive", "negative"]}).set_index("sample_index")
    selected = source.loc[pd.Index([11, 10])].rename_axis("sample_index").reset_index()
    assert selected.sample_index.tolist() == [11, 10]
    assert selected.prediction.tolist() == ["negative", "positive"]

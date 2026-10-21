import json

import pandas as pd
import pytest
from pydantic import ValidationError

from src.proposed_v1.runner import (
    freeze_prompts,
    predict_partition,
    score_predictions,
    select_prompt,
)
from src.proposed_v1.runtime import LocalQwen, messages, parse_output
from src.storage.manifest import atomic_json


@pytest.mark.parametrize("label", ["negative", "neutral", "positive"])
def test_strict_json_labels(label):
    assert parse_output(json.dumps({"sentiment": label})) == label


@pytest.mark.parametrize(
    "raw",
    [
        "positive",
        '```json\n{"sentiment":"positive"}\n```',
        '{"sentiment":"POSITIVE"}',
        '{"sentiment":"positive","explanation":"x"}',
        '{"sentiment":"neutral","sentiment":"positive"}',
        '{"sentiment":null}',
        "[]",
    ],
)
def test_invalid_output_never_silently_mapped(raw):
    with pytest.raises((ValueError, ValidationError)):
        parse_output(raw)


def test_request_contains_text_only():
    request = messages("instruction", "unit review")
    assert json.loads(request[1]["content"]) == {"text": "unit review"}
    assert "aspect" not in request[1]["content"] and "split" not in request[1]["content"]


def test_prompt_examples_train_only_and_hashed(tmp_path):
    prompt_dir = tmp_path / "prompts"
    root = tmp_path / "reports"
    prompt_dir.mkdir()
    root.mkdir()
    for name in ("v1_a", "v1_b", "v1_c"):
        (prompt_dir / f"{name}.txt").write_text(
            "instruction {{TRAIN_EXAMPLES}}" if name == "v1_c" else "instruction"
        )
    rows = [
        {
            "review_id": label,
            "text": f"Train-only unit example for label {label} with enough length",
            "sentiment": label,
            "split": "train",
        }
        for label in ("negative", "neutral", "positive")
    ]
    rows += [
        {
            "review_id": "validation",
            "text": "HELDOUT_SECRET",
            "sentiment": "neutral",
            "split": "validation",
        }
    ]
    prompts, manifest = freeze_prompts(pd.DataFrame(rows), prompt_dir, root)
    assert len(manifest["few_shot_examples"]) == 3
    assert all(row["origin"] == "train" for row in manifest["few_shot_examples"])
    assert "HELDOUT_SECRET" not in prompts["v1_c"]["text"]
    assert all(len(row["sha256"]) == 64 for row in prompts.values())


def test_validation_selection_and_tie_breaks():
    rows = [
        {
            "prompt": "a",
            "macro_f1": 0.6,
            "json_validity_rate": 0.9,
            "mean_latency_ms": 1,
            "test_macro_f1": 1,
        },
        {
            "prompt": "b",
            "macro_f1": 0.6,
            "json_validity_rate": 1,
            "mean_latency_ms": 4,
            "test_macro_f1": 0,
        },
        {
            "prompt": "c",
            "macro_f1": 0.6,
            "json_validity_rate": 1,
            "mean_latency_ms": 3,
            "test_macro_f1": 0,
        },
    ]
    assert select_prompt(rows)["prompt"] == "c"


def test_invalid_terminal_output_counts_as_error():
    metrics, report = score_predictions(
        ["positive", "neutral", "negative"], ["positive", None, "negative"]
    )
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert report["neutral"]["recall"] == 0


def test_completed_rows_checkpointed_no_second_inference(tmp_path):
    class Spy:
        def __init__(self):
            self.calls = 0

        def predict(self, instruction, text):
            self.calls += 1
            return {
                "prediction": "neutral",
                "json_valid": True,
                "first_attempt_valid": True,
                "invalid_outputs": 0,
                "input_tokens": 2,
                "output_tokens": 3,
                "latency_ms": 1,
            }

    spy = Spy()
    frame = pd.DataFrame({"text": ["unit_a", "unit_b"]})
    state_path = tmp_path / "state.json"
    atomic_json(state_path, {"pending_sample": None})
    path = tmp_path / "checkpoint.jsonl"
    records, _ = predict_partition(spy, "instruction", frame, path, "unit", state_path)
    assert records.sample_index.tolist() == [0, 1]
    assert (
        len(
            frame.reset_index()
            .rename(columns={"index": "sample_index"})
            .merge(records, on="sample_index", validate="one_to_one")
        )
        == 2
    )
    predict_partition(spy, "instruction", frame, path, "unit", state_path)
    assert spy.calls == 2


def test_unknown_inflight_test_request_cannot_repeat(tmp_path):
    atomic_json(tmp_path / "state.json", {"pending_sample": 0})
    with pytest.raises(ValueError, match="interrupted"):
        predict_partition(
            None,
            "instruction",
            pd.DataFrame({"text": ["unit"]}),
            tmp_path / "checkpoint.jsonl",
            "unit",
            tmp_path / "state.json",
        )


def test_invalid_json_retry_and_accounting():
    client = LocalQwen(
        {"port": 8193, "max_attempts": 2, "temperature": 0, "seed": 42, "max_new_tokens": 32}, {}
    )

    class Response:
        def __init__(self, raw):
            self.raw = raw

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": self.raw}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, url, json, timeout):
            self.calls.append(json)
            return Response("free text" if len(self.calls) == 1 else '{"sentiment":"neutral"}')

    session = Session()
    client.session.close()
    client.session = session
    result = client.predict("instruction", "unit text")
    assert result["prediction"] == "neutral" and result["invalid_outputs"] == 1
    assert result["attempts"] == 2 and result["input_tokens"] == 20
    assert not result["first_attempt_valid"]

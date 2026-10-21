import json

import pandas as pd
import pytest
import yaml

from src.proposed_v1.runtime import messages
from src.proposed_v2.analysis import diagnostic_rows, protocol_documents
from src.proposed_v2.examples import choose_examples, freeze_examples_and_prompts
from src.proposed_v2.runner import (
    assert_frozen,
    choose_candidate,
    join_predictions,
    verified_records,
)
from src.utils.hashing import file_checksum


def train_rows():
    examples = [
        ("Rất ngon và phục vụ tuyệt vời", "positive"),
        ("Đồ ăn rất tệ và quá chán", "negative"),
        ("Quán ở đường số năm", "neutral"),
        ("Món này khá ổn và giá hợp lý", "positive"),
        ("Đồ ăn hơi nguội và hơi nhạt", "negative"),
        ("Món ngon nhưng phục vụ quá tệ", "negative"),
        ("Không thể chê món ăn này", "positive"),
        ("Bình thường thôi", "negative"),
        ("Địa chỉ ở tầng hai của tòa nhà", "neutral"),
    ]
    return pd.DataFrame(
        [
            {"review_id": f"r{i:02}", "text": text, "sentiment": label, "split": "train"}
            for i, (text, label) in enumerate(examples)
        ]
    )


def test_selector_train_only_reproducible_all_contrasts():
    rows = train_rows()
    first = choose_examples(rows)
    assert first == choose_examples(rows.sample(frac=1, random_state=3))
    assert len(first) == 9 and len({e["text"] for e in first}) == 9
    assert {e["category"] for e in first} == {
        "clear_positive",
        "clear_negative",
        "true_neutral",
        "mild_positive",
        "mild_negative",
        "mixed_sentiment",
        "negation",
        "short_ambiguous",
        "neutral_contrast",
    }
    assert all(e["split"] == "train" for e in first)
    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="TRAIN rows only"):
        choose_examples(rows)


def test_heldout_fallback_and_english_example_prohibited():
    rows = train_rows()
    rows.loc[8, "text"] = "This is a factual address without Vietnamese diacritics"
    with pytest.raises(ValueError, match="no held-out fallback"):
        choose_examples(rows)


def test_prompt_counts_hashes_and_inference_contract(tmp_path):
    settings = {
        "example_max_characters": 150,
        "prompt_dir": str(tmp_path / "prompts"),
        "candidate_names": ["v2_a1", "v2_a2", "v2_a3"],
    }
    prompts, manifest = freeze_examples_and_prompts(train_rows(), settings, tmp_path)
    assert [p["example_count"] for p in prompts.values()] == [6, 9, 8]
    assert all(len(p["sha256"]) == 64 for p in prompts.values())
    for p in prompts.values():
        request = messages(p["text"], "current review")
        assert json.loads(request[1]["content"]) == {"text": "current review"}
        assert (
            "review_id" not in request[0]["content"] and "gold_label" not in request[0]["content"]
        )
    assert freeze_examples_and_prompts(train_rows(), settings, tmp_path)[1] == manifest


def test_macro_f1_selection_beats_neutral_recall_only():
    rows = [
        {
            "prompt": "recall_only",
            "macro_f1": 0.5,
            "neutral_precision": 0.1,
            "neutral_recall": 1,
            "json_validity_rate": 1,
            "mean_latency_ms": 1,
        },
        {
            "prompt": "better_macro",
            "macro_f1": 0.7,
            "neutral_precision": 0.4,
            "neutral_recall": 0.4,
            "json_validity_rate": 1,
            "mean_latency_ms": 3,
        },
    ]
    assert choose_candidate(rows)["prompt"] == "better_macro"
    rows[0]["macro_f1"] = 0.7
    assert choose_candidate(rows)["prompt"] == "better_macro"


def test_join_preserves_nonconsecutive_indices_and_gold():
    rows = pd.DataFrame(
        {"text": ["unit a", "unit b"], "sentiment": ["positive", "negative"]}, index=[10, 30]
    )
    records = pd.DataFrame({"sample_index": [30, 10], "prediction": ["neutral", "positive"]})
    joined = join_predictions(rows, records)
    assert joined.sample_index.tolist() == [10, 30]
    assert joined.true_label.tolist() == ["positive", "negative"]
    assert joined.prediction.tolist() == ["positive", "neutral"]


def test_frozen_inputs_audited_without_writing(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"frozen")
    audit = {str(path): file_checksum(path)}
    assert_frozen(audit)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="protected"):
        assert_frozen(audit)


def test_incomplete_or_duplicate_checkpoint_not_finalized(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        json.dumps({"sample_index": 0, "text_sha256": "x", "protocol_hash": "unit"}) + "\n"
    )
    rows = pd.DataFrame({"text": ["a", "b"]})
    with pytest.raises(ValueError, match="membership"):
        verified_records(path, rows, "unit")


def test_posthoc_flags_not_inference_features():
    rows = pd.DataFrame(
        {
            "text": ["ngon nhưng chậm", "không ngon"],
            "true_label": ["positive", "negative"],
            "prediction": ["neutral", "negative"],
            "sarcasm": [0, 1],
            "aspect": ["food", "service"],
        }
    )
    flagged, cohorts, _, confusions = diagnostic_rows(rows)
    assert confusions["positive_to_neutral"] == 1
    assert flagged.mixed_proxy.tolist() == [True, False]
    assert any(row["cohort"] == "sarcasm_gold" for row in cohorts)


def test_external_and_aspect_protocol_proposals_only(tmp_path):
    protocol_documents(tmp_path)
    assert "DISTINCT" in (tmp_path / "aspect_conditioned_protocol.md").read_text()
    assert (
        "no download/evaluation" in (tmp_path / "external_evaluation_recommendation.md").read_text()
    )
    stages = yaml.safe_load(open("dvc.yaml", encoding="utf-8"))["stages"]
    assert stages["proposed_v1"]["frozen"] is True
    assert "proposed_v2" in stages

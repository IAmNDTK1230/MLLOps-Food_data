import json
import unicodedata
from argparse import Namespace

import pandas as pd
import pytest
import yaml

from src.food.adapters import NTCSCVAdapter, VLSP2018RestaurantAdapter, adapt_vn_msasar
from src.food.analysis import analyze, chunk_count, rag_documents, token_analysis
from src.food.config import FoodConfig
from src.food.data import import_dataset, load_prepared, prepare
from src.food.interfaces import StructuredReviewAnalysis
from src.food.split import assign_splits, leakage_groups, load_splits, split_dataset
from src.ml.cli import DISABLED, run_command


@pytest.fixture
def config(tmp_path):
    return FoodConfig(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports")


def source_rows(count=60):
    # Unit-only strings; no estimators or Qwen are fitted/executed.
    return pd.DataFrame(
        [
            {
                "review_id": f"r{i}",
                "sentence": f"Unit sentence {i}",
                "aspect": aspect,
                "sentiment": (-3, 0, 2)[i % 3],
                "sarcasm": i % 2,
                "source": "unit",
                "language": "vi",
                "place_type": "restaurant",
            }
            for i in range(count)
            for aspect in ("food", "service")
        ]
    )


def imported(config, tmp_path):
    path = tmp_path / "unit.parquet"
    source_rows().to_parquet(path, index=False)
    import_dataset(config, input_path=path)
    return prepare(config)


def test_adapter_gold_labels_filters_cleaning_rejections_duplicates():
    raw = source_rows(3)
    raw.loc[0, "sentence"] = unicodedata.normalize("NFD", "  Món ăn\n Ngon  ")
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    for updates in (
        {"language": "en"},
        {"place_type": "hotel"},
        {"sentence": " "},
        {"sentiment": 8},
        {"sarcasm": 3},
    ):
        row = source_rows(1).iloc[0].to_dict() | updates
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
    provenance = {
        "import_timestamp": "2026-10-13T00:00:00+00:00",
        "license": "test",
        "source_url": "https://example.org",
        "license_url": "https://example.org/license",
    }
    frame, rejected, stats = adapt_vn_msasar(raw, provenance)
    assert stats == {
        "raw_rows": 12,
        "filtered_vietnamese_restaurant_rows": 10,
        "filtered_out_rows": 2,
        "valid_rows": 6,
        "rejected_rows": 3,
        "duplicates_removed": 1,
    }
    assert set(frame.sentiment) == {"negative", "neutral", "positive"}
    assert set(frame.original_sentiment) == {-3, 0, 2}
    assert frame.restaurant_id.isna().all() and "rating" not in frame
    assert "Món ăn Ngon" in frame.text.tolist()
    assert raw.iloc[0].sentence in frame.original_text.tolist()
    assert set(frame.aspect) == {"food", "service"}
    assert set(row["reason"] for row in rejected) == {"empty_text", "invalid_label"}


def test_missing_required_schema_is_explicit():
    with pytest.raises(ValueError, match="required columns"):
        adapt_vn_msasar(pd.DataFrame({"rating": [5]}), {})


def test_failed_download_provides_manual_instructions_without_substitution(config, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("unit network failure")

    monkeypatch.setattr("src.food.data.hub_metadata", unavailable)
    with pytest.raises(ValueError, match="No other dataset has been substituted") as error:
        import_dataset(config)
    assert "food-data-import --dataset vn_msasar_v3 --input" in str(error.value)
    assert not (config.data_dir / "source/vn_msasar_v3/provenance.json").exists()


def test_manual_schema_mismatch_cannot_be_imported_as_food(config, tmp_path):
    path = tmp_path / "not_v3.csv"
    pd.DataFrame({"comment": ["unit"], "rating": [5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="required columns missing"):
        import_dataset(config, input_path=path)


def test_import_provenance_and_tamper_checks(config, tmp_path):
    metadata = imported(config, tmp_path)
    assert metadata["raw_rows"] == 120 and metadata["processed_sha256"]
    assert metadata["files"][0]["raw_sha256"]
    frame, _ = load_prepared(config)
    assert len(frame) == 120 and frame.dataset_name.eq("vn_msasar_v3").all()
    mapping = json.loads((config.reports_dir / "food_label_mapping.json").read_text())
    assert mapping["ratings_used"] is False
    assert mapping["mapping"]["-1"] == "negative" and mapping["mapping"]["0"] == "neutral"
    path = config.data_dir / "processed/reviews.parquet"
    path.write_bytes(b"edited")
    with pytest.raises(ValueError, match="changed"):
        load_prepared(config)


def test_raw_tamper_and_missing_source_fail(config, tmp_path):
    with pytest.raises(ValueError, match="Real VN-MSASar"):
        prepare(config)
    imported(config, tmp_path)
    (config.data_dir / "source/vn_msasar_v3/unit.parquet").write_bytes(b"edited")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(config)


def test_transitive_review_and_text_grouping():
    frame = pd.DataFrame(
        {"review_id": ["a", "a", "b", "c"], "text": ["text1", "text2", " text2 ", "text3"]}
    )
    groups = leakage_groups(frame)
    assert groups[0] == groups[1] == groups[2] and groups[2] != groups[3]


def test_shared_deterministic_splits_no_leakage(config, tmp_path):
    imported(config, tmp_path)
    frame, _ = load_prepared(config)
    first, metadata = assign_splits(frame, config)
    second, _ = assign_splits(frame, config)
    assert first.split.tolist() == second.split.tolist()
    assert first.groupby("review_id").split.nunique().max() == 1
    assert first.groupby("text").split.nunique().max() == 1
    assert metadata["requested_fractions"] == {"train": 0.7, "validation": 0.15, "test": 0.15}
    assert metadata["shared_by"] == ["baseline_rule", "baseline_tfidf", "proposed_v1"]
    assert metadata["leakage_checks"] == {
        "review_ids_spanning_splits": 0,
        "normalized_texts_spanning_splits": 0,
    }
    split_dataset(config)
    restored, _, _ = load_splits(config)
    assert restored.split.tolist() == first.split.tolist()
    with pytest.raises(ValueError, match="changed"):
        load_splits(config.model_copy(update={"random_seed": 3}))


@pytest.mark.parametrize(
    "tokens,size,overlap,expected",
    [
        (0, 128, 0, 0),
        (128, 128, 32, 1),
        (129, 128, 0, 2),
        (224, 128, 32, 2),
        (225, 128, 32, 3),
        (512, 512, 64, 1),
    ],
)
def test_chunk_count_boundaries(tokens, size, overlap, expected):
    assert chunk_count(tokens, size, overlap) == expected


def test_rag_metadata_keeps_aspect_sentiment_alignment(config, tmp_path):
    imported(config, tmp_path)
    split_dataset(config)
    frame, _, _ = load_splits(config)
    reviews, sentences = rag_documents(frame)
    assert len(reviews) == len(sentences) == 60
    assert len(reviews[0]["annotations"]) == 2
    assert {"review_id", "source", "aspect", "sentiment", "dataset_name", "split"} <= reviews[
        0
    ].keys()
    report, comparison = token_analysis(
        reviews, sentences, lambda text: text.split(), config, {"test_tokenizer": True}
    )
    assert report["statistics"]["sentence_level"]["document_count"] == 60
    assert report["selected_chunk_size"] is None
    assert len(comparison) == 20  # two intact strategies + 2 bases * 3 sizes * 3 overlaps


def test_analysis_outputs_actual_unit_counts_no_model_results(config, tmp_path):
    imported(config, tmp_path)
    split_dataset(config)
    analyze(config, encoder=lambda text: text.split(), tokenizer_metadata={"unit_only": True})
    report = json.loads((config.reports_dir / "food_dataset_report.json").read_text())
    assert report["valid_rows"] == 120 and report["unique_reviews"] == 60
    assert report["sentiment_counts"] == {"negative": 40, "neutral": 40, "positive": 40}
    assert report["imbalance_after"] is None and report["no_training_executed"]
    assert (config.reports_dir / "sentiment_distribution.png").stat().st_size > 100
    assert (config.reports_dir / "task2_slide_evidence.json").exists()


@pytest.mark.parametrize("command", sorted(DISABLED))
def test_legacy_model_commands_are_disabled_before_any_fit(command, capsys):
    assert run_command(Namespace(command=command)) == 1
    assert "disabled" in capsys.readouterr().out


def test_external_datasets_never_primary(config):
    with pytest.raises(ValueError, match="cannot be merged"):
        import_dataset(config, dataset="ntc_scv")
    for adapter in (NTCSCVAdapter(), VLSP2018RestaurantAdapter()):
        with pytest.raises(NotImplementedError, match="external evaluation only"):
            adapter.adapt(pd.DataFrame(), {})


def test_dvc_only_data_stages_and_food_defaults():
    config = FoodConfig.load()
    assert config.data_dir.as_posix() == "data/food" and config.random_seed == 42
    stages = yaml.safe_load(open("dvc.yaml", encoding="utf-8"))["stages"]
    assert set(stages) == {
        "food_import",
        "food_prepare",
        "food_split",
        "food_analysis",
        "baseline_rule",
        "baseline_tfidf",
        "baseline_evaluate",
        "proposed_v1",
        "proposed_v2",
        "proposed_v2_aspect",
    }
    assert all(
        stages[name]["frozen"]
        for name in ("food_import", "food_prepare", "food_split", "food_analysis")
    )
    assert all("train-baseline" not in stage["cmd"] for stage in stages.values())
    result = StructuredReviewAnalysis(sentiment="positive", aspect="food")
    assert result.food_quality is None and result.topics == []

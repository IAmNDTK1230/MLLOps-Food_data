import json

import pandas as pd
import pytest

from src.foody_v2_audit import (
    fraction,
    parse_dates,
    readiness,
    restaurant_audit,
    search_provenance,
    text_quality,
)


def test_utf8_quality_distinguishes_replacement_from_question_punctuation():
    comments = pd.Series(
        [
            "Đồ ăn ngon 👍",
            "Có ngon không???",
            "sao?Quay lại",
            "m.me/test?ref=abc",
            "thiếu \ufffd",
            "",
        ]
    )
    quality, flags, cases = text_quality(comments)
    assert quality["nonempty_comments"] == 5
    assert quality["u_fffd_occurrences"] == 1
    assert quality["comments_with_word_internal_question"] == 2
    assert quality["broad_corruption_indicator_rate"] == fraction(3, 5)
    assert quality["replacement_affected_rate"] == fraction(1, 5)
    assert int(flags.replacement.sum()) == 1
    assert len(cases) == 3
    assert comments.iloc[0] == "Đồ ăn ngon 👍"


def test_duplicate_restaurant_headers_are_not_silently_coalesced():
    frame = pd.DataFrame(
        {
            "RestaurantID": ["1", "1", "2"],
            "RestaurantID.1": ["1", "2", "2"],
            "Unnamed: 0": ["100", "101", "102"],
            "Comment": ["ngon", "ngon", "tốt"],
        }
    )
    report, stats, conflicts = restaurant_audit(frame)
    assert report["conflicting_rows"] == 1
    assert not report["canonical_business_key_selected"]
    assert conflicts.tolist() == [False, True, False]
    assert set(stats.candidate_column) == {"RestaurantID", "RestaurantID.1"}
    assert stats.groupby("candidate_column").review_count.sum().tolist() == [3, 3]
    assert report["candidates"][0]["restaurant_comment_duplicate_extra_nonempty"] == 1
    assert report["candidates"][1]["restaurant_comment_duplicate_extra_nonempty"] == 0


def test_explicit_date_suffix_parsing_keeps_raw_values():
    raw = pd.Series(["31/03/2022 12:02:37,", "07/11/2020 12:51", "31/02/2022 12:00", ""])
    original = raw.copy()
    parsed, missing, invalid, report = parse_dates(raw)
    assert parsed.iloc[0].second == 37
    assert parsed.iloc[1].month == 11
    assert report["valid_timestamps"] == 2
    assert report["missing_timestamps"] == 1
    assert report["invalid_nonempty_timestamps"] == 1
    assert report["format_counts"]["second_trailing_comma"] == 1
    assert raw.equals(original)
    assert missing.iloc[3] and invalid.iloc[2]


def test_unknown_labels_do_not_block_demo_rag_after_cleaning():
    quality = {"broad_corruption_indicator_rate": fraction(2, 10000)}
    state = readiness(quality, True, 0, "UNKNOWN")
    assert state["training_readiness"] == "NOT_READY_LABEL_PROVENANCE"
    assert state["demo_readiness"] == state["rag_readiness"] == "READY_AFTER_CLEANING"
    assert (
        readiness(quality, True, 10, "UNKNOWN")["training_readiness"] == "NOT_READY_MULTIPLE_ISSUES"
    )
    assert readiness(quality, False, 0, "UNKNOWN")["rag_readiness"] == "NOT_READY_TEXT_CORRUPTION"


def test_provenance_search_does_not_turn_correlation_or_reader_code_into_derivation(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/reader.py").write_text('labels = frame["Classification"]\n', encoding="utf-8")
    result = search_provenance(tmp_path)
    assert result["provenance"] == "UNKNOWN"
    assert result["creator_function_found"] is False
    assert result["matches"][0]["line"] == 1
    assert len(result["searched_files"]) == 1


def test_indicator_rates_expose_denominators_without_fabricated_zero():
    assert fraction(0, 0)["percentage"] is None
    assert fraction(7, 15640)["percentage"] == pytest.approx(0.04475703324808184)
    # Evidence exports remain strict JSON, including unavailable percentages.
    assert json.loads(json.dumps(fraction(0, 0), allow_nan=False))["percentage"] is None

import csv
import hashlib
import io
import json

import pandas as pd
import pytest

from src.foody_audit import duplicate_counts, inspect_encoding, run, timestamp_audit


def source_csv(rows, encoding="utf-8"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "Unnamed: 0",
            "UserID",
            "User",
            "Review Time",
            "Rating",
            "Comment",
            "RestaurantID",
            "Classification",
        ]
    )
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def execute(tmp_path, raw):
    source = tmp_path / "source" / "review.csv"
    source.parent.mkdir()
    source.write_bytes(raw)
    result = run(source, tmp_path / "reports", tmp_path / "audit", "2026-10-20")
    assert source.read_bytes() == raw
    assert result["source_sha256"] == hashlib.sha256(raw).hexdigest()
    return result


def test_utf8_vietnamese_is_not_misclassified_due_to_excel(tmp_path):
    raw = source_csv(
        [
            [0, 1, "A", "31/03/2022 12:02", "8.40", "Đồ ăn ngon, phục vụ tốt!", 1, 1],
            [1, 2, "B", "01/04/2022 12:00", "10", "Có thật không??", 2, 1],
        ]
    )
    result = execute(tmp_path, raw)
    assert result["encoding"]["encoding_status"] == "VALID_UTF8"
    assert result["encoding"]["comments_with_retained_accented_latin_characters"] == 2
    assert not result["encoding"]["substantial_text_loss_evidence"]
    assert result["training_readiness"] == "NOT_READY_LABEL_PROVENANCE"
    assert pd.read_parquet(tmp_path / "audit/row_quality.parquet").raw_rating.tolist() == [
        "8.40",
        "10",
    ]


def test_legacy_loss_indicators_and_unknown_labels_block_training(tmp_path):
    raw = source_csv(
        [
            [0, 1, "A", "31/03/2022 12:02", "0", "Trái cây t??i và ngon", 1, -1],
            [0, 2, "B", "", "", "s? ?ng h? ti?p", 1, -1],
        ],
        "cp1252",
    )
    result = execute(tmp_path, raw)
    assert not result["encoding"]["utf8_strict_valid"]
    assert result["encoding"]["inspection_view_byte_roundtrip"]
    assert result["encoding"]["comments_with_word_internal_question"] == 2
    assert result["encoding"]["inspection_view_u_fffd_count"] == 0
    assert result["training_readiness"] == "NOT_READY_MULTIPLE_ISSUES"
    assert result["demo_rag_readiness"] == "NOT_READY_FOR_TEXT_RAG"
    assert result["ratings"]["zero_rows"] == 1
    assert result["ratings"]["missing_rows"] == 1
    assert "Trái cây t??i" in (tmp_path / "reports/encoding_report.json").read_text(
        encoding="utf-8"
    )


def test_legacy_encoding_alone_is_not_proof_of_corruption(tmp_path):
    result = execute(
        tmp_path, source_csv([[0, 1, "A", "01/01/2020 12:00", "10", "Café ngon", 1, 1]], "cp1252")
    )
    assert not result["encoding"]["utf8_strict_valid"]
    assert not result["encoding"]["substantial_text_loss_evidence"]
    assert result["training_readiness"] == "NOT_READY_LABEL_PROVENANCE"


def test_utf8_bom_and_actual_replacement_character_distinguished():
    _, normal = inspect_encoding("\ufeffĐồ ăn ngon".encode())
    assert normal["utf8_bom_present"] and normal["utf8_strict_valid"]
    _, damaged = inspect_encoding("Đồ \ufffd".encode())
    assert damaged["raw_utf8_replacement_sequence_count"] == 1
    assert damaged["inspection_view_u_fffd_count"] == 1


def test_duplicate_denominators_distinguish_missing_keys():
    frame = pd.DataFrame({"Comment": ["same", "same", "", ""], "UserID": ["1", "2", "3", "4"]})
    result = duplicate_counts(frame, ["Comment"])
    assert result["all_rows_missing_keys_compared_as_equal"]["extra_rows_after_first"] == 2
    assert result["complete_keys_only"]["extra_rows_after_first"] == 1
    assert result["complete_keys_only"]["rows_in_duplicate_groups"] == 2
    assert (
        duplicate_counts(frame, ["UserID", "Comment"])["complete_keys_only"][
            "extra_rows_after_first"
        ]
        == 0
    )


def test_dates_missing_and_invalid_are_retained_separately():
    parsed, missing, invalid, result = timestamp_audit(
        pd.Series(["03/04/2020 12:30", "", "31/02/2020 12:30"])
    )
    assert parsed.iloc[0].month == 4
    assert missing.tolist() == [False, True, False]
    assert invalid.tolist() == [False, False, True]
    assert result["valid_timestamps"] == 1
    assert result["missing_timestamps"] == 1
    assert result["invalid_nonempty_timestamps"] == 1
    assert result["unparsed_total"] == 2


def test_label_association_is_not_presented_as_annotation_provenance(tmp_path):
    result = execute(
        tmp_path,
        source_csv(
            [
                [0, 1, "A", "01/01/2020 12:00", "6.0", "Đồ ăn", 1, -1],
                [1, 2, "B", "01/01/2020 12:00", "6.0", "Khác", 2, 1],
            ]
        ),
    )
    assert result["labels"]["rating_values_with_multiple_classifications"] == [6.0]
    assert result["labels"]["label_provenance"] == "UNKNOWN"
    assert not result["labels"]["manual_annotation_evidence"]
    assert (
        sum(
            sum(row["classification_counts"].values())
            for row in result["labels"]["rating_band_classification_counts"]
        )
        == 2
    )
    audit = pd.read_parquet(tmp_path / "audit/row_quality.parquet")
    assert len(audit) == 2
    assert audit.raw_classification.tolist() == ["-1", "1"]
    json.loads((tmp_path / "reports/audit_summary.json").read_text(encoding="utf-8"))


def test_outputs_cannot_contain_immutable_source(tmp_path):
    source = tmp_path / "review.csv"
    source.write_bytes(b"untouched")
    with pytest.raises(ValueError, match="immutable"):
        run(source, tmp_path, tmp_path / "audit", "2026-10-20")
    assert source.read_bytes() == b"untouched"

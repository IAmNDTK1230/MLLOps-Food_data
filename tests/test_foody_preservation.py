import csv
import hashlib
import io
import json

import pytest

from scripts.preserve_foody_source import preserve
from src.foody_cleaning_preflight import require_canonical_restaurant_id, run


def example_bytes():
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
            "RestaurantID",
            "Classification",
        ]
    )
    writer.writerows(
        [
            [0, 7, "A", "31/03/2022 12:02:37,", "8.40", "Đồ ăn ngon 👍", 1, 2, 1],
            [1, 8, "B", "", "", "", 1, 3, -1],
            [2, 9, "C", "01/04/2022 12:00", "0", "có \ufffd", 1, 3, 0],
            [3, 10, "D", "01/04/2022 12:00", "10", "Đồ ăn ngon 👍", 2, 2, 1],
        ]
    )
    return stream.getvalue().encode("utf-8")


def test_snapshot_is_byte_identical_and_preserves_raw_fields(tmp_path):
    source, snapshot = tmp_path / "review.csv", tmp_path / "review_utf8_20261020.csv"
    raw = example_bytes()
    source.write_bytes(raw)
    before_stat = source.stat().st_mtime_ns
    result = preserve(source, snapshot, hashlib.sha256(raw).hexdigest())
    assert source.read_bytes() == snapshot.read_bytes() == raw
    assert source.stat().st_mtime_ns == before_stat
    assert result["size_bytes"] == len(raw)
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["snapshot_created_at_utc"] and result["snapshot_created_at_asia_ho_chi_minh"]
    records = list(csv.reader(io.StringIO(snapshot.read_text(encoding="utf-8"))))
    assert records[0].count("RestaurantID") == 2
    assert records[1][4] == "8.40" and records[2][4] == "" and records[3][4] == "0"
    assert [r[-1] for r in records[1:]] == ["1", "-1", "0", "1"]
    assert records[1][3] == "31/03/2022 12:02:37,"
    assert "\ufffd" in records[3][5] and records[2][5] == ""


def test_snapshot_rerun_verifies_without_rewriting_manifest_or_copy(tmp_path):
    source, snapshot = tmp_path / "review.csv", tmp_path / "snapshot.csv"
    source.write_bytes(example_bytes())
    original = preserve(source, snapshot)
    manifest = snapshot.with_suffix(".manifest.json")
    manifest_bytes, modified = manifest.read_bytes(), manifest.stat().st_mtime_ns
    snapshot_modified = snapshot.stat().st_mtime_ns
    assert preserve(source, snapshot) == original
    assert manifest.read_bytes() == manifest_bytes and manifest.stat().st_mtime_ns == modified
    assert snapshot.stat().st_mtime_ns == snapshot_modified


def test_existing_snapshot_is_never_overwritten_on_source_change(tmp_path):
    source, snapshot = tmp_path / "review.csv", tmp_path / "snapshot.csv"
    source.write_bytes(example_bytes())
    preserve(source, snapshot)
    snapshot_bytes = snapshot.read_bytes()
    manifest_bytes = snapshot.with_suffix(".manifest.json").read_bytes()
    source.write_bytes(b"new source contents")
    with pytest.raises(ValueError, match="refusing overwrite"):
        preserve(source, snapshot)
    assert source.read_bytes() == b"new source contents"
    assert snapshot.read_bytes() == snapshot_bytes
    assert snapshot.with_suffix(".manifest.json").read_bytes() == manifest_bytes


def test_mismatched_expected_hash_does_not_create_snapshot(tmp_path):
    source, snapshot = tmp_path / "review.csv", tmp_path / "snapshot.csv"
    source.write_bytes(example_bytes())
    with pytest.raises(ValueError, match="expected audited SHA256"):
        preserve(source, snapshot, "0" * 64)
    assert not snapshot.exists() and not snapshot.with_suffix(".manifest.json").exists()
    assert source.read_bytes() == example_bytes()


def test_incomplete_snapshot_pair_is_not_silently_repaired(tmp_path):
    source, snapshot = tmp_path / "review.csv", tmp_path / "snapshot.csv"
    source.write_bytes(example_bytes())
    snapshot.write_bytes(example_bytes())
    with pytest.raises(ValueError, match="Incomplete snapshot"):
        preserve(source, snapshot)
    assert not snapshot.with_suffix(".manifest.json").exists()


def test_snapshot_cannot_replace_source(tmp_path):
    source = tmp_path / "review.csv"
    source.write_bytes(example_bytes())
    with pytest.raises(ValueError, match="must differ"):
        preserve(source, source)
    assert source.read_bytes() == example_bytes()


def test_invalid_utf8_is_not_silently_reencoded(tmp_path):
    source, snapshot = tmp_path / "review.csv", tmp_path / "snapshot.csv"
    source.write_bytes(b"caf\xe9")
    with pytest.raises(UnicodeDecodeError):
        preserve(source, snapshot)
    assert source.read_bytes() == b"caf\xe9" and not snapshot.exists()


@pytest.mark.parametrize(
    "decision",
    [
        {},
        {
            "resolution_status": "LIKELY",
            "canonical_column": "RestaurantID",
            "reason": "first column / fewer duplicates",
        },
        {"resolution_status": "RESOLVED_FROM_SOURCE_CODE", "canonical_column": "RestaurantID.1"},
    ],
)
def test_statistics_or_column_order_cannot_bypass_provenance_gate(decision):
    with pytest.raises(ValueError, match="processing must stop"):
        require_canonical_restaurant_id(decision)


def test_preflight_stops_without_creating_processed_data_or_renaming_labels(tmp_path):
    source = tmp_path / "data/foody/source/review.csv"
    source.parent.mkdir(parents=True)
    raw = example_bytes()
    source.write_bytes(raw)
    audit = tmp_path / "data/foody/audit_v2"
    audit.mkdir(parents=True)
    (audit / "source_manifest.json").write_text(
        json.dumps({"source_sha256": hashlib.sha256(raw).hexdigest()})
    )
    historical = tmp_path / "reports/foody_v2"
    historical.mkdir(parents=True)
    prior = {
        "restaurants": {"candidates": []},
        "date_audit": {
            "missing_timestamps": 1,
            "earliest": "2022-03-31T12:02:37",
            "latest": "2022-04-01T12:00:00",
        },
        "missing_ratings": 1,
        "rating_range": [0.0, 10.0],
    }
    (historical / "schema_report.json").write_text(json.dumps(prior))
    old_report = (historical / "schema_report.json").read_bytes()
    model = tmp_path / "models/tfidf_logreg.joblib"
    model.parent.mkdir()
    model.write_bytes(b"frozen benchmark model; must never be loaded or retrained")
    snapshot = source.with_name("review_utf8_20261020.csv")
    reports = tmp_path / "reports/foody_cleaning"
    metrics = run(tmp_path, source, snapshot, reports)
    assert metrics["execution_status"] == "BLOCKED"
    assert metrics["raw_rows"] == 4 and metrics["text_eligible_rows_before_id_validation"] == 2
    assert (
        metrics["processed_ai_valid_rows"] is None
        and metrics["processed_deduplicated_rows"] is None
    )
    assert not metrics["cleaning_executed"] and not metrics["processed_output_created"]
    assert not (tmp_path / "data/foody/processed/reviews.parquet").exists()
    assert metrics["demo_readiness"] == metrics["rag_readiness"] == "BLOCKED"
    decision = json.loads((reports / "restaurant_id_decision.json").read_text())
    assert decision["canonical_column"] is None
    assert decision["classification_role"] == "unknown_provenance_source_metadata"
    assert decision["classification_derived_name_planned"] == "source_classification"
    assert not decision["classification_rename_applied"]
    assert source.read_bytes() == snapshot.read_bytes() == raw
    assert (historical / "schema_report.json").read_bytes() == old_report
    assert model.read_bytes() == b"frozen benchmark model; must never be loaded or retrained"

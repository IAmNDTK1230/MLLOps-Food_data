"""Read-only reconciliation of UTF-8 audit, preserving duplicate header positions."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    root, audit = Path("reports/foody_v2"), Path("data/foody/audit_v2")
    manifest = json.loads((audit / "source_manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["actual_source_path"])
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.expected_sha256 == manifest["source_sha256"]
    parsed = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="strict"))))
    header, records = parsed[0], parsed[1:]
    schema = json.loads((root / "schema_report.json").read_text(encoding="utf-8"))
    encoding = json.loads((root / "encoding_report.json").read_text(encoding="utf-8"))
    assert header == schema["raw_header"]
    assert len(records) == schema["row_count"] == 15729
    assert all(len(r) == len(header) for r in records)
    rows = pd.read_parquet(audit / "row_quality.parquet")
    assert len(rows) == len(records)
    assert rows.source_record_number.tolist() == list(range(1, len(records) + 1))
    assert rows.raw_rating.tolist() == [r[header.index("Rating")] for r in records]
    assert rows.raw_review_time.tolist() == [r[header.index("Review Time")] for r in records]
    assert rows.raw_classification.tolist() == [r[header.index("Classification")] for r in records]
    ids = [n for n, name in enumerate(header) if name == "RestaurantID"]
    assert len(ids) == 2
    assert rows.RestaurantID.tolist() == [r[ids[0]] for r in records]
    assert rows["RestaurantID.1"].tolist() == [r[ids[1]] for r in records]
    mismatches = sum(r[ids[0]] != r[ids[1]] for r in records)
    assert (
        mismatches
        == schema["restaurants"]["conflicting_rows"]
        == int(rows.restaurant_id_conflict.sum())
    )
    comments = [r[header.index("Comment")] for r in records]
    assert sum(bool(c.strip()) for c in comments) == encoding["quality"]["nonempty_comments"]
    assert sum(c.count("\ufffd") for c in comments) == encoding["quality"]["u_fffd_occurrences"]
    assert sum("\ufffd" in c for c in comments) == encoding["quality"]["comments_with_u_fffd"]
    stats = pd.read_csv(root / "restaurant_statistics.csv")
    assert stats.groupby("candidate_column").review_count.sum().eq(len(records)).all()
    for name, sha in manifest["historical_reports_used_read_only"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
    assert hashlib.sha256(source.read_bytes()).hexdigest() == args.expected_sha256
    print(
        "PASS: strict UTF-8; 15,729 records and lexical raw fields preserved; both conflicting ID columns retained; source and historical evidence unchanged during audit."
    )


if __name__ == "__main__":
    main()

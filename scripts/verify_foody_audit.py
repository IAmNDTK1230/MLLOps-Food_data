"""Independently reconcile Foody reports using stdlib CSV parsing; no source writes."""

import argparse
import csv
import hashlib
import io
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    root = Path("reports/foody")
    summary = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
    source = Path(summary["source_path"])
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.expected_sha256 == summary["source_sha256"]
    decoded = raw.decode(summary["encoding"]["inspection_encoding"], errors="strict")
    records = list(csv.DictReader(io.StringIO(decoded)))
    count = len(records)
    assert count == summary["schema"]["row_count"]
    nonempty = [r["Comment"] for r in records if r["Comment"].strip()]
    assert len(nonempty) == summary["text"]["nonempty_comments"]
    assert len(set(nonempty)) == summary["text"]["unique_nonempty_comments"]
    assert (
        len(nonempty) - len(set(nonempty))
        == summary["text"]["exact_duplicate_nonempty_comments_extra_rows"]
    )
    lengths = list(map(len, nonempty))
    assert abs(statistics.mean(lengths) - summary["text"]["mean_characters"]) < 1e-9
    assert statistics.median(lengths) == summary["text"]["median_characters"]
    assert (
        len({r["RestaurantID"] for r in records if r["RestaurantID"].strip()})
        == summary["restaurants"]["unique_restaurants"]
    )
    assert Counter(r["Classification"] for r in records) == {
        r["value"]: r["count"] for r in summary["labels"]["classification_distribution"]
    }
    for duplicate in summary["duplicates"]:
        keys = duplicate["keys"]
        for population, name in (
            (records, "all_rows_missing_keys_compared_as_equal"),
            ([r for r in records if all(r[k].strip() for k in keys)], "complete_keys_only"),
        ):
            values = Counter(tuple(row[k] for k in keys) for row in population)
            assert sum(n - 1 for n in values.values()) == duplicate[name]["extra_rows_after_first"]
            assert (
                sum(n for n in values.values() if n > 1)
                == duplicate[name]["rows_in_duplicate_groups"]
            )
    parsed, missing, invalid = [], 0, 0
    for record in records:
        text = record["Review Time"]
        if not text.strip():
            missing += 1
            continue
        try:
            parsed.append(datetime.strptime(text, "%d/%m/%Y %H:%M"))
        except ValueError:
            invalid += 1
    assert len(parsed) == summary["dates"]["valid_timestamps"]
    assert missing == summary["dates"]["missing_timestamps"]
    assert invalid == summary["dates"]["invalid_nonempty_timestamps"]
    assert min(parsed).isoformat() == summary["dates"]["earliest"]
    assert max(parsed).isoformat() == summary["dates"]["latest"]
    assert int(pd.read_csv(root / "restaurant_statistics.csv").review_count.sum()) == count
    row_audit = pd.read_parquet("data/foody/audit/row_quality.parquet")
    assert len(row_audit) == count
    assert row_audit.source_record_number.tolist() == list(range(1, count + 1))
    assert row_audit.raw_rating.tolist() == [r["Rating"] for r in records]
    assert row_audit.raw_classification.tolist() == [r["Classification"] for r in records]
    assert row_audit.raw_review_time.tolist() == [r["Review Time"] for r in records]
    assert int(row_audit.comment_empty_or_whitespace.sum()) == count - len(nonempty)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == args.expected_sha256
    print(
        f"PASS: independent CSV counts/duplicates/dates/classification reconcile; all {count} rows and lexical raw_rating preserved; source SHA256 unchanged."
    )


if __name__ == "__main__":
    main()

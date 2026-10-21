"""Immutable snapshot, source-row preserving cleaning, inference-only deduplication."""

import json
from datetime import UTC, datetime

import pandas as pd

from src.foody.schema import (
    IDENTITY_TYPE,
    MAPPING,
    PROCESSED,
    REPORTS,
    SNAPSHOT,
    SOURCE,
    digest,
    identity,
    normalize,
    restaurant_key,
    review_id,
    save_json,
    save_parquet,
)
from src.foody.validation import load_source, parse_date, validate_processed
from src.foody_v2_audit import text_quality


def snapshot(source=SOURCE, target=SNAPSHOT):
    frame = load_source(source)
    sha = digest(source)
    manifest_path = target.with_suffix(".manifest.json")
    if target.exists() or manifest_path.exists():
        if not target.exists() or not manifest_path.exists():
            raise ValueError("Incomplete first-party snapshot pair; refusing overwrite")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if digest(target) != sha or manifest["sha256"] != sha:
            raise ValueError("Immutable first-party snapshot differs; use a new version")
        if manifest["row_count"] != len(frame) or manifest["file_size"] != source.stat().st_size:
            raise ValueError("Snapshot manifest does not reconcile")
        return manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(source.read_bytes())
    if digest(target) != sha or digest(source) != sha:
        raise ValueError("Source changed while preserving")
    manifest = {
        "source": str(source), "snapshot": str(target), "sha256": sha,
        "file_size": target.stat().st_size, "row_count": len(frame),
        "timestamp": datetime.now(UTC).isoformat(), "strict_utf8": True,
        "byte_identical": True, "data_role": "first_party_foody_inference",
        "provenance": "First-party ownership asserted by user; snapshot verifies bytes only",
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return manifest


def clean(frame, ingestion_timestamp):
    result = frame.rename(columns=MAPPING).copy()
    result["source_row"] = range(1, len(result) + 1)
    result["text"] = result.text_original.map(normalize)
    result["restaurant_key"] = result.restaurant_name.map(restaurant_key)
    result["restaurant_identity_type"] = IDENTITY_TYPE
    result["review_id"] = [
        review_id(k, u, t, c)
        for k, u, t, c in zip(
            result.restaurant_key, result.user_name, result.raw_review_time,
            result.text_original, strict=True,
        )
    ]
    result["observation_id"] = [
        identity("foody-ai-observation-v1", k, t)
        for k, t in zip(result.restaurant_key, result.text, strict=True)
    ]
    result["review_timestamp"] = pd.to_datetime(result.raw_review_time.map(parse_date))
    result["date_status"] = result.raw_review_time.map(
        lambda t: "missing" if not t.strip() else "parsed" if pd.notna(parse_date(t)) else "unparseable"
    )
    result["valid_for_ai"] = (
        result.text.ne("") & ~result.text.str.contains("\ufffd", regex=False)
        & result.restaurant_key.notna()
    )
    result["source"] = "foody"
    result["ingestion_timestamp"] = ingestion_timestamp
    groups = result.groupby("observation_id").source_row.agg(list)
    result["source_rows"] = result.observation_id.map(groups)
    result["duplicate_count"] = result.source_rows.map(len)
    validate_processed(result)
    return result


def prepare():
    manifest = snapshot()
    raw = load_source(SNAPSHOT)
    frame = clean(raw, manifest["timestamp"])
    quality, flags, cases = text_quality(raw.Comment)
    save_json(REPORTS / "suspicious_text_contexts.json", cases)
    save_parquet(PROCESSED, frame)
    flags["source_row"] = frame.source_row
    flags["text_original"] = frame.text_original
    suspicious = flags.loc[
        flags[["replacement", "internal_question", "double_question", "mojibake"]].any(axis=1)
    ]
    save_parquet(REPORTS / "suspicious_text_cases.parquet", suspicious)
    ratings = pd.to_numeric(frame.raw_rating, errors="coerce")
    counts = frame.groupby("restaurant_key").size()
    report = {
        "raw_rows": len(frame), "valid_ai_rows": int(frame.valid_for_ai.sum()),
        "deduplicated_ai_observations": int(
            frame.loc[frame.valid_for_ai, "observation_id"].nunique()
        ),
        "distinct_valid_review_ids": int(frame.loc[frame.valid_for_ai, "review_id"].nunique()),
        "unique_restaurant_names_exact": int(frame.restaurant_name.nunique()),
        "restaurant_count": int(frame.restaurant_key.nunique()),
        "restaurant_identity_type": IDENTITY_TYPE,
        "reviews_per_restaurant": counts.describe().to_dict(),
        "date_status": frame.date_status.value_counts().to_dict(),
        "date_range": [str(frame.review_timestamp.min()), str(frame.review_timestamp.max())],
        "timestamp_timezone": "source local time assumed Asia/Ho_Chi_Minh; stored naive",
        "rating_range": [float(ratings.min()), float(ratings.max())],
        "rating_distribution": frame.raw_rating.value_counts().sort_index().to_dict(),
        "missing_ratings": int(frame.raw_rating.str.strip().eq("").sum()),
        "invalid_ratings": int((ratings.isna() & frame.raw_rating.str.strip().ne("")).sum()),
        "ratings_outside_0_10": int((ratings.notna() & ~ratings.between(0, 10)).sum()),
        "strict_utf8": True, "text_quality": quality,
        "source_sha256": manifest["sha256"], "processed_sha256": digest(PROCESSED),
        "row_policy": "Every source row retained; inference grouped by restaurant_key + NFC/whitespace-normalized text",
        "classification": "Absent; no synthetic source_classification column or rating labels",
    }
    save_json(REPORTS / "cleaning.json", report)
    return report

import hashlib
import json
import math
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.ml.config import LABELS, MLConfig
from src.ml.features import normalize_text
from src.ml.labels import label_rating
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum
from src.utils.timestamps import utc_now

FIELDS = ("review_id", "product_id", "rating", "comment", "review_timestamp", "source")


def load_provenance(config: MLConfig) -> dict:
    path = config.data_dir / "source/provenance.json"
    if not path.exists():
        raise ValueError(
            "No real review dataset imported. Run import-ml-data with a publicly licensed dataset first; fixtures cannot train a model."
        )
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if (
        provenance.get("dataset_kind") != "external_public"
        or not provenance.get("license")
        or not provenance.get("source_url")
        or not provenance.get("license_url")
    ):
        raise ValueError(
            "Dataset needs external public license provenance; synthetic fixtures are prohibited"
        )
    snapshot = config.data_dir / "source" / provenance["snapshot_filename"]
    if not snapshot.exists() or file_checksum(snapshot) != provenance["input_sha256"]:
        raise ValueError("Dataset snapshot is missing or changed; reimport the licensed dataset")
    return provenance


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Dataset input must be UTF-8 CSV or Parquet")


def nullable_text(value):
    return None if pd.isna(value) or not str(value).strip() else normalize_text(str(value))


def normalize_rows(
    frame: pd.DataFrame, source: str, mapping: dict
) -> tuple[pd.DataFrame, list[dict], int]:
    columns = {field: mapping.get(field, field) for field in FIELDS}
    missing = [
        columns[field] for field in ("rating", "comment") if columns[field] not in frame.columns
    ]
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(missing)}")
    rows, rejected = [], []
    for index, row in frame.iterrows():
        try:
            raw_comment = row[columns["comment"]]
            if pd.isna(raw_comment) or not isinstance(raw_comment, str):
                raise ValueError("comment must be text")
            comment = normalize_text(raw_comment)
            if not comment:
                raise ValueError("empty comment")
            value = row[columns["rating"]]
            if isinstance(value, bool):
                raise ValueError("rating must be an integer 1–5")
            rating = float(str(value).strip().replace(",", "."))
            if not math.isfinite(rating) or rating != int(rating) or not 1 <= rating <= 5:
                raise ValueError("rating must be an integer 1–5")
            product_id = nullable_text(row.get(columns["product_id"]))
            timestamp = nullable_text(row.get(columns["review_timestamp"]))
            if timestamp:
                timestamp = pd.Timestamp(timestamp)
                if timestamp.tzinfo is None:
                    raise ValueError("review_timestamp must include timezone when supplied")
                timestamp = timestamp.tz_convert("UTC").isoformat()
            review_id = nullable_text(row.get(columns["review_id"]))
            if not review_id:
                review_id = hashlib.sha256(
                    json.dumps(
                        [source, product_id, int(rating), comment, timestamp], ensure_ascii=False
                    ).encode()
                ).hexdigest()
            rows.append(
                {
                    "review_id": review_id,
                    "product_id": product_id,
                    "rating": int(rating),
                    "comment": comment,
                    "review_timestamp": timestamp,
                    "source": source,
                    "original_source": nullable_text(row.get(columns["source"])),
                }
            )
        except (ValueError, TypeError, OverflowError):
            # Report row number and reason; do not include private review contents.
            rejected.append(
                {"row": int(index) + 2, "reason": "Invalid comment, rating or timestamp"}
            )
    normalized = pd.DataFrame(rows, columns=[*FIELDS, "original_source"])
    before = len(normalized)
    normalized = normalized.drop_duplicates(subset=list(FIELDS)).reset_index(drop=True)
    normalized = normalized.sort_values(
        ["source", "review_id", "comment", "rating"], kind="stable"
    ).reset_index(drop=True)
    return normalized, rejected, before - len(normalized)


def dataset_hash(frame: pd.DataFrame) -> str:
    columns = [c for c in frame.columns if c != "ingestion_timestamp"]
    records = sorted(
        json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        for row in frame[columns].to_dict("records")
    )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def data_report(frame: pd.DataFrame) -> dict:
    required = {
        "review_id",
        "product_id",
        "comment",
        "rating",
        "sentiment",
        "source",
        "ingestion_timestamp",
    }
    missing = sorted(required - set(frame.columns))
    reasons = []
    if missing:
        reasons.append(f"Required columns missing: {missing}")
    if frame.empty:
        reasons.append("Dataset is empty")
    labels = (
        frame["sentiment"].fillna("<missing>").astype(str).value_counts().to_dict()
        if "sentiment" in frame
        else {}
    )
    if len(labels) < 2:
        reasons.append("Fewer than 2 sentiment classes")
    if set(labels) - set(LABELS):
        reasons.append("Invalid sentiment labels")
    if (
        "source" in frame
        and (
            frame["source"].isna() | frame["source"].fillna("").astype(str).str.strip().eq("")
        ).any()
    ):
        reasons.append("Source provenance missing from rows")
    if "rating" in frame:
        ratings = pd.to_numeric(frame["rating"], errors="coerce")
        if (~ratings.isin([1, 2, 3, 4, 5])).any():
            reasons.append("Invalid ratings")
    comments = frame["comment"] if "comment" in frame else pd.Series(dtype=str)
    nulls = int(comments.isna().sum())
    if nulls or comments.fillna("").astype(str).str.strip().eq("").any():
        reasons.append("Null or empty comments")
    lengths = comments.dropna().astype(str).str.len()
    return {
        "row_count": len(frame),
        "missing_columns": missing,
        "null_comments": nulls,
        "duplicate_comments": int(comments.duplicated().sum()),
        "label_distribution": labels,
        "rating_distribution": {str(k): int(v) for k, v in frame["rating"].value_counts().items()}
        if "rating" in frame
        else {},
        "average_text_length": float(lengths.mean()) if len(lengths) else None,
        "min_text_length": int(lengths.min()) if len(lengths) else None,
        "max_text_length": int(lengths.max()) if len(lengths) else None,
        "trainable": not reasons,
        "trainability_errors": reasons,
    }


def prepare(config: MLConfig) -> dict:
    provenance = load_provenance(config)
    original = read_input(config.data_dir / "source" / provenance["snapshot_filename"])
    normalized, rejected, duplicates = normalize_rows(
        original, provenance["source"], provenance["columns"]
    )
    normalized["sentiment"] = [
        label_rating(int(value), config.label_rules) for value in normalized["rating"]
    ]
    normalized["ingestion_timestamp"] = pd.to_datetime(provenance["ingestion_timestamp"], utc=True)
    target = config.data_dir / "processed/review_ml_dataset.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(target, index=False)
    report = {
        **data_report(normalized),
        "input_rows": len(original),
        "rejected_rows": len(rejected),
        "exact_duplicates_removed": duplicates,
        "dataset_hash": dataset_hash(normalized),
        "source": provenance["source"],
        "license": provenance["license"],
        "source_url": provenance["source_url"],
        "label_rules": config.label_rules,
        "input_sha256": provenance["input_sha256"],
        "gold_contract": "shopee.gold.review_ml_dataset",
        "storage": str(target),
        "storage_scope": "local external training snapshot; no cloud table modified",
    }
    atomic_json(config.reports_dir / "ml_data_report.json", report)
    atomic_json(config.reports_dir / "ml_rejections.json", rejected)
    return report


def import_dataset(
    path: Path,
    source: str,
    license_name: str,
    source_url: str,
    license_url: str,
    mapping: dict,
    config: MLConfig,
) -> dict:
    if not source.strip() or any(word in source.lower() for word in ("synthetic", "fixture")):
        raise ValueError(
            "Use a legitimate external dataset source; synthetic fixtures are prohibited"
        )
    if "fixtures" in {p.lower() for p in path.parts}:
        raise ValueError("Test fixtures cannot be imported for model training")
    if not license_name or not source_url or not license_url:
        raise ValueError(
            "Provide --license, --source-url and --license-url documenting the public dataset license"
        )
    if any(urlsplit(url).scheme not in {"http", "https"} for url in (source_url, license_url)):
        raise ValueError("Source and license URLs must be HTTP(S)")
    original = read_input(path)
    normalize_rows(original, source, mapping)  # validate schema before creating a snapshot
    snapshot = config.data_dir / "source" / ("dataset" + path.suffix.lower())
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != snapshot.resolve():
        shutil.copyfile(path, snapshot)
    atomic_json(
        snapshot.parent / "provenance.json",
        {
            "dataset_kind": "external_public",
            "source": source.strip(),
            "license": license_name,
            "source_url": source_url,
            "license_url": license_url,
            "columns": mapping,
            "input_sha256": file_checksum(snapshot),
            "snapshot_filename": snapshot.name,
            "ingestion_timestamp": utc_now().isoformat(),
            "license_note": "Operator-supplied provenance; the importer does not independently certify licensing",
        },
    )
    return prepare(config)


def require_trainable(config: MLConfig) -> tuple[pd.DataFrame, dict]:
    provenance = load_provenance(config)
    path = config.data_dir / "processed/review_ml_dataset.parquet"
    if not path.exists():
        raise ValueError("Processed ML dataset missing; run ml-prepare")
    frame = pd.read_parquet(path)
    report = data_report(frame)
    if not report["trainable"]:
        raise ValueError("Dataset not trainable: " + "; ".join(report["trainability_errors"]))
    stored = json.loads((config.reports_dir / "ml_data_report.json").read_text(encoding="utf-8"))
    if (
        dataset_hash(frame) != stored["dataset_hash"]
        or stored["input_sha256"] != provenance["input_sha256"]
    ):
        raise ValueError("ML data/report hashes disagree; rerun ml-prepare")
    if {int(k): v for k, v in stored["label_rules"].items()} != config.label_rules:
        raise ValueError("Label configuration changed; rerun ml-prepare")
    return frame, stored

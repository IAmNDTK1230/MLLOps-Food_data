"""The published v3 card defines sentiment on the signed ordinal scale -3..3.

Our documented three-class task collapses its intensity by sign; stars are unused.
Source: https://huggingface.co/datasets/phamluan/vn_msa_sar_v3
"""

import math
from typing import Protocol

import pandas as pd

from src.ml.features import normalize_text

SOURCE_URL = "https://huggingface.co/datasets/phamluan/vn_msa_sar_v3"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
REQUIRED = (
    "review_id",
    "source",
    "place_type",
    "language",
    "sentence",
    "aspect",
    "sentiment",
    "sarcasm",
)
LABEL_MAPPING = {
    str(i): "negative" if i < 0 else "positive" if i > 0 else "neutral" for i in range(-3, 4)
}


def ordinal(value, allowed):
    if isinstance(value, bool) or pd.isna(value):
        raise ValueError("invalid_label")
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError("invalid_label") from None
    if not math.isfinite(number) or number != int(number) or int(number) not in allowed:
        raise ValueError("invalid_label")
    return int(number)


def adapt_vn_msasar(frame: pd.DataFrame, provenance: dict):
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"VN-MSASar v3 required columns missing: {sorted(missing)}")
    filtered = frame.loc[(frame.language == "vi") & (frame.place_type == "restaurant")].copy()
    accepted, rejected = [], []
    for index, row in filtered.iterrows():
        try:
            text = normalize_text(str(row.sentence)) if pd.notna(row.sentence) else ""
            if not text:
                raise ValueError("empty_text")
            if any(
                pd.isna(row[key]) or not str(row[key]).strip()
                for key in ("review_id", "aspect", "source")
            ):
                raise ValueError("missing_metadata")
            score = ordinal(row.sentiment, range(-3, 4))
            sarcasm = ordinal(row.sarcasm, (0, 1))
            record = {
                "review_id": str(row.review_id),
                "restaurant_id": None,
                "text": text,
                "original_text": row.sentence,
                "aspect": row.aspect,
                "sentiment": LABEL_MAPPING[str(score)],
                "original_sentiment": score,
                "sarcasm": sarcasm,
                "source": row.source,
                "dataset_name": "vn_msasar_v3",
                "language": "vi",
                "place_type": "restaurant",
                "split": None,
                "original_split": row.get("original_split"),
                "ingestion_timestamp": provenance["import_timestamp"],
                "license": provenance["license"],
                "source_url": provenance["source_url"],
                "license_url": provenance["license_url"],
            }
            for field in ("restaurant_id", "rating", "review_timestamp"):
                if field in row:
                    record[field] = row[field] if pd.notna(row[field]) else None
            accepted.append(record)
        except ValueError as error:
            # Original rejected rows remain locally available, never printed in CLI logs.
            rejected.append({**row.to_dict(), "source_row": int(index), "reason": str(error)})
    if not accepted:
        raise ValueError("No valid Vietnamese restaurant rows; inspect source schema and labels")
    rows = pd.DataFrame(accepted)
    deduplicated = rows.drop_duplicates(
        subset=["review_id", "text", "aspect", "original_sentiment", "sarcasm", "source"]
    )
    stats = {
        "raw_rows": len(frame),
        "filtered_vietnamese_restaurant_rows": len(filtered),
        "filtered_out_rows": len(frame) - len(filtered),
        "rejected_rows": len(rejected),
        "duplicates_removed": len(rows) - len(deduplicated),
        "valid_rows": len(deduplicated),
    }
    return (
        deduplicated.sort_values(["review_id", "text", "aspect", "original_sentiment"]).reset_index(
            drop=True
        ),
        rejected,
        stats,
    )


class ExternalEvaluationAdapter(Protocol):
    """Future adapters must supply their OWN documented gold label mapping."""

    dataset_name: str

    def adapt(self, frame: pd.DataFrame, provenance: dict) -> pd.DataFrame: ...


class NTCSCVAdapter:
    dataset_name = "ntc_scv"

    def adapt(self, frame, provenance):
        raise NotImplementedError(
            "NTC-SCV is external evaluation only; confirm its schema/license first"
        )


class VLSP2018RestaurantAdapter:
    dataset_name = "vlsp2018_restaurant"

    def adapt(self, frame, provenance):
        raise NotImplementedError(
            "VLSP2018 Restaurant is external evaluation only; confirm its schema/license first"
        )

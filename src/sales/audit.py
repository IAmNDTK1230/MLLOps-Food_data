"""Read-only order quality and conservative review identity audit."""

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SOURCE = Path("data/foody/source/order_cleaning.csv")
REVIEW_SUMMARY = Path("data/foody/mart/restaurant_summary.parquet")
COLUMNS = ("order_id", "order_date", "sales_amount", "order_status",
           "restaurant_id", "restaurant_name")
STATUSES = {"Delivered", "Cancelled", "Refunded"}


def normalized_name(value):
    """Preserve accents and punctuation; only standardize Unicode, case, spaces."""
    if pd.isna(value):
        return None
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip().casefold() or None


def read_orders(path=SOURCE):
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"Sales source columns differ from expected contract: {list(frame.columns)}")
    return frame


def audit_orders(path=SOURCE, review_summary=REVIEW_SUMMARY):
    orders = read_orders(path)
    dates = pd.to_datetime(orders.order_date, format="%Y-%m-%d", errors="coerce")
    amounts = pd.to_numeric(orders.sales_amount, errors="coerce")
    names = orders[["restaurant_id", "restaurant_name"]].drop_duplicates().copy()
    names["normalized_name"] = names.restaurant_name.map(normalized_name)
    reviews = pd.read_parquet(review_summary, columns=["restaurant_key", "restaurant_name"])
    reviews["normalized_name"] = reviews.restaurant_name.map(normalized_name)
    by_id = names.groupby("restaurant_id").normalized_name.agg(
        lambda values: {value for value in values if value}
    )
    sales_by_name = names.dropna(subset=["normalized_name"]).groupby(
        "normalized_name"
    ).restaurant_id.nunique()
    review_by_name = reviews.groupby("normalized_name").size()
    review_key_by_name = dict(zip(reviews.normalized_name, reviews.restaurant_key, strict=True))
    matched, ambiguous = [], []
    for restaurant_id, candidates in by_id.items():
        if len(candidates) != 1:
            ambiguous.append({"restaurant_id": restaurant_id,
                              "reason": "multiple_names" if candidates else "no_name"})
            continue
        candidate = next(iter(candidates))
        if sales_by_name.get(candidate, 0) == review_by_name.get(candidate, 0) == 1:
            matched.append({"restaurant_id": restaurant_id,
                            "restaurant_key": review_key_by_name[candidate],
                            "normalized_name": candidate,
                            "match_method": "unique_normalized_name"})
        elif sales_by_name.get(candidate, 0) > 1 or review_by_name.get(candidate, 0) > 1:
            ambiguous.append({"restaurant_id": restaurant_id, "reason": "name_collision"})
    percentiles = amounts.quantile([0.01, 0.5, 0.95, 0.99])
    row_count = len(orders)
    as_of = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()
    return {
        "source": str(path), "rows": row_count, "columns": list(orders.columns),
        "null_counts": {key: int(value) for key, value in orders.isna().sum().items()},
        "duplicate_order_id_rows": int(orders.order_id.duplicated(keep=False).sum()),
        "unique_order_ids": int(orders.order_id.nunique()),
        "date_min": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_max": dates.max().date().isoformat() if dates.notna().any() else None,
        "invalid_dates": int(dates.isna().sum()),
        "analytics_as_of_date": as_of.isoformat(),
        "future_dated_rows_excluded_from_live_metrics": int(dates.dt.date.gt(as_of).sum()),
        "sales_amount": {"min": float(amounts.min()), "max": float(amounts.max()),
                         "mean": float(amounts.mean()),
                         "p01": float(percentiles.loc[0.01]),
                         "median": float(percentiles.loc[0.5]),
                         "p95": float(percentiles.loc[0.95]),
                         "p99": float(percentiles.loc[0.99]),
                         "negative_count": int(amounts.lt(0).sum()),
                         "zero_count": int(amounts.eq(0).sum()),
                         "invalid_count": int(amounts.isna().sum())},
        "status_distribution": {str(key): int(value) for key, value in
                                orders.order_status.value_counts(dropna=False).items()},
        "restaurant_ids": len(by_id),
        "missing_restaurant_name_rows": int(orders.restaurant_name.isna().sum()),
        "review_restaurants": len(reviews),
        "exact_id_overlap": None,
        "exact_id_overlap_note": "Review source has no restaurant_id; review key is name-derived.",
        "matched_count": len(matched),
        "ambiguous_count": len(ambiguous),
        "unmatched_count": len(by_id) - len(matched) - len(ambiguous),
        "sales_only_count": len(by_id) - len(matched),
        "review_only_count": len(reviews) - len(matched),
        "matched": matched, "ambiguous": ambiguous,
        "revenue_policy": "Delivered sales_amount only; cancelled/refunded excluded",
        "aov_policy": "Delivered sales / Delivered order count",
        "rate_policy": "Cancelled or Refunded order count / all order count",
    }


def require_loadable(audit):
    if (audit["duplicate_order_id_rows"] or audit["invalid_dates"]
            or audit["sales_amount"]["invalid_count"]
            or audit["sales_amount"]["negative_count"]
            or audit["null_counts"]["restaurant_id"]
            or audit["null_counts"]["order_id"]
            or set(audit["status_distribution"]) - STATUSES):
        raise ValueError("Sales source has invalid IDs, dates, amounts, statuses, or duplicates")

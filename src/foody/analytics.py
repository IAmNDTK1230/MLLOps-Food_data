"""Restaurant marts with explicit denominators and historical comparable-period alerts."""

from pathlib import Path

import pandas as pd

from src.foody.schema import ENRICHED, REPORTS, save_json, save_parquet

MART = Path("data/foody/mart")


def business_rows(frame):
    # Exact duplicate review IDs are counted once; separate users/dates stay separate.
    result = frame.drop_duplicates("review_id").copy()
    result["numeric_rating"] = pd.to_numeric(result.raw_rating, errors="coerce")
    result["numeric_rating"] = result.numeric_rating.where(result.numeric_rating.between(0, 10))
    return result


def aggregate(frame, keys):
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key, strict=True))
        ai = group.loc[group.valid_for_ai & group.predicted_sentiment.notna()]
        row.update(
            review_count=len(group), ai_review_count=len(ai),
            rating_count=int(group.numeric_rating.notna().sum()),
            average_rating=group.numeric_rating.mean(),
        )
        for label in ("positive", "neutral", "negative"):
            count = int(ai.predicted_sentiment.eq(label).sum())
            row[f"predicted_{label}_count"] = count
            row[f"predicted_{label}_rate"] = count / len(ai) if len(ai) else None
        for aspect in ("service", "price", "food"):
            count = sum(
                sentiment == "negative" and aspect in aspects
                for sentiment, aspects in zip(ai.predicted_sentiment, ai.predicted_aspects, strict=True)
            )
            row[f"predicted_{aspect}_complaint_count"] = int(count)
            row[f"predicted_{aspect}_complaint_rate"] = count / len(ai) if len(ai) else None
        rows.append(row)
    return pd.DataFrame(rows)


ALERT_COLUMNS = [
    "restaurant_key", "current_period", "previous_period", "metric",
    "current_value", "previous_value", "change", "current_sample_count",
    "previous_sample_count", "minimum_samples", "historical_only",
]


def alerts(monthly, minimum_samples=10, rate_increase=0.2, rating_decrease=1.0):
    if minimum_samples < 1 or not 0 < rate_increase <= 1 or rating_decrease <= 0:
        raise ValueError("Alert thresholds must be positive and rate increase <= 1")
    output = []
    for key, group in monthly.groupby("restaurant_key"):
        indexed = {pd.Period(row["period"], freq="M"): row for row in group.to_dict("records")}
        # All pairs of adjacent complete calendar months; historical backtest, not live alerts.
        for period, current in indexed.items():
            previous = indexed.get(period - 1)
            if previous is None:
                continue
            for metric in (
                "predicted_negative_rate", "predicted_service_complaint_rate",
                "predicted_price_complaint_rate", "predicted_food_complaint_rate", "average_rating",
            ):
                denominator = "rating_count" if metric == "average_rating" else "ai_review_count"
                if min(current[denominator], previous[denominator]) < minimum_samples:
                    continue
                delta = current[metric] - previous[metric]
                triggered = delta <= -rating_decrease if metric == "average_rating" else delta >= rate_increase
                if triggered:
                    output.append({
                        "restaurant_key": key, "current_period": str(period),
                        "previous_period": str(period - 1), "metric": metric,
                        "current_value": current[metric], "previous_value": previous[metric],
                        "change": delta, "current_sample_count": current[denominator],
                        "previous_sample_count": previous[denominator],
                        "minimum_samples": minimum_samples, "historical_only": True,
                    })
    return pd.DataFrame(output, columns=ALERT_COLUMNS)


def label_summary(frame, column, label_name):
    records = []
    for key, group in frame.loc[frame.valid_for_ai].groupby("restaurant_key"):
        counts = group[column].explode().dropna().value_counts()
        for label, count in counts.items():
            records.append({"restaurant_key": key, label_name: label,
                            "predicted_review_count": int(count),
                            "ai_review_count": len(group),
                            "predicted_rate": count / len(group)})
    return pd.DataFrame(records, columns=[
        "restaurant_key", label_name, "predicted_review_count", "ai_review_count", "predicted_rate"
    ])


def build_mart(minimum_samples=10, rate_increase=0.2, rating_decrease=1.0):
    raw = pd.read_parquet(ENRICHED)
    frame = business_rows(raw)
    summary = aggregate(frame, ["restaurant_key"])
    names = frame.groupby("restaurant_key").restaurant_name.agg(lambda x: sorted(set(x)))
    summary["restaurant_names"] = summary.restaurant_key.map(names)
    summary["restaurant_name"] = summary.restaurant_names.map(lambda x: x[0])
    summary["restaurant_identity_type"] = "derived_from_restaurant_name"
    summary["source_row_count"] = summary.restaurant_key.map(raw.groupby("restaurant_key").size())
    daily = aggregate(frame.assign(period=frame.review_timestamp.dt.strftime("%Y-%m-%d")).dropna(
        subset=["period"]
    ), ["restaurant_key", "period"])
    monthly = aggregate(frame.assign(period=frame.review_timestamp.dt.strftime("%Y-%m")).dropna(
        subset=["period"]
    ), ["restaurant_key", "period"])
    # The final observed month might be partial; exclude it from historical warning comparisons.
    maximum_month = frame.review_timestamp.max().strftime("%Y-%m")
    warning = alerts(monthly.loc[monthly.period < maximum_month],
                     minimum_samples, rate_increase, rating_decrease)
    outputs = {
        "restaurant_summary": summary, "restaurant_daily_metrics": daily,
        "restaurant_monthly_metrics": monthly,
        "restaurant_aspect_summary": label_summary(frame, "predicted_aspects", "aspect"),
        "restaurant_issue_summary": label_summary(frame, "detected_issues", "issue"),
        "alerts": warning,
    }
    for name, table in outputs.items():
        save_parquet(MART / f"{name}.parquet", table)
    report = {
        "outputs": {name: len(table) for name, table in outputs.items()},
        "counting": "distinct review_id; every source row preserved in enriched. Distinct users/dates count separately even when sharing inference",
        "ai_denominator": "valid distinct reviews with a prediction",
        "rating_denominator": "nonmissing numeric raw_rating in [0,10]; unchanged raw value retained",
        "undated_distinct_reviews": int(frame.review_timestamp.isna().sum()),
        "alerts": {"minimum_samples_each_period": minimum_samples,
                   "rate_increase_absolute": rate_increase, "rating_decrease": rating_decrease,
                   "excluded_latest_partial_month": maximum_month,
                   "mode": "historical adjacent calendar months, not current live conditions",
                   "examples": warning.head(10).to_dict("records")},
    }
    save_json(REPORTS / "mart.json", report)
    return report

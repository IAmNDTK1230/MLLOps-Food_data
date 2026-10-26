"""Side-by-side 2025–2026 review corpus preparation with verified sales IDs."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.foody.enrichment import extract_issues
from src.foody.review_paths import CANONICAL, DATA_VERSION, DOCUMENTS, ENRICHED, MANIFEST, SOURCE
from src.foody.schema import digest, normalize, save_json
from src.sales.audit import SOURCE as SALES_SOURCE
from src.sales.audit import normalized_name, read_orders
from src.sales.loader import canonical_names

AUDIT = Path("reports/review/review_dataset_migration_audit.md")
ALIGNMENT = Path("reports/review/sales_review_id_alignment.md")
COLUMNS = ("Review Time", "Rating", "Comment", "RestaurantID", "Restaurant Name", "Address")


def read_source(path=SOURCE):
    frame = pd.read_csv(path, dtype="string", encoding="utf-8", encoding_errors="strict")
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"Review source columns differ from contract: {list(frame.columns)}")
    return frame


def audit_source(frame):
    dates = pd.to_datetime(frame["Review Time"], format="%d/%m/%Y", errors="coerce")
    ratings = pd.to_numeric(frame.Rating, errors="coerce")
    ids = pd.to_numeric(frame.RestaurantID, errors="coerce")
    pairs = frame[["RestaurantID", "Restaurant Name"]].drop_duplicates()
    report = {
        "rows": len(frame), "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_review_records": int(frame.duplicated(
            subset=["RestaurantID", "Review Time", "Rating", "Comment"]).sum()),
        "deduplicated_rows": int(len(frame.drop_duplicates())),
        "columns": list(frame), "null_counts": frame.isna().sum().astype(int).to_dict(),
        "unique_restaurant_ids": int(frame.RestaurantID.nunique()),
        "unique_restaurant_names": int(frame["Restaurant Name"].nunique()),
        "restaurant_id_min": int(ids.min()) if ids.notna().any() else None,
        "restaurant_id_max": int(ids.max()) if ids.notna().any() else None,
        "invalid_restaurant_ids": int(ids.isna().sum()),
        "invalid_dates": int(dates.isna().sum()),
        "invalid_ratings": int((ratings.isna() | ratings.lt(0) | ratings.gt(10)).sum()),
        "replacement_character_rows": int(frame.Comment.fillna("").map(
            lambda value: "\ufffd" in value).sum()),
        "date_min": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_max": dates.max().date().isoformat() if dates.notna().any() else None,
        "restaurant_id_multiple_names": int(pairs.groupby("RestaurantID")["Restaurant Name"].nunique().gt(1).sum()),
        "restaurant_name_multiple_ids": int(pairs.groupby("Restaurant Name").RestaurantID.nunique().gt(1).sum()),
    }
    return report


def align_ids(reviews, sales):
    review_pairs = reviews[["RestaurantID", "Restaurant Name"]].drop_duplicates()
    if review_pairs.groupby("RestaurantID")["Restaurant Name"].nunique().gt(1).any():
        raise ValueError("Review restaurant ID maps to multiple names")
    if review_pairs.groupby("Restaurant Name").RestaurantID.nunique().gt(1).any():
        raise ValueError("Review restaurant name maps to multiple IDs")
    review_names = dict(zip(review_pairs.RestaurantID, review_pairs["Restaurant Name"], strict=True))
    sales_names = canonical_names(sales)
    review_ids, sales_ids = set(review_names), set(sales_names)
    overlap = review_ids & sales_ids
    rows = [{"restaurant_id": rid, "sales_name": sales_names[rid],
             "review_name": review_names[rid],
             "exact": sales_names[rid] == review_names[rid],
             "normalized": normalized_name(sales_names[rid]) == normalized_name(review_names[rid])}
            for rid in sorted(overlap, key=int)]
    result = {
        "sales_unique_restaurant_ids": len(sales_ids),
        "review_unique_restaurant_ids": len(review_ids),
        "id_intersection": len(overlap),
        "review_ids_in_sales": len(overlap), "sales_ids_with_reviews": len(overlap),
        "review_ids_absent_from_sales": len(review_ids - sales_ids),
        "sales_ids_without_reviews": len(sales_ids - review_ids),
        "review_restaurant_coverage_pct": round(100 * len(overlap) / len(review_ids), 2)
        if review_ids else 0.0,
        "exact_name_matches": sum(row["exact"] for row in rows),
        "normalized_name_matches": sum(row["normalized"] for row in rows),
        "name_mismatches": sum(not row["normalized"] for row in rows),
        "mismatch_sample": [row for row in rows if not row["normalized"]][:20],
    }
    result["direct_join_accepted"] = (len(overlap) == len(review_ids)
                                      and result["name_mismatches"] == 0)
    return result


def require_accepted(audit, alignment):
    invalid = (audit["invalid_restaurant_ids"] or audit["invalid_dates"]
               or audit["invalid_ratings"] or audit["restaurant_id_multiple_names"]
               or audit["restaurant_name_multiple_ids"] or any(audit["null_counts"].values()))
    if invalid or not alignment["direct_join_accepted"]:
        raise ValueError("Review schema or Sales/Review ID namespace failed the migration gate")


def stable_review_id(restaurant_id, review_time, rating, comment):
    canonical = json.dumps([str(restaurant_id), review_time.date().isoformat(),
                            format(float(rating), ".12g"), normalize(comment)],
                           ensure_ascii=False, separators=(",", ":"))
    return "r26_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonicalize(frame):
    clean = frame.drop_duplicates().copy()
    dates = pd.to_datetime(clean["Review Time"], format="%d/%m/%Y", errors="raise")
    ratings = pd.to_numeric(clean.Rating, errors="raise")
    output = pd.DataFrame({
        "restaurant_id": clean.RestaurantID.astype(str),
        "restaurant_name": clean["Restaurant Name"].astype(str),
        "address": clean.Address.astype(str),
        "review_time": dates,
        "rating": ratings.astype(float),
        "comment": clean.Comment.map(normalize),
        "source": "foody_review_2025_2026",
    })
    output.insert(0, "review_id", [stable_review_id(i, t, r, c) for i, t, r, c in zip(
        output.restaurant_id, output.review_time, output.rating, output.comment, strict=True)])
    if output.review_id.duplicated().any():
        raise ValueError("Stable review ID collision after exact deduplication")
    return output.reset_index(drop=True)


def enrich(frame, sentiment_model, aspect_bundle):
    texts = frame.comment.tolist()
    sentiment_prob = sentiment_model.predict_proba(texts)
    sentiment_indices = np.argmax(sentiment_prob, axis=1)
    aspect_prob = aspect_bundle["pipeline"].predict_proba(texts)
    enriched = frame.copy()
    enriched["predicted_sentiment"] = [str(sentiment_model.classes_[i]) for i in sentiment_indices]
    enriched["sentiment_confidence"] = sentiment_prob.max(axis=1).astype(float)
    classes = list(aspect_bundle["classes"])
    threshold = float(aspect_bundle["threshold"])
    enriched["predicted_aspects"] = [[name for name, score in zip(classes, scores, strict=True)
                                      if score >= threshold] for scores in aspect_prob]
    enriched["aspect_scores"] = [dict(zip(classes, map(float, scores), strict=True))
                                  for scores in aspect_prob]
    enriched["detected_issues"] = [extract_issues(text, aspects, sentiment)
                                   for text, aspects, sentiment in zip(
                                       texts, enriched.predicted_aspects,
                                       enriched.predicted_sentiment, strict=True)]
    return enriched


def write_documents(frame, path=DOCUMENTS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eligible = frame.loc[frame.comment.map(lambda text: bool(text.strip()) and "\ufffd" not in text)]
    with path.open("w", encoding="utf-8") as output:
        for row in eligible.itertuples(index=False):
            metadata = {
                "review_id": row.review_id, "restaurant_id": row.restaurant_id,
                "restaurant_name": row.restaurant_name,
                "review_time": row.review_time.isoformat(), "rating": row.rating,
                "predicted_sentiment": row.predicted_sentiment,
                "predicted_aspects": row.predicted_aspects,
                "detected_issues": row.detected_issues, "source": row.source,
            }
            output.write(json.dumps({"id": row.review_id, "text": row.comment,
                                     "metadata": metadata}, ensure_ascii=False) + "\n")
    return {"document_count": len(eligible), "excluded_corrupted_or_empty": len(frame) - len(eligible),
            "sha256": digest(path)}


def prepare(source=SOURCE):
    source = Path(source)
    reviews = read_source(source)
    sales = read_orders(SALES_SOURCE)
    audit = audit_source(reviews)
    alignment = align_ids(reviews, sales)
    _write_reports(audit, alignment, source)
    require_accepted(audit, alignment)
    canonical = canonicalize(reviews)
    CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(CANONICAL, index=False)
    sentiment_path = Path("models/tfidf_logreg.joblib")
    aspect_path = Path("models/aspect_tfidf.joblib")
    enriched = enrich(canonical, joblib.load(sentiment_path), joblib.load(aspect_path))
    enriched.to_parquet(ENRICHED, index=False)
    docs = write_documents(enriched)
    manifest = {
        "data_version": DATA_VERSION, "schema_version": 1,
        "source_file": str(source), "source_sha256": digest(source),
        "input_rows": audit["rows"], "exact_duplicates_removed": audit["exact_duplicate_rows"],
        "row_count": len(enriched), "restaurant_count": enriched.restaurant_id.nunique(),
        "date_min": audit["date_min"], "date_max": audit["date_max"],
        "sentiment_model_sha256": digest(sentiment_path),
        "aspect_model_sha256": digest(aspect_path),
        "sentiment_enrichment_count": int(enriched.predicted_sentiment.notna().sum()),
        "aspect_enrichment_count": int(enriched.predicted_aspects.notna().sum()),
        "documents": docs, "created_at": datetime.now(UTC).isoformat(),
    }
    save_json(MANIFEST, manifest)
    return manifest


def _write_reports(audit, alignment, source):
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("# Review dataset migration audit\n\n"
                     f"Source: `{source}`; SHA256: `{digest(source)}`.\n\n"
                     "```json\n" + json.dumps(audit, ensure_ascii=False, indent=2)
                     + "\n```\n", encoding="utf-8")
    sample = alignment["mismatch_sample"]
    ALIGNMENT.write_text("# Sales ↔ Review ID alignment\n\n"
                         "Names are diagnostic only; joins use the ID after this gate.\n\n"
                         "```json\n" + json.dumps(alignment, ensure_ascii=False, indent=2)
                         + "\n```\n\n## Mismatched ID sample\n\n"
                         + ("| restaurant_id | sales_name | review_name |\n|---|---|---|\n"
                            + "\n".join(f"| {row['restaurant_id']} | {row['sales_name']} | "
                                        f"{row['review_name']} |" for row in sample)
                            if sample else "None.\n"), encoding="utf-8")

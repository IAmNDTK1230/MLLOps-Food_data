"""Frozen sentiment + independently trained aspect inference and transparent issue rules."""

import json
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.foody.aspect import MODEL as ASPECT_MODEL
from src.foody.schema import ENRICHED, PROCESSED, REPORTS, digest, save_json, save_parquet

SENTIMENT_MODEL = Path("models/tfidf_logreg.joblib")
# Issues are complaint signals gated by overall negative sentiment + the relevant aspect.
# Phrase matches are deliberately narrow; no issue accuracy or aspect sentiment claim.
ISSUE_RULES = {
    "food": {
        "food_taste": ["dở", "khó ăn", "quá mặn", "quá nhạt", "không ngon"],
        "food_temperature": ["nguội", "lạnh ngắt"],
        "food_freshness": ["ôi thiu", "không tươi", "hôi", "mốc"],
        "portion_size": ["phần nhỏ", "quá ít", "ít đồ ăn"],
    },
    "service": {
        "slow_service": ["phục vụ chậm", "lên món chậm"],
        "waiting_time": ["chờ lâu", "đợi lâu", "đợi mãi", "chờ mãi"],
        "staff_attitude": ["thái độ tệ", "thái độ kém", "bất lịch sự", "khó chịu", "cau có"],
    },
    "price": {
        "expensive": ["đắt", "mắc", "giá cao"],
        "value_for_money": ["không đáng tiền", "không xứng", "phí tiền"],
    },
    "ambiance": {
        "crowded": ["đông đúc", "chen chúc", "quá đông"],
        "noisy": ["ồn", "ồn ào"],
        "space": ["chật", "bí bách"],
    },
    "cleanliness": {"dirty": ["bẩn", "dơ", "mất vệ sinh", "không sạch"]},
}


def extract_issues(text, aspects, sentiment):
    if sentiment != "negative":
        return []
    lowered = text.casefold()
    found = []
    for aspect in aspects:
        for issue, phrases in ISSUE_RULES.get(aspect, {}).items():
            for phrase in phrases:
                for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lowered):
                    prefix = lowered[max(0, match.start() - 18):match.start()]
                    # Avoid obvious "không đắt"/"chẳng bẩn"; negative phrases such as
                    # "không ngon" remain explicit rules. Sarcasm/mixed clauses unresolved.
                    if not re.search(r"(?:không|chẳng|chưa|không hề)\s+$", prefix):
                        found.append(issue)
                        break
                if issue in found:
                    break
    return sorted(set(found))


def predict_observations(frame, sentiment_model, aspect_bundle):
    eligible = frame.loc[frame.valid_for_ai].drop_duplicates("observation_id").copy()
    sentiment_times, aspect_times = [], []
    sentiments, confidences, aspect_labels, aspect_probabilities = [], [], [], []
    for text in eligible.text:
        start = perf_counter()
        p = sentiment_model.predict_proba([text])[0]
        sentiment_times.append((perf_counter() - start) * 1000)
        index = int(np.argmax(p))
        sentiments.append(str(sentiment_model.classes_[index]))
        confidences.append(float(p[index]))
        start = perf_counter()
        ap = aspect_bundle["pipeline"].predict_proba([text])[0]
        aspect_times.append((perf_counter() - start) * 1000)
        aspect_labels.append([
            name for name, probability in zip(aspect_bundle["classes"], ap, strict=True)
            if probability >= aspect_bundle["threshold"]
        ])
        aspect_probabilities.append(dict(zip(aspect_bundle["classes"], ap.tolist(), strict=True)))
    predictions = pd.DataFrame({
        "observation_id": eligible.observation_id,
        "predicted_sentiment": sentiments, "sentiment_confidence": confidences,
        "predicted_aspects": aspect_labels, "aspect_confidences": aspect_probabilities,
    })
    predictions["detected_issues"] = [
        extract_issues(t, a, s) for t, a, s in zip(
            eligible.text, aspect_labels, sentiments, strict=True
        )
    ]
    def timing(values):
        return {"inference_count": len(values),
                "mean_latency_ms": float(np.mean(values)) if values else None,
                "p95_latency_ms": float(np.percentile(values, 95)) if values else None,
                "total_prediction_seconds": sum(values) / 1000,
                "measurement": "per-observation predict_proba; batch=1; includes TF-IDF transform"}
    return predictions, {"sentiment": timing(sentiment_times), "aspect": timing(aspect_times)}


def enrich():
    if not ASPECT_MODEL.exists():
        raise ValueError("Run train-aspect-baseline explicitly first; enrichment never trains")
    frame = pd.read_parquet(PROCESSED)
    sentiment_sha, aspect_sha = digest(SENTIMENT_MODEL), digest(ASPECT_MODEL)
    selection = json.loads(Path("reports/aspect_classifier/selection.json").read_text("utf-8"))
    if selection["model_sha256"] != aspect_sha:
        raise ValueError("Aspect model differs from validation-selected artifact")
    started = perf_counter()
    predictions, timing = predict_observations(
        frame, joblib.load(SENTIMENT_MODEL), joblib.load(ASPECT_MODEL)
    )
    # Many-to-one join preserves every raw row and its own rating/timestamp/user provenance.
    enriched = frame.merge(predictions, on="observation_id", how="left", validate="many_to_one")
    for column in ("predicted_aspects", "detected_issues"):
        enriched[column] = enriched[column].map(lambda x: x if isinstance(x, list) else [])
    enriched["aspect_confidences"] = enriched.aspect_confidences.map(
        lambda x: x if isinstance(x, dict) else None
    )
    enriched["sentiment_model_name"] = "vn-msasar-tfidf-logreg"
    enriched["sentiment_model_version"] = sentiment_sha
    enriched["aspect_model_name"] = "vn-msasar-tfidf-ovr-aspect-v1"
    enriched["aspect_model_version"] = aspect_sha
    enriched["issue_extraction_method"] = "heuristic_v1"
    save_parquet(ENRICHED, enriched)
    if digest(SENTIMENT_MODEL) != sentiment_sha or digest(ASPECT_MODEL) != aspect_sha:
        raise ValueError("Model changed during inference")
    unique = enriched.loc[enriched.valid_for_ai].drop_duplicates("review_id")
    report = {
        "rows_preserved": len(enriched), "unique_valid_reviews": len(unique),
        "inference_observations": len(predictions), "latency": timing,
        "total_enrichment_seconds": perf_counter() - started,
        "sentiment_distribution_source_rows": enriched.predicted_sentiment.value_counts().to_dict(),
        "sentiment_distribution_unique_reviews": unique.predicted_sentiment.value_counts().to_dict(),
        "aspect_distribution_unique_reviews": unique.predicted_aspects.explode().value_counts().to_dict(),
        "issue_distribution_unique_reviews": unique.detected_issues.explode().value_counts().to_dict(),
        "sentiment_model_sha256": sentiment_sha, "aspect_model_sha256": aspect_sha,
        "processed_sha256": digest(PROCESSED), "enriched_sha256": digest(ENRICHED),
        "confidence": "Uncalibrated predict_proba scores; not measured Foody correctness",
        "foody_accuracy": None, "issue_method": "heuristic_v1; negative sentiment and predicted aspect gates",
    }
    save_json(REPORTS / "enrichment.json", report)
    save_json(REPORTS / "issue_rules.json", ISSUE_RULES)
    return report

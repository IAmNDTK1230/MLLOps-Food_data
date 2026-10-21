"""CLI orchestration and evidence for the first-party Foody product dataset."""

import json
from pathlib import Path

import pandas as pd

from src.foody.schema import ENRICHED, REPORTS, SOURCE, digest, json_default, save_json

COMMANDS = {
    "foody-prepare", "train-aspect-baseline", "foody-enrich", "foody-build-mart",
    "foody-prepare-rag", "run-foody-intelligence",
}


def prepare_rag():
    frame = pd.read_parquet(ENRICHED)
    valid = frame.loc[frame.valid_for_ai & frame.predicted_sentiment.notna()].copy()
    # One document per deterministic review identity, not per CSV duplicate.
    lineage = valid.groupby("review_id").source_row.agg(list)
    valid = valid.drop_duplicates("review_id")
    path = Path("data/foody/rag/documents.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_id", "restaurant_key", "restaurant_name", "raw_rating",
        "predicted_sentiment", "predicted_aspects", "detected_issues", "source",
        "restaurant_identity_type", "sentiment_model_version", "aspect_model_version",
    ]
    with path.open("w", encoding="utf-8") as stream:
        for row in valid.to_dict("records"):
            metadata = {name: row[name] for name in fields}
            timestamp = row["review_timestamp"]
            metadata["review_timestamp"] = timestamp.isoformat() if pd.notna(timestamp) else None
            metadata["timestamp_timezone"] = "Asia/Ho_Chi_Minh assumed"
            metadata["source_rows"] = lineage[row["review_id"]]
            document = {"id": row["review_id"], "text": row["text"], "metadata": metadata}
            stream.write(json.dumps(document, ensure_ascii=False, default=json_default) + "\n")
    result = {"document_count": len(valid), "sha256": digest(path),
              "policy": "one valid distinct review_id per document; duplicate CSV rows in metadata",
              "embeddings_built": False, "vector_database_built": False}
    save_json(REPORTS / "rag.json", result)
    return result


def protected_files():
    files = [SOURCE, Path("models/tfidf_logreg.joblib"),
             Path("data/food/splits/reviews.parquet")]
    for root in (
        "reports/foody", "reports/foody_v2", "reports/foody_cleaning",
        "data/foody/audit", "data/foody/audit_v2", "databricks",
    ):
        files += [p for p in Path(root).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += list(Path("data/foody/source").glob("review*"))
    return {str(path): digest(path) for path in files if path.is_file()}


def make_report():
    names = ("cleaning", "enrichment", "mart", "rag")
    reports = {
        name: json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8")) for name in names
    }
    aspect = json.loads(Path("reports/aspect_classifier/test_metrics.json").read_text("utf-8"))
    versions = {
        "source_sha256": reports["cleaning"]["source_sha256"],
        "processed_sha256": reports["cleaning"]["processed_sha256"],
        "enriched_sha256": reports["enrichment"]["enriched_sha256"],
        "sentiment_model_sha256": reports["enrichment"]["sentiment_model_sha256"],
        "aspect_model_sha256": reports["enrichment"]["aspect_model_sha256"],
        "rag_sha256": reports["rag"]["sha256"],
        "research_split_sha256": aspect["split_sha256"],
        "pipeline_code_sha256": {p.name: digest(p) for p in Path("src/foody").glob("*.py")},
    }
    save_json(REPORTS / "versions.json", versions)
    content = """# First-party Foody intelligence data report

## Data roles and scientific scope

VN-MSASar v3 is the research training/validation/test benchmark. Its frozen sentiment
model is reused without fitting. The new aspect model is trained only on VN-MSASar.
Foody `foody_reviews.csv` is the user-designated first-party real-world inference source.
The previous `review.csv` and `review_utf8_20261020.csv` are external ShopeeFood-related
supplementary data. Their historical audits remain unchanged and their unresolved ID
and Classification provenance do not gate this independent first-party pipeline.

Foody has no trusted sentiment/aspect labels: its predictions do not establish accuracy.
Rating is business metadata on its original 0–10 scale, never a sentiment target.
Restaurant identity is `derived_from_restaurant_name`, not an official Foody ID.
NFC, trimming, whitespace collapse and casefold apply to key generation only; no fuzzy
merges. Identical names can still refer to different physical branches. Future official
IDs/URLs should be stored in an explicit mapping table; retain historical keys/lineage.

## Cleaning and observation semantics

All source rows and original values remain in processed/enriched Parquet.
Empty/U+FFFD text is excluded from AI, never repaired. Literal question marks are
diagnostic punctuation flags, not automatic corruption. See suspicious-text contexts
and manual inspection notes under reports/foody_intelligence.
Text cleaning uses NFC and whitespace normalization, preserving case, accents, emoji
and punctuation. Dates parse only observed day/month/year minute/second formats;
unparseable dates remain null. Source timestamps are assumed local Vietnam time.
Inference runs once per restaurant_key + normalized text, then joins predictions back
to all source rows. Source row lists and duplicate_count preserve lineage.
Review IDs hash a versioned JSON tuple of key, raw User, raw time and original Comment.
User is not a unique user ID. Marts and RAG use distinct review_id: repeated CSV rows
do not inflate metrics, while different users/dates remain distinct observations.

## Inference and limitations

Confidence values are uncalibrated model probabilities. Aspect labels are the union
of available annotations for a normalized research text; incomplete aspect annotation
can induce false-negative targets. Rare aspect metrics need support-aware interpretation.
Issues require overall predicted negative sentiment, the corresponding predicted aspect,
and a listed Vietnamese phrase. Simple negation is screened. This heuristic misses mixed
sentiment, sarcasm and implicit complaints; no supervised issue accuracy is claimed.
Complaint-rate metrics combine overall sentiment with aspect presence; they are not
aspect-level sentiment predictions. Sparse and selectively crawled reviews are not
representative population samples. Alerts compare adjacent historical calendar months
with sufficient observations in both periods and exclude the final observed month.
They are demonstrations of rules, not current operational alerts or causal conclusions.

## Measured outputs
"""
    for name, report in {**reports, "aspect_research_benchmark": aspect, "versions": versions}.items():
        content += f"\n### {name}\n\n```json\n{json.dumps(report, indent=2, ensure_ascii=False)}\n```\n"
    content += """
## Execution

Use the project virtual environment:

```
python -m src.cli foody-prepare
python -m src.cli train-aspect-baseline
python -m src.cli foody-enrich
python -m src.cli foody-build-mart --minimum-samples 10
python -m src.cli foody-prepare-rag
python -m src.cli run-foody-intelligence
```

The full pipeline requires an already trained aspect artifact and never trains either
model. Repeating train-aspect-baseline returns completed metrics without retraining or
test reinference. Test predictions are checkpointed once after validation selection.

Next: evaluate Vietnamese embedding/retrieval quality using the prepared review documents
and restaurant filters, then build a source-citing RAG service and Streamlit interface.
No embeddings, vector database or Streamlit app are built by this task.
"""
    Path("reports/FOODY_INTELLIGENCE_DATA_REPORT.md").write_text(content, encoding="utf-8")
    return reports


def run_all(minimum_samples=10, rate_increase=0.2, rating_decrease=1.0):
    from src.foody.analytics import build_mart
    from src.foody.aspect import MODEL
    from src.foody.cleaning import prepare
    from src.foody.enrichment import enrich

    if not MODEL.exists():
        raise ValueError("Train aspect baseline explicitly first; run-foody-intelligence never trains")
    before = protected_files()
    prepare()
    enrich()
    build_mart(minimum_samples, rate_increase, rating_decrease)
    prepare_rag()
    after = protected_files()
    if before != after:
        raise ValueError("Protected source/research/historical audit changed")
    save_json(REPORTS / "preservation_checks.json", {"passed": True, "sha256": after})
    return make_report()


def add_commands(commands):
    for name in sorted(COMMANDS):
        parser = commands.add_parser(name, help="First-party Foody intelligence / research aspect baseline")
        if name in {"foody-build-mart", "run-foody-intelligence"}:
            parser.add_argument("--minimum-samples", type=int, default=10)
            parser.add_argument("--rate-increase", type=float, default=0.2)
            parser.add_argument("--rating-decrease", type=float, default=1.0)


def run_command(args):
    from src.foody.analytics import build_mart
    from src.foody.aspect import train
    from src.foody.cleaning import prepare
    from src.foody.enrichment import enrich

    actions = {
        "foody-prepare": prepare, "train-aspect-baseline": train,
        "foody-enrich": enrich, "foody-prepare-rag": prepare_rag,
        "foody-build-mart": lambda: build_mart(
            args.minimum_samples, args.rate_increase, args.rating_decrease
        ),
        "run-foody-intelligence": lambda: run_all(
            args.minimum_samples, args.rate_increase, args.rating_decrease
        ),
    }
    result = actions[args.command]()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
    return 0

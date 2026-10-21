"""Assemble presentation numbers from frozen machine-readable artifacts."""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from src.foody.schema import save_json

ROOT = Path("reports/final")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_by(path, key, value):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return next(row for row in csv.DictReader(stream) if row[key] == value)


def test_count(path):
    root = ET.parse(path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    return int(suite.attrib["tests"]), int(suite.attrib.get("failures", 0)), int(
        suite.attrib.get("errors", 0),
    )


def build():
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = {
        "dataset": "reports/food_dataset_report.json",
        "split": "data/food/splits/metadata.json",
        "rule": "reports/baselines/rule_based_test_metrics.json",
        "tfidf": "reports/baselines/tfidf_test_metrics.json",
        "qwen_v1": "reports/proposed_v1/qwen_test_metrics.json",
        "qwen_v2a": "reports/proposed_v2/test_metrics.json",
        "aspect": "reports/aspect_classifier/test_metrics.json",
        "foody": "reports/foody_intelligence/enrichment.json",
        "restaurant_mart": "data/foody/mart/restaurant_summary.parquet",
        "chroma": "data/foody/vector_store/index_manifest.json",
        "bm25": "data/foody/lexical_index/index_manifest.json",
        "rag_retrieval": "reports/rag_v3/retrieval_policies.csv",
        "rag_generation": "reports/rag_v3/generation_metrics.json",
        "qwen_1_5b_provenance": "models/qwen_v1/provenance.json",
        "qwen_3b_provenance": "models/qwen_v3/provenance.json",
        "tests": str(ROOT / "test_results.xml"),
    }
    data = {key: read(path) for key, path in sources.items()
            if Path(path).suffix == ".json"}
    rows = {name: row_by(sources["rag_retrieval"], "policy", policy) for name, policy in (
        ("v1", "RAG_V1_DENSE"), ("v2", "RAG_V2_EVIDENCE_RERANK"),
        ("v3", "RAG_V3_ROUTED"),
    )}
    tests, failures, errors = test_count(sources["tests"])
    from pyarrow.parquet import read_metadata

    numbers = {
        "research": {
            "vn_msasar_filtered_rows": data["dataset"]["filtered_vietnamese_restaurant_rows"],
            "split_counts": data["split"]["counts"],
            "class_distributions": data["split"]["class_distributions"],
            "split_leakage_checks": data["split"]["leakage_checks"],
            "sentiment_tfidf_test_macro_f1": data["tfidf"]["macro_f1"],
            "qwen_v1_test_macro_f1": data["qwen_v1"]["macro_f1"],
            "qwen_v2a_test_macro_f1": data["qwen_v2a"]["macro_f1"],
            "aspect_test_micro_f1": data["aspect"]["micro_f1"],
            "aspect_test_macro_f1": data["aspect"]["macro_f1"],
        },
        "foody": {
            "raw_review_count": data["foody"]["rows_preserved"],
            "ai_valid_count": data["foody"]["inference_observations"],
            "unique_valid_reviews": data["foody"]["unique_valid_reviews"],
            "restaurant_count": read_metadata(sources["restaurant_mart"]).num_rows,
        },
        "rag": {
            "document_count": data["chroma"]["document_count"],
            "vector_count_manifest": data["chroma"]["document_count"],
            "bm25_count_manifest": data["bm25"]["document_count"],
            "collection": data["chroma"]["collection"],
            "retrieval": {name: {"known_positive_hit_at_5": float(row["known_hit_at_5"]),
                                 "known_positive_hit_at_10": float(row["known_hit_at_10"]),
                                 "known_positive_mrr": float(row["known_mrr"]),
                                 "mean_latency_ms": float(row["mean_latency_ms"])}
                          for name, row in rows.items()},
            "v3_generator_counts": {name: item["counts"] for name, item in
                                    data["rag_generation"].items()},
        },
        "models": {
            "qwen_1_5b": {key: data["qwen_1_5b_provenance"].get(key) for key in
                          ("model", "revision", "model_sha256", "model_size_bytes",
                           "runtime", "runtime_tag")},
            "qwen_3b": {key: data["qwen_3b_provenance"].get(key) for key in
                        ("model", "revision", "model_sha256", "model_size_bytes",
                         "runtime", "runtime_tag", "quantization")},
        },
        "verification": {"tests": tests, "failures": failures, "errors": errors},
        "sources": sources,
        "caveats": ["RAG known-positive labels are incomplete for six queries.",
                    "Foody predictions have no trusted gold-label accuracy.",
                    "Qwen V2-A reused the viewed test split and is exploratory."],
    }
    save_json(ROOT / "KEY_NUMBERS.json", numbers)

    fields = ("task", "model", "dataset", "input", "accuracy", "macro_f1", "micro_f1",
              "neutral_f1", "retrieval_hit5", "retrieval_hit10", "MRR", "latency_ms",
              "status", "notes")
    records = []
    neutral_reports = {
        "rule": "reports/baselines/rule_based_classification_report.csv",
        "tfidf": "reports/baselines/tfidf_classification_report.csv",
    }
    for key, name in (("rule", "Rule-Based Sentiment"),
                      ("tfidf", "TF-IDF Sentiment"),
                      ("qwen_v1", "Qwen V1"), ("qwen_v2a", "Qwen V2-A")):
        metric = data[key]
        neutral = (float(row_by(neutral_reports[key], "label", "neutral")["f1-score"])
                   if key in neutral_reports else metric.get("neutral_f1"))
        records.append({"task": "sentiment", "model": name, "dataset": "VN-MSASar v3 test",
                        "input": "text", "accuracy": metric.get("accuracy"),
                        "macro_f1": metric.get("macro_f1"), "neutral_f1": neutral,
                        "latency_ms": metric.get("mean_latency_ms"),
                        "status": "exploratory" if key == "qwen_v2a" else "frozen_test",
                        "notes": "Viewed test reused; requires external confirmation" if
                        key == "qwen_v2a" else ""})
    records.append({"task": "aspect", "model": "Aspect TF-IDF",
                    "dataset": "VN-MSASar v3 aspect test", "input": "text",
                    "macro_f1": data["aspect"]["macro_f1"],
                    "micro_f1": data["aspect"]["micro_f1"], "status": "frozen_test"})
    for key, label in (("v1", "RAG V1"), ("v2", "RAG V2"), ("v3", "RAG V3")):
        metric = numbers["rag"]["retrieval"][key]
        records.append({"task": "retrieval", "model": label,
                        "dataset": "Foody six-query known-positive set", "input": "question + reviews",
                        "retrieval_hit5": metric["known_positive_hit_at_5"],
                        "retrieval_hit10": metric["known_positive_hit_at_10"],
                        "MRR": metric["known_positive_mrr"],
                        "latency_ms": metric["mean_latency_ms"],
                        "status": "accepted_demo" if key == "v3" else "research_baseline",
                        "notes": "Incomplete known positives; latency from one shared policy run"})
    with (ROOT / "MODEL_COMPARISON.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(ROOT / "KEY_NUMBERS.json")
    print(ROOT / "MODEL_COMPARISON.csv")


if __name__ == "__main__":
    build()

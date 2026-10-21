"""Compare V2 retrieval variants on the frozen six-query known-positive benchmark."""

import json
from pathlib import Path

import pandas as pd

from src.foody.schema import digest, save_json
from src.rag.cli import selected_retriever
from src.rag.evaluation import development, known_positive_scores
from src.rag.schemas import RetrievalRequest
from src.rag_v2.hybrid import CANDIDATES, HybridRetriever
from src.rag_v2.lexical import BM25Retriever, build_index

REPORTS = Path("reports/rag_v2")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    build_index()
    dense = selected_retriever()
    dense.monitor = False
    lexical = BM25Retriever()
    cases = development()["cases"]
    rows = []
    for config in CANDIDATES:
        retriever = HybridRetriever(dense, lexical, config)
        for case in cases:
            hits, elapsed, debug = retriever.search_debug(
                RetrievalRequest(query=case["query"], top_k=10)
            )
            metrics = known_positive_scores([hit.review_id for hit in hits],
                                            case["known_relevant_review_ids"])
            rows.append({"candidate": config.name, "query": case["query"], **metrics,
                         "latency_ms": elapsed, "candidate_count": debug["candidate_count"],
                         "top_review_ids": json.dumps([hit.review_id for hit in hits]),
                         "effective_filters": json.dumps(debug["effective_filters"]),
                         "expanded_query_terms": json.dumps(debug["expanded_query_terms"],
                                                            ensure_ascii=False)})
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORTS / "retrieval_candidate_rows.csv", index=False)
    summary = frame.groupby("candidate", sort=False).agg(
        known_positive_hit_at_5=("known_positive_recall_at_5", "mean"),
        known_positive_hit_at_10=("known_positive_recall_at_10", "mean"),
        known_positive_mrr=("known_positive_mrr", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", lambda x: x.quantile(.95)),
    ).reset_index()
    summary["evaluation_scope"] = "six manually labeled queries, incomplete known positives"
    summary.to_csv(REPORTS / "retrieval_candidates.csv", index=False)
    save_json(REPORTS / "evaluation_metadata.json", {
        "source_sha256": digest("data/foody/rag/documents.jsonl"),
        "benchmark_sha256": digest("reports/rag/development_judgments.json"),
        "benchmark_caveat": development()["completeness"],
        "lexical_backend": "SQLite FTS5 BM25 (unicode61 remove_diacritics 2)",
        "reranker": "local deterministic evidence-aware scoring; no cross-encoder model",
        "selection_rule": "known-positive Hit@5/Hit@10, then manual semantic relevance and latency; MRR reported separately",
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

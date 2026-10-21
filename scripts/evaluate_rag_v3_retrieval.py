"""Compare four frozen local retrieval policies and explicit intent routing."""

import json
from pathlib import Path

import pandas as pd

from src.foody.schema import save_json
from src.rag.cli import selected_retriever
from src.rag.evaluation import development, known_positive_scores
from src.rag.schemas import RetrievalRequest
from src.rag_v2.hybrid import BM25_ONLY, HYBRID_EVIDENCE, HYBRID_RRF, V1_DENSE, HybridRetriever
from src.rag_v2.lexical import BM25Retriever, build_index
from src.rag_v3.retriever import RoutedRetriever
from src.rag_v3.routing import route_query

REPORTS = Path("reports/rag_v3")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    build_index()
    dense = selected_retriever()
    dense.monitor = False
    lexical = BM25Retriever()
    policies = {config.name: HybridRetriever(dense, lexical, config)
                for config in (BM25_ONLY, V1_DENSE, HYBRID_RRF, HYBRID_EVIDENCE)}
    policies["RAG_V3_ROUTED"] = RoutedRetriever(dense, lexical)
    rows = []
    for name, retriever in policies.items():
        for case in development()["cases"]:
            request = RetrievalRequest(query=case["query"], top_k=10)
            hits, latency, debug = retriever.search_debug(request)
            scores = known_positive_scores([hit.review_id for hit in hits],
                                           case["known_relevant_review_ids"])
            rows.append({"policy": name, "query": case["query"],
                         "route": route_query(case["query"]).category,
                         "actual_retrieval_method": debug["retrieval_method"],
                         **scores, "latency_ms": latency,
                         "top_review_ids": json.dumps([hit.review_id for hit in hits]),
                         "effective_filters": json.dumps(debug["effective_filters"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORTS / "retrieval_policy_rows.csv", index=False)
    summary = frame.groupby("policy", sort=False).agg(
        known_hit_at_5=("known_positive_recall_at_5", "mean"),
        known_hit_at_10=("known_positive_recall_at_10", "mean"),
        known_mrr=("known_positive_mrr", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", lambda values: values.quantile(.95)),
    ).reset_index()
    summary.to_csv(REPORTS / "retrieval_policies.csv", index=False)
    save_json(REPORTS / "retrieval_policy_notes.json", {
        "benchmark": "same six frozen queries and incomplete known-relevant review IDs",
        "selection_rule": "known-positive metrics, manual relevance, then latency",
        "routed_rule": "specific aspect complaint -> BM25 with negative/aspect filter; "
        "general aspect or summary -> RRF hybrid; trend causation -> mart guard",
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

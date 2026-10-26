"""Known-positive retrieval check for only the new review corpus."""

import json
from pathlib import Path

from src.foody.review_migration import DOCUMENTS
from src.foody.review_runtime import LEXICAL, VECTOR
from src.foody.schema import digest, save_json
from src.rag.embeddings import CANDIDATES, LocalEmbeddings
from src.rag.retriever import Retriever
from src.rag.schemas import RetrievalRequest
from src.rag.vector_store import open_collection, read_documents
from src.rag_v2.lexical import BM25Retriever

JUDGMENTS = Path("reports/rag/review_2025_2026_known_relevance.json")
REPORT = Path("reports/review/retrieval_benchmark.json")


def score(hits, relevant):
    ids = [hit.review_id for hit in hits]
    positions = [ids.index(review_id) + 1 for review_id in relevant if review_id in ids]
    return {"hit_at_5": float(any(position <= 5 for position in positions)),
            "mrr_at_10": 1 / min(positions) if positions else 0.0}


def main():
    judgments = json.loads(JUDGMENTS.read_text("utf-8"))
    owners = {doc["id"]: str(doc["metadata"]["restaurant_id"])
              for doc in read_documents(DOCUMENTS)}
    for case in judgments["cases"]:
        if any(owners.get(review_id) != case["restaurant_id"]
               for review_id in case["known_relevant_review_ids"]):
            raise ValueError("Benchmark judgment references a missing or wrong-restaurant review")
    collection, manifest = open_collection(VECTOR / "index_manifest.json", VECTOR)
    spec = next(item for item in CANDIDATES if item.name == manifest["model_name"])
    dense = Retriever(LocalEmbeddings(spec, local_files_only=True), collection, manifest,
                      monitor=False)
    lexical = BM25Retriever(LEXICAL / "reviews.sqlite", LEXICAL / "index_manifest.json",
                            DOCUMENTS)
    cases = []
    for case in judgments["cases"]:
        request = RetrievalRequest(query=case["query"], restaurant_id=case["restaurant_id"],
                                   top_k=10)
        records = {"query": case["query"], "restaurant_id": case["restaurant_id"]}
        for name, retriever in (("dense", dense), ("bm25", lexical)):
            hits, latency = retriever.search(request)
            if any(hit.metadata.get("restaurant_id") != case["restaurant_id"] for hit in hits):
                raise ValueError("Retriever returned evidence from another restaurant ID")
            records[name] = {**score(hits, case["known_relevant_review_ids"]),
                             "latency_ms": latency,
                             "top_review_ids": [hit.review_id for hit in hits]}
        cases.append(records)
    report = {
        "dataset_version": judgments["dataset_version"],
        "documents_sha256": digest(DOCUMENTS), "judgments_sha256": digest(JUDGMENTS),
        "judgment_scope": judgments["judgment_scope"], "case_count": len(cases),
        "summary": {name: {metric: sum(case[name][metric] for case in cases) / len(cases)
                           for metric in ("hit_at_5", "mrr_at_10", "latency_ms")}
                    for name in ("dense", "bm25")},
        "cases": cases,
    }
    save_json(REPORT, report)
    print(json.dumps(report["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()

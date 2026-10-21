"""Small known-positive retrieval comparison and read-only full-index evaluation."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.foody.schema import digest, save_json
from src.rag.embeddings import CANDIDATES, LocalEmbeddings
from src.rag.retriever import Retriever
from src.rag.schemas import RetrievalRequest
from src.rag.vector_store import DOCUMENTS, MANIFEST, read_documents

REPORTS = Path("reports/rag")
SELECTION = REPORTS / "embedding_selection.json"


def known_positive_scores(ids, positives):
    positions = [ids.index(review_id) + 1 for review_id in positives if review_id in ids]
    return {
        "known_positive_recall_at_5": sum(position <= 5 for position in positions) / len(positives),
        "known_positive_recall_at_10": sum(position <= 10 for position in positions) / len(positives),
        "known_positive_mrr": 1 / min(positions) if positions else 0.0,
    }


def development():
    path = REPORTS / "development_judgments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["source_sha256"] != digest(DOCUMENTS):
        raise ValueError("Development judgments no longer match RAG source")
    return data


def pool_documents(docs, cases, size=600):
    positive_ids = {item for case in cases for item in case["known_relevant_review_ids"]}
    rng = np.random.default_rng(42)
    extra = rng.choice(len(docs), size=min(size, len(docs)), replace=False)
    selected = {docs[int(i)]["id"] for i in extra} | positive_ids
    return [doc for doc in docs if doc["id"] in selected]


def model_files(embedder):
    files = sorted(embedder.cache_dir.rglob("*.onnx"))
    if not files:
        raise ValueError("FastEmbed ONNX model was not stored in the configured cache")
    # Only model paths for this instance, not unrelated candidates in shared cache.
    path = Path(getattr(embedder.model, "model_path", ""))
    if not path.is_file():
        path = files[-1]
    return path


def compare_candidates():
    if SELECTION.exists():
        return json.loads(SELECTION.read_text(encoding="utf-8"))
    development_set = development()
    cases = development_set["cases"]
    pool = pool_documents(list(read_documents()), cases)
    records, summaries = [], []
    for spec in CANDIDATES:
        try:
            embedder = LocalEmbeddings(spec)
            vectors, embed_seconds = embedder.documents([doc["text"] for doc in pool])
            queries, query_seconds = embedder.queries([case["query"] for case in cases])
            for case, vector in zip(cases, queries, strict=True):
                started = perf_counter()
                scores = vectors @ vector
                rank = np.argsort(-scores)[:10]
                ids = [pool[int(index)]["id"] for index in rank]
                metrics = known_positive_scores(ids, case["known_relevant_review_ids"])
                records.append({"model": spec.name, "query": case["query"], **metrics,
                                "retrieval_latency_ms": (perf_counter() - started) * 1000,
                                "top_review_ids": json.dumps(ids),
                                "judgment_scope": "known positives within fixed 600-document development pool"})
            own = [row for row in records if row["model"] == spec.name]
            path = model_files(embedder)
            summaries.append({"model": spec.name, "onnx_source_repo": spec.repo,
                              "revision": path.parent.name,
                              "dimension": spec.dimension, "model_file": str(path),
                              "model_hash": digest(path), "model_size_mb": path.stat().st_size / 1e6,
                              "device": embedder.device, "pool_documents": len(pool),
                              "embedding_seconds": embed_seconds,
                              "documents_per_second": len(pool) / embed_seconds,
                              "mean_query_embedding_ms": query_seconds * 1000 / len(cases),
                              **{key: float(np.mean([item[key] for item in own])) for key in (
                                  "known_positive_recall_at_5", "known_positive_recall_at_10",
                                  "known_positive_mrr", "retrieval_latency_ms")}})
        except Exception as error:
            summaries.append({"model": spec.name, "onnx_source_repo": spec.repo,
                              "status": "UNAVAILABLE",
                              "error_type": type(error).__name__})
    pd.DataFrame(records).to_csv(REPORTS / "retrieval_development.csv", index=False)
    pd.DataFrame(summaries).to_csv(REPORTS / "embedding_comparison.csv", index=False)
    eligible = [row for row in summaries if "model_hash" in row]
    if not eligible:
        raise RuntimeError("No multilingual embedding candidate could run locally")
    selected = max(eligible, key=lambda row: (
        row["known_positive_recall_at_10"], row["known_positive_mrr"],
        -row["mean_query_embedding_ms"],
    ))
    selected = {**selected, "selection_rule": "known-positive recall@10, then MRR, then query time",
                "benchmark_scope": development_set["completeness"]}
    save_json(SELECTION, selected)
    return selected


def evaluate_index():
    cases = development()["cases"]
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec = next(item for item in CANDIDATES if item.name == metadata["model_name"])
    embedder = LocalEmbeddings(spec, local_files_only=True)
    retriever = Retriever(embedder, monitor=False)
    rows = []
    for case in cases:
        hits, latency = retriever.search(RetrievalRequest(query=case["query"], top_k=10))
        metrics = known_positive_scores([item.review_id for item in hits],
                                        case["known_relevant_review_ids"])
        rows.append({"query": case["query"], **metrics,
                     "retrieval_latency_ms": latency,
                     "retrieved_review_ids": json.dumps([item.review_id for item in hits]),
                     "judgment_scope": "incomplete known positives against full index"})
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORTS / "retrieval_evaluation.csv", index=False)
    result = {"document_count": metadata["document_count"],
              "queries": len(cases), "evaluation_type": "single-reviewer incomplete known positives",
              "retrieval_recall_at_5": None, "retrieval_recall_at_10": None,
              "known_positive_recall_at_5": float(frame.known_positive_recall_at_5.mean()),
              "known_positive_recall_at_10": float(frame.known_positive_recall_at_10.mean()),
              "known_positive_mrr": float(frame.known_positive_mrr.mean()),
              "mean_retrieval_latency_ms": float(frame.retrieval_latency_ms.mean()),
              "p95_retrieval_latency_ms": float(frame.retrieval_latency_ms.quantile(0.95)),
              "context_precision": None, "context_recall": None,
              "answer_relevance": None, "faithfulness": None,
              "evidence_correctness": None, "end_to_end_latency_ms": None,
              "caveat": development()["completeness"]}
    save_json(REPORTS / "rag_metrics.json", result)
    return result

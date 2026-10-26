"""Dense/BM25 retrieval with explicit RRF, limited expansion and evidence reranking."""

import json
from dataclasses import dataclass
from time import perf_counter

from src.rag.retriever import where_filter
from src.rag.schemas import RetrievalRequest
from src.rag_v2.intent import detect, evidence_compatible, issue_matches
from src.rag_v2.lexical import BM25Retriever


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    dense_top_n: int = 20
    lexical_top_n: int = 20
    rrf_constant: int = 60
    final_top_k: int = 5
    use_dense: bool = True
    use_lexical: bool = True
    expand: bool = False
    auto_filter: bool = False
    evidence_rerank: bool = False


V1_DENSE = RetrievalConfig("RAG_V1_DENSE", use_lexical=False, auto_filter=False)
BM25_ONLY = RetrievalConfig("RAG_V2_BM25", use_dense=False)
HYBRID_RRF = RetrievalConfig("RAG_V2_RRF")
HYBRID_EXPANDED = RetrievalConfig("RAG_V2_RRF_EXPANDED", expand=True, auto_filter=True)
HYBRID_EVIDENCE = RetrievalConfig("RAG_V2_EVIDENCE_RERANK", expand=True,
                                  auto_filter=True, evidence_rerank=True)
CANDIDATES = (V1_DENSE, BM25_ONLY, HYBRID_RRF, HYBRID_EXPANDED, HYBRID_EVIDENCE)


def effective_request(request, intent, auto_filter):
    if not auto_filter or intent.intent != "complaint":
        return request
    changes = {}
    if request.aspect is None and intent.aspect:
        changes["aspect"] = intent.aspect
    if request.sentiment is None:
        changes["sentiment"] = "negative"
    return request.model_copy(update=changes)


def _matches(hit, request):
    meta = hit.metadata
    if request.restaurant_id and meta.get("restaurant_id") != str(request.restaurant_id):
        return False
    if request.restaurant_key and meta.get("restaurant_key") != request.restaurant_key:
        return False
    if request.sentiment and meta["predicted_sentiment"] != request.sentiment:
        return False
    if request.aspect and request.aspect not in json.loads(meta["predicted_aspects"]):
        return False
    if request.date_from or request.date_to:
        conditions = where_filter(request)
        parts = conditions.get("$and", [conditions])
        stamp = meta.get("timestamp_epoch")
        if stamp is None:
            return False
        for part in parts:
            if "timestamp_epoch" not in part:
                continue
            for op, value in part["timestamp_epoch"].items():
                if op == "$gte" and stamp < value:
                    return False
                if op == "$lt" and stamp >= value:
                    return False
                if op == "$lte" and stamp > value:
                    return False
    return True


def rrf_fuse(rankings, constant=60):
    scores, hits = {}, {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, 1):
            scores[hit.review_id] = scores.get(hit.review_id, 0.0) + 1 / (constant + position)
            hits.setdefault(hit.review_id, hit)
    return [(hits[key], score) for key, score in
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


class HybridRetriever:
    def __init__(self, dense, lexical=None, config=HYBRID_EVIDENCE):
        self.dense = dense
        self.lexical = lexical or BM25Retriever()
        self.config = config
        self.metadata = dense.metadata
        self.last_debug = None

    def search(self, request):
        hits, latency, _ = self.search_debug(request)
        return hits, latency

    def search_debug(self, request):
        if not isinstance(request, RetrievalRequest):
            request = RetrievalRequest.model_validate(request)
        started = perf_counter()
        query_intent = detect(request.query)
        effective = effective_request(request, query_intent, self.config.auto_filter)
        expanded = " ".join([request.query, *query_intent.expanded_query_terms]) \
            if self.config.expand else request.query
        rankings, timings = [], {}
        if self.config.use_dense:
            dense_request = effective.model_copy(update={"top_k": self.config.dense_top_n})
            dense_hits, dense_ms = self.dense.search(dense_request)
            rankings.append(dense_hits)
            timings["dense_ms"] = dense_ms
        if self.config.use_lexical:
            lexical_hits, lexical_ms = self.lexical.search(effective, query=expanded,
                                                          top_n=self.config.lexical_top_n)
            rankings.append(lexical_hits)
            timings["lexical_ms"] = lexical_ms
        if len(rankings) == 1:
            fused = [(hit, hit.score) for hit in rankings[0]]
        else:
            fused = rrf_fuse(rankings, self.config.rrf_constant)
        candidates = []
        for hit, score in fused:
            if not _matches(hit, effective):
                raise ValueError("Hybrid candidate violates metadata filters")
            if self.config.evidence_rerank:
                # Small, transparent second-stage boost; incompatible evidence is penalized.
                if query_intent.aspect and query_intent.aspect in json.loads(
                    hit.metadata["predicted_aspects"]
                ):
                    score += 0.004
                if query_intent.intent in ("complaint", "praise"):
                    score += 0.006 if evidence_compatible(
                        hit, query_intent.intent, query_intent.aspect
                    ) else -0.012
                issue = issue_matches(hit.text, request.query)
                if issue is not None:
                    score += 0.006 if issue else -0.006
            candidates.append((hit, score))
        candidates.sort(key=lambda item: (-item[1], item[0].review_id))
        hits = [hit.model_copy(update={"score": score})
                for hit, score in candidates[:request.top_k]]
        elapsed_ms = (perf_counter() - started) * 1000
        debug = {"retrieval_version": "RAG_V2", "retrieval_method": self.config.name,
                 "original_query": request.query,
                 "expanded_query_terms": list(query_intent.expanded_query_terms)
                 if self.config.expand else [],
                 "detected_intent": query_intent.intent,
                 "detected_aspect": query_intent.aspect,
                 "effective_filters": effective.model_dump(exclude={"query", "top_k"},
                                                          exclude_none=True, mode="json"),
                 "dense_top_n": self.config.dense_top_n if self.config.use_dense else 0,
                 "lexical_top_n": self.config.lexical_top_n if self.config.use_lexical else 0,
                 "rrf_constant": self.config.rrf_constant if len(rankings) == 2 else None,
                 "candidate_count": len(candidates), "final_top_k": request.top_k,
                 "timings": timings, "latency_ms": elapsed_ms}
        self.last_debug = debug
        return hits, elapsed_ms, debug

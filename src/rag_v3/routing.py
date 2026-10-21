"""Explicit Vietnamese query routes and retrieval policies."""

from dataclasses import dataclass

from src.rag_v2.hybrid import HYBRID_RRF, RetrievalConfig
from src.rag_v2.intent import detect

BM25_FILTERED = RetrievalConfig("RAG_V3_BM25_FILTERED", use_dense=False,
                                auto_filter=True, evidence_rerank=True)
HYBRID_GENERAL = HYBRID_RRF


@dataclass(frozen=True)
class Route:
    category: str
    intent: str
    aspect: str | None
    retrieval_method: str


def route_query(query):
    detected = detect(query)
    if detected.intent == "causal":
        return Route("TREND_CAUSATION", detected.intent, detected.aspect,
                     "MART_TREND_GUARD")
    if detected.intent == "complaint" and detected.aspect:
        return Route(f"{detected.aspect.upper()}_COMPLAINT", detected.intent,
                     detected.aspect, BM25_FILTERED.name)
    if detected.aspect:
        return Route("GENERAL_ASPECT", detected.intent, detected.aspect,
                     HYBRID_GENERAL.name)
    return Route("GENERAL_SUMMARY", detected.intent, None, HYBRID_GENERAL.name)

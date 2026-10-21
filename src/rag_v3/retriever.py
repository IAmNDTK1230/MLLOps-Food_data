"""Route each V3 query through an evaluated, simple local retrieval policy."""

from src.rag.cli import selected_retriever
from src.rag_v2.hybrid import HybridRetriever
from src.rag_v2.lexical import BM25Retriever, build_index
from src.rag_v3.routing import BM25_FILTERED, HYBRID_GENERAL, route_query


class RoutedRetriever:
    def __init__(self, dense=None, lexical=None):
        self.dense = dense or selected_retriever()
        self.dense.monitor = False
        if lexical is None:
            build_index()
            lexical = BM25Retriever()
        self.lexical = lexical
        self.metadata = self.dense.metadata
        self.policies = {
            BM25_FILTERED.name: HybridRetriever(self.dense, self.lexical, BM25_FILTERED),
            HYBRID_GENERAL.name: HybridRetriever(self.dense, self.lexical, HYBRID_GENERAL),
        }

    def search_debug(self, request):
        decision = route_query(request.query)
        if decision.category == "TREND_CAUSATION":
            return [], 0.0, {"retrieval_version": "RAG_V3",
                             "retrieval_method": "MART_TREND_GUARD",
                             "query_route": decision.category,
                             "candidate_count": 0, "effective_filters": {},
                             "final_top_k": request.top_k}
        hits, elapsed, debug = self.policies[decision.retrieval_method].search_debug(request)
        return hits, elapsed, {**debug, "retrieval_version": "RAG_V3",
                               "query_route": decision.category}

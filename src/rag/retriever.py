"""Filter-first retrieval over the persisted Foody review collection."""

import json
from datetime import timedelta
from time import perf_counter

from src.rag.monitoring import record
from src.rag.schemas import RetrievalRequest, ReviewHit
from src.rag.vector_store import epoch, open_collection


def where_filter(request):
    parts = []
    if request.restaurant_id:
        parts.append({"restaurant_id": str(request.restaurant_id)})
    if request.restaurant_key:
        parts.append({"restaurant_key": request.restaurant_key})
    if request.sentiment:
        parts.append({"predicted_sentiment": request.sentiment})
    if request.aspect:
        parts.append({f"aspect_{request.aspect}": True})
    if request.date_from:
        parts.append({"timestamp_epoch": {"$gte": epoch(request.date_from.isoformat())}})
    if request.date_to:
        end = request.date_to
        if end.hour == end.minute == end.second == 0:
            end += timedelta(days=1)
            parts.append({"timestamp_epoch": {"$lt": epoch(end.isoformat())}})
        else:
            parts.append({"timestamp_epoch": {"$lte": epoch(end.isoformat())}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


class Retriever:
    def __init__(self, embedder, collection=None, index_metadata=None, monitor=True):
        if collection is None or index_metadata is None:
            collection, index_metadata = open_collection()
        if index_metadata["model_name"] != embedder.spec.name:
            raise ValueError("Query embedding model differs from index embedding model")
        self.embedder = embedder
        self.collection = collection
        self.metadata = index_metadata
        self.monitor = monitor

    def search(self, request):
        if not isinstance(request, RetrievalRequest):
            request = RetrievalRequest.model_validate(request)
        start = perf_counter()
        vector, _ = self.embedder.queries([request.query])
        where = where_filter(request)
        args = {"query_embeddings": vector.tolist(), "n_results": request.top_k,
                "include": ["metadatas", "documents", "distances"]}
        if where:
            args["where"] = where
        found = self.collection.query(**args)
        hits = []
        for review_id, text, meta, distance in zip(
            found["ids"][0], found["documents"][0], found["metadatas"][0],
            found["distances"][0], strict=True,
        ):
            if (request.restaurant_id and meta.get("restaurant_id") != str(request.restaurant_id)) or (
                request.restaurant_key and meta.get("restaurant_key") != request.restaurant_key) or (
                request.sentiment and meta["predicted_sentiment"] != request.sentiment
            ) or (request.aspect and request.aspect not in json.loads(meta["predicted_aspects"])):
                raise ValueError("Chroma returned a document outside requested filters")
            hits.append(ReviewHit(review_id=review_id, restaurant_name=meta["restaurant_name"],
                                  text=text, score=1 - float(distance), metadata=meta))
        latency_ms = (perf_counter() - start) * 1000
        if self.monitor:
            filters = request.model_dump(exclude={"query", "top_k"}, exclude_none=True)
            filters = {k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in filters.items()}
            record("retrieval", self.metadata["model_name"], self.metadata["model_hash"], None,
                   latency_ms, "ok", top_k=request.top_k, filters=filters)
        return hits, latency_ms

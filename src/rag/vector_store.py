"""Persisted Chroma index; explicit source and embedding identity checks."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.foody.schema import digest, save_json

DOCUMENTS = Path("data/foody/rag/documents.jsonl")
STORE = Path("data/foody/vector_store")
MANIFEST = STORE / "index_manifest.json"


def read_documents(path=DOCUMENTS):
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            doc = json.loads(line)
            metadata = doc["metadata"]
            review_id = doc["id"]
            if review_id in seen or review_id != metadata["review_id"]:
                raise ValueError("Duplicate or inconsistent RAG review ID")
            if (not doc["text"].strip() or "\ufffd" in doc["text"]
                    or metadata["source"] not in {"foody", "foody_review_2025_2026"}):
                raise ValueError("Invalid document in first-party RAG source")
            seen.add(review_id)
            yield doc


def epoch(value):
    if not value:
        return None
    # Source timestamps are naive assumed Vietnam local time; numeric values are only
    # for consistent ordering/filtering, not claims of verified UTC instants.
    date = datetime.fromisoformat(value)
    return int(date.replace(tzinfo=UTC).timestamp())


def chroma_metadata(doc):
    source = doc["metadata"]
    metadata = {
        "review_id": doc["id"],
        "restaurant_name": source["restaurant_name"],
        "review_timestamp": (source.get("review_timestamp") or source.get("review_time") or ""),
        "raw_rating": source.get("raw_rating", source.get("rating")),
        "predicted_sentiment": source["predicted_sentiment"],
        "predicted_aspects": json.dumps(source["predicted_aspects"], ensure_ascii=False),
        "detected_issues": json.dumps(source["detected_issues"], ensure_ascii=False),
        "source": source["source"],
    }
    if source.get("restaurant_id") is not None:
        metadata["restaurant_id"] = str(source["restaurant_id"])
    if source.get("restaurant_key") is not None:
        metadata["restaurant_key"] = source["restaurant_key"]
    if metadata["review_timestamp"]:
        metadata["timestamp_epoch"] = epoch(metadata["review_timestamp"])
    for aspect in source["predicted_aspects"]:
        metadata[f"aspect_{aspect}"] = True
    return metadata


def client(path=STORE):
    import chromadb

    return chromadb.PersistentClient(path=str(path))


def collection_name(model_name, model_hash):
    import hashlib

    key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return f"foody_{key}_{model_hash[:12]}"


def open_collection(index_manifest=MANIFEST, store=STORE):
    metadata = json.loads(Path(index_manifest).read_text(encoding="utf-8"))
    collection = client(store).get_collection(metadata["collection"])
    if collection.count() != metadata["document_count"]:
        raise ValueError("Chroma count differs from verified index manifest")
    return collection, metadata


def build_index(embedder, model_hash, model_revision, model_size_mb,
                documents=DOCUMENTS, store=STORE, manifest=MANIFEST, batch_size=128):
    documents, store, manifest = Path(documents), Path(store), Path(manifest)
    source_sha = digest(documents)
    docs = list(read_documents(documents))
    name = collection_name(embedder.spec.name, model_hash)
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (saved["documents_sha256"] != source_sha or saved["model_hash"] != model_hash
                or saved["collection"] != name or saved["document_count"] != len(docs)):
            raise ValueError("Existing index manifest differs; use a new index directory/version")
        stored = client(store).get_collection(name)
        if stored.count() != len(docs):
            raise ValueError("Incomplete existing index")
        return saved
    store.mkdir(parents=True, exist_ok=True)
    collection = client(store).get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    if collection.count() > len(docs):
        raise ValueError("Existing partial index has excess records")
    elapsed = 0.0
    for offset in range(0, len(docs), batch_size):
        batch = docs[offset:offset + batch_size]
        vectors, seconds = embedder.documents([d["text"] for d in batch])
        elapsed += seconds
        collection.upsert(ids=[d["id"] for d in batch], embeddings=vectors.tolist(),
                          documents=[d["text"] for d in batch],
                          metadatas=[chroma_metadata(d) for d in batch])
    if collection.count() != len(docs):
        raise ValueError("Index count did not reconcile")
    result = {
        "collection": name, "document_count": len(docs),
        "documents_sha256": source_sha, "model_name": embedder.spec.name,
        "model_revision": model_revision, "onnx_source_repo": embedder.spec.repo,
        "model_hash": model_hash,
        "model_size_mb": model_size_mb, "device": embedder.device,
        "dimension": embedder.spec.dimension, "embedding_seconds": elapsed,
        "documents_per_second": len(docs) / elapsed, "mean_embedding_ms": elapsed * 1000 / len(docs),
        "backend": "ChromaDB persistent cosine HNSW", "timestamp": datetime.now(UTC).isoformat(),
    }
    save_json(manifest, result)
    return result

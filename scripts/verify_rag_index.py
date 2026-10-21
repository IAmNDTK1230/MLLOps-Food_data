"""Reconcile the full Foody source, selected ONNX file, and persistent Chroma IDs."""

import json
from pathlib import Path

from src.foody.schema import digest, save_json
from src.rag.vector_store import DOCUMENTS, MANIFEST, STORE, client, read_documents


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = json.loads(Path("reports/rag/embedding_selection.json").read_text(encoding="utf-8"))
    collection = client(STORE).get_collection(manifest["collection"])
    source_ids = {doc["id"] for doc in read_documents(DOCUMENTS)}
    stored_ids = set(collection.get(include=[])["ids"])
    sample = collection.get(limit=1, include=["embeddings", "metadatas"])
    result = {
        "source_document_count": len(source_ids),
        "source_sha256_matches": digest(DOCUMENTS) == manifest["documents_sha256"],
        "collection_count": collection.count(),
        "stored_unique_review_ids": len(stored_ids),
        "ids_exactly_match_source": source_ids == stored_ids,
        "vector_dimension": len(sample["embeddings"][0]),
        "distance_space": collection.metadata.get("hnsw:space"),
        "model_name_matches_selection": manifest["model_name"] == selected["model"],
        "model_revision_matches_selection": manifest["model_revision"] == selected["revision"],
        "onnx_sha256_matches": digest(selected["model_file"]) == selected["model_hash"],
        "collection_path": str(STORE.resolve()),
    }
    result["pass"] = (
        result["source_sha256_matches"] and result["ids_exactly_match_source"]
        and result["onnx_sha256_matches"] and result["model_name_matches_selection"]
        and result["model_revision_matches_selection"]
        and result["collection_count"] == manifest["document_count"] == len(source_ids)
        and result["vector_dimension"] == manifest["dimension"]
        and result["distance_space"] == "cosine"
    )
    save_json("reports/rag/index_integrity.json", result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("RAG index integrity failed")


if __name__ == "__main__":
    main()

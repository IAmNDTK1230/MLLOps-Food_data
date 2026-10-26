"""Build the accepted review corpus indexes beside the active indexes."""

import json
from pathlib import Path

from src.foody.review_migration import DOCUMENTS
from src.foody.schema import digest
from src.rag.embeddings import CANDIDATES, LocalEmbeddings
from src.rag.vector_store import build_index
from src.rag_v2.lexical import build_index as build_lexical

VECTOR_STORE = Path("data/foody/vector_store_review_2025_2026")
LEXICAL_STORE = Path("data/foody/lexical_index_review_2025_2026")


def main():
    selection = json.loads(Path("reports/rag/embedding_selection.json").read_text("utf-8"))
    model_path = Path(selection["model_file"].replace("\\", "/"))
    if digest(model_path) != selection["model_hash"]:
        raise ValueError("Selected embedding model artifact checksum changed")
    spec = next(item for item in CANDIDATES if item.name == selection["model"])
    if spec.dimension != 384:
        raise ValueError("Review migration requires the selected 384-dimensional model")
    lexical = build_lexical(DOCUMENTS, LEXICAL_STORE / "reviews.sqlite",
                            LEXICAL_STORE / "index_manifest.json")
    embedder = LocalEmbeddings(spec, local_files_only=True)
    vector = build_index(embedder, selection["model_hash"], selection["revision"],
                         selection["model_size_mb"], documents=DOCUMENTS,
                         store=VECTOR_STORE, manifest=VECTOR_STORE / "index_manifest.json")
    if lexical["document_count"] != vector["document_count"]:
        raise ValueError("Lexical and dense index counts differ")
    print(json.dumps({"vector": vector, "lexical": lexical}, ensure_ascii=True))


if __name__ == "__main__":
    main()

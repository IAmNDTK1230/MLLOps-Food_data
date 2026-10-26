"""Product RAG CLI; no model training or Foody cleaning."""

import json
from pathlib import Path

from src.foody.schema import digest, json_default
from src.rag.embeddings import CANDIDATES, LocalEmbeddings
from src.rag.evaluation import SELECTION, compare_candidates, evaluate_index
from src.rag.retriever import Retriever
from src.rag.schemas import RetrievalRequest
from src.rag.vector_store import build_index

COMMANDS = {"rag-build-index", "rag-evaluate", "rag-query", "restaurant-health-report"}


def portable_artifact_path(value):
    """Resolve a saved project-relative path on Windows or Linux containers."""
    return Path(value.replace("\\", "/"))


def add_commands(commands):
    commands.add_parser("rag-build-index", help="Compare embeddings, then persist Foody Chroma index")
    commands.add_parser("rag-evaluate", help="Evaluate known-positive retrieval on full index")
    query = commands.add_parser("rag-query", help="Grounded Qwen answer from retrieved Foody reviews")
    query.add_argument("--restaurant", help="Exact RestaurantName or derived restaurant_key")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--sentiment", choices=["positive", "neutral", "negative"])
    query.add_argument("--aspect")
    query.add_argument("--date-from")
    query.add_argument("--date-to")
    health = commands.add_parser("restaurant-health-report", help="Grounded restaurant report")
    health.add_argument("--restaurant", required=True)


def selected_retriever():
    from src.foody.review_runtime import VECTOR, new_corpus_active
    from src.rag.vector_store import open_collection

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    model_path = portable_artifact_path(selection["model_file"])
    if digest(model_path) != selection["model_hash"]:
        raise ValueError("Embedding artifact checksum differs from selected model")
    spec = next(item for item in CANDIDATES if item.name == selection["model"])
    embedder = LocalEmbeddings(spec, local_files_only=True)
    if new_corpus_active():
        collection, metadata = open_collection(VECTOR / "index_manifest.json", VECTOR)
        return Retriever(embedder, collection, metadata)
    return Retriever(embedder)


def execute(args):
    if args.command == "rag-build-index":
        selection = compare_candidates()
        spec = next(item for item in CANDIDATES if item.name == selection["model"])
        embedder = LocalEmbeddings(spec, local_files_only=True)
        result = build_index(embedder, selection["model_hash"], selection["revision"],
                             selection["model_size_mb"])
    elif args.command == "rag-evaluate":
        result = evaluate_index()
    elif args.command == "rag-query":
        from src.llm.gateway import LLMGateway
        from src.rag.service import answer_question, resolve_restaurant

        key = resolve_restaurant(args.restaurant)[0] if args.restaurant else None
        request = RetrievalRequest(query=args.question, restaurant_key=key, top_k=args.top_k,
                                   sentiment=args.sentiment, aspect=args.aspect,
                                   date_from=args.date_from, date_to=args.date_to)
        with LLMGateway() as gateway:
            answer, hits, telemetry = answer_question(selected_retriever(), gateway, request)
        result = {"response": answer.model_dump(), "retrieved_reviews": [
            hit.model_dump() for hit in hits], "telemetry": telemetry}
    else:
        from src.llm.gateway import LLMGateway
        from src.rag.service import health_report

        with LLMGateway() as gateway:
            report, hits, telemetry = health_report(selected_retriever(), gateway, args.restaurant)
        result = {"report": report.model_dump(), "retrieved_review_ids": [
            hit.review_id for hit in hits], "telemetry": telemetry}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0

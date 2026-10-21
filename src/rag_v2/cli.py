"""Separate RAG V2 commands; V1 CLI and evaluation remain unchanged."""

import json

from src.foody.schema import json_default
from src.rag.cli import selected_retriever
from src.rag.schemas import RetrievalRequest
from src.rag.service import resolve_restaurant
from src.rag_v2.hybrid import HYBRID_EVIDENCE, HybridRetriever
from src.rag_v2.lexical import BM25Retriever, build_index

COMMANDS = {"rag-v2-build-lexical", "rag-v2-evaluate", "rag-v2-query",
            "rag-v2-health-report"}


def add_commands(commands):
    commands.add_parser("rag-v2-build-lexical", help="Persist SQLite BM25 index")
    commands.add_parser("rag-v2-evaluate", help="Compare V1 dense and V2 retrieval candidates")
    query = commands.add_parser("rag-v2-query", help="Evidence-aware hybrid RAG answer")
    query.add_argument("--restaurant")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--sentiment", choices=["positive", "neutral", "negative"])
    query.add_argument("--aspect")
    query.add_argument("--date-from")
    query.add_argument("--date-to")
    health = commands.add_parser("rag-v2-health-report", help="Hybrid-retrieval health report")
    health.add_argument("--restaurant", required=True)


def selected_v2_retriever():
    build_index()
    dense = selected_retriever()
    dense.monitor = False
    return HybridRetriever(dense, BM25Retriever(), HYBRID_EVIDENCE)


def execute(args):
    if args.command == "rag-v2-build-lexical":
        result = build_index()
    elif args.command == "rag-v2-evaluate":
        from scripts.evaluate_rag_v2 import main

        main()
        return 0
    elif args.command == "rag-v2-query":
        from src.llm.gateway import LLMGateway
        from src.rag_v2.service import answer_question_v2

        key = resolve_restaurant(args.restaurant)[0] if args.restaurant else None
        request = RetrievalRequest(query=args.question, restaurant_key=key, top_k=args.top_k,
                                   sentiment=args.sentiment, aspect=args.aspect,
                                   date_from=args.date_from, date_to=args.date_to)
        with LLMGateway() as gateway:
            answer, hits, telemetry, debug = answer_question_v2(
                selected_v2_retriever(), gateway, request,
            )
        result = {"response": answer.model_dump(),
                  "retrieved_reviews": [hit.model_dump() for hit in hits],
                  "telemetry": telemetry, "retrieval_debug": debug}
    else:
        from src.llm.gateway import LLMGateway
        from src.rag_v2.health import health_report_v2

        with LLMGateway() as gateway:
            report, hits, telemetry, verification = health_report_v2(
                selected_v2_retriever(), gateway, args.restaurant,
            )
        result = {"report": report.model_dump(),
                  "retrieved_reviews": [hit.model_dump() for hit in hits],
                  "telemetry": telemetry, "verification": verification}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0

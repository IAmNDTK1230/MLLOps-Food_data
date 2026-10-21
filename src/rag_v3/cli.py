"""Explicit V3 evaluation, query, and health-report commands."""

import json
from dataclasses import asdict

from src.demo import demo_config
from src.foody.schema import json_default
from src.llm.gateway import LLMGateway
from src.rag.schemas import RetrievalRequest
from src.rag.service import resolve_restaurant
from src.rag_v3.retriever import RoutedRetriever

COMMANDS = {"rag-v3-evaluate-retrieval", "rag-v3-evaluate-generation",
            "rag-v3-query", "rag-v3-health-report"}


def selected_v3_gateway():
    config = demo_config()
    if config["rag_version"] != "RAG_V3":
        raise ValueError("Active demo configuration must use RAG_V3")
    return LLMGateway(config_path=config["qwen_config"],
                      provenance_path=config["qwen_provenance"])


def add_commands(commands):
    commands.add_parser("rag-v3-evaluate-retrieval", help="Compare frozen local policies")
    commands.add_parser("rag-v3-evaluate-generation", help="Compare 1.5B and 3B packets")
    query = commands.add_parser("rag-v3-query", help="Evidence-first local RAG answer")
    query.add_argument("--restaurant", required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    health = commands.add_parser("rag-v3-health-report", help="Evidence-first health report")
    health.add_argument("--restaurant", required=True)


def execute(args):
    if args.command == "rag-v3-evaluate-retrieval":
        from scripts.evaluate_rag_v3_retrieval import main

        main()
        return 0
    if args.command == "rag-v3-evaluate-generation":
        from scripts.evaluate_rag_v3_generation import main

        main()
        return 0
    retriever = RoutedRetriever()
    with selected_v3_gateway() as gateway:
        if args.command == "rag-v3-query":
            from src.rag_v3.service import answer_question_v3

            key, _ = resolve_restaurant(args.restaurant)
            result, hits, telemetry, debug, packet = answer_question_v3(
                retriever, gateway, RetrievalRequest(
                    query=args.question, restaurant_key=key, top_k=args.top_k,
                ),
            )
            output = {"response": result.model_dump(), "telemetry": telemetry,
                      "retrieved_reviews": [hit.model_dump() for hit in hits],
                      "retrieval_debug": debug, "evidence_packet": asdict(packet)}
        else:
            from src.rag_v3.health import health_report_v3

            report, telemetry, verification = health_report_v3(
                retriever, gateway, args.restaurant,
            )
            output = {"report": report.model_dump(), "telemetry": telemetry,
                      "verification": verification}
    print(json.dumps(output, ensure_ascii=False, indent=2, default=json_default))
    return 0

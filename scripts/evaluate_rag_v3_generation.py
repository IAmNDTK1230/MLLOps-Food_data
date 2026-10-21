"""Replay frozen evidence packets through both local Qwen generators."""

import json
from pathlib import Path

from src.foody.schema import save_json
from src.llm.gateway import LLMGateway
from src.rag.schemas import RetrievalRequest, ReviewHit
from src.rag_v3.evidence import Packet
from src.rag_v3.routing import Route
from src.rag_v3.service import answer_question_v3

MODELS = {
    "qwen_1_5b": {"config_path": "qwen_v1.yaml",
                  "provenance_path": "models/qwen_v1/provenance.json"},
    "qwen_3b": {"config_path": "qwen_v3_3b.yaml",
                "provenance_path": "models/qwen_v3/provenance.json"},
}


class ReplayRetriever:
    metadata = {"model_name": "frozen-evidence-replay"}

    def search_debug(self, _request):
        raise AssertionError("Generation comparison must not perform retrieval")


def load_frozen_packets(path):
    """Hydrate exact saved generation inputs; no retrieval or packet rebuild."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    restaurant_key = next(case["retrieval_debug"]["effective_filters"]["restaurant_key"]
                          for case in artifact.values()
                          if "restaurant_key" in case["retrieval_debug"]["effective_filters"])
    prepared = {}
    for name, case in artifact.items():
        route = Route(**case["route"])
        saved = case["packet"]
        packet = Packet(saved["question_intent"], saved["aspect"],
                        saved["restaurant"], tuple(saved["evidence"]),
                        saved["aggregates"], saved["support_gate_status"])
        hits = [ReviewHit(
            review_id=item["review_id"], restaurant_name=packet.restaurant,
            text=item["text"], score=0.0,
            metadata={"predicted_sentiment": item["sentiment"],
                      "predicted_aspects": json.dumps(item["aspects"]),
                      "detected_issues": json.dumps(item["issues"]),
                      "raw_rating": item["rating"]},
        ) for item in packet.evidence]
        debug = case["retrieval_debug"]
        filters = debug["effective_filters"]
        request = RetrievalRequest(query=case["question"],
                                   restaurant_key=filters.get("restaurant_key", restaurant_key),
                                   top_k=debug["final_top_k"],
                                   date_from=filters.get("date_from"),
                                   date_to=filters.get("date_to"))
        prepared[name] = request, (route, hits, case["retrieval_latency_ms"],
                                   debug, packet)
    return prepared


def main():
    root = Path("reports/rag_v3")
    prepared = load_frozen_packets(root / "evidence_packets.json")
    retriever = ReplayRetriever()
    results = {}
    for model_key, paths in MODELS.items():
        model_results = {}
        with LLMGateway(**paths) as gateway:
            for name, (request, state) in prepared.items():
                audit = []
                response, _, telemetry, debug, _ = answer_question_v3(
                    retriever, gateway, request, prepared=state,
                    generation_audit=audit,
                )
                model_results[name] = {"question": request.query,
                                       "response": response.model_dump(),
                                       "telemetry": telemetry,
                                       "route": debug["query_route"],
                                       "generation_audit": audit}
                print(model_key, name, telemetry["status"],
                      telemetry["generation_mode"], flush=True)
            runtime = gateway.runtime
            resources = {"peak_server_rss_bytes": runtime.peak_rss if runtime else None,
                         "gpu_samples": len(runtime.gpu_samples) if runtime else 0,
                         "peak_gpu_memory_mib": max((row[1] for row in runtime.gpu_samples),
                                                    default=None) if runtime else None}
        results[model_key] = {"model_name": gateway.name,
                              "model_sha256": gateway.version,
                              "cases": model_results, "resources": resources,
                              "evidence_packet_file": str(root / "evidence_packets.json")}
        save_json(root / "generator_comparison.json", results)
    print("Saved RAG V3 model comparison")


if __name__ == "__main__":
    main()

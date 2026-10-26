"""Optional backend for Docker/API users with a shared review retriever."""

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.monitoring.metrics import API_DURATION, API_REQUESTS, observe_rag, observe_rag_error

api = FastAPI(title="Foody grounded review API")


class Question(BaseModel):
    question: str = Field(min_length=1)
    restaurant: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


@api.middleware("http")
async def count_requests(request, call_next):
    started = perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    label = route.path if route is not None else "unmatched"
    API_REQUESTS.labels(route=label, status=str(response.status_code)).inc()
    API_DURATION.labels(route=label).observe(perf_counter() - started)
    return response


@api.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@api.get("/version")
def version():
    import yaml

    platform = yaml.safe_load(Path("config/platform_versions.yaml").read_text(encoding="utf-8"))
    demo = yaml.safe_load(Path("config/demo.yaml").read_text(encoding="utf-8"))
    versions_path = Path("reports/foody_intelligence/versions.json")
    versions = json.loads(versions_path.read_text(encoding="utf-8")) if versions_path.exists() else {}
    embedding_path = Path("reports/rag/embedding_selection.json")
    embedding = json.loads(embedding_path.read_text(encoding="utf-8")) if embedding_path.exists() else {}
    qwen_path = Path(demo["qwen_provenance"])
    qwen = json.loads(qwen_path.read_text(encoding="utf-8")) if qwen_path.exists() else {}
    from src.foody.review_runtime import active_manifest, new_corpus_active

    review = active_manifest()
    sha = os.getenv("GIT_SHA")
    if not sha and Path(".git").exists():
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip() or None
    return {"project_version": platform["project_version"], "git_sha": sha,
            "warehouse_schema_version": platform["warehouse_schema_version"],
            "event_schema_version": platform["event_schema_version"],
            "rag_version": demo["rag_version"], "prompt_id": demo["prompt_version"],
            "sentiment_sha256": versions.get("sentiment_model_sha256"),
            "aspect_sha256": versions.get("aspect_model_sha256"),
            "dataset_sha256": (review.get("source_sha256") if new_corpus_active() else
                               versions.get("source_sha256")),
            "review_data_version": review.get("data_version", "legacy"),
            "review_count": review.get("row_count", review.get("document_count")),
            "embedding_model": demo["embedding_model"],
            "embedding_revision": embedding.get("revision"),
            "embedding_sha256": embedding.get("model_hash"),
            "qwen_model": demo["qwen_model"], "qwen_revision": qwen.get("revision"),
            "qwen_sha256": qwen.get("model_sha256")}


@api.post("/ask-v3")
def ask_v3(body: Question):
    """Monitor the accepted V3 service without changing its retrieval/answer logic."""
    try:
        from src.foody.review_runtime import new_corpus_active, review_identity
        from src.rag.schemas import RetrievalRequest
        from src.rag.service import resolve_restaurant
        from src.rag_v3.cli import selected_v3_gateway
        from src.rag_v3.retriever import cached_retriever
        from src.rag_v3.service import answer_question_v3

        key = (review_identity(body.restaurant)[0] if new_corpus_active() else
               resolve_restaurant(body.restaurant)[0]) if body.restaurant else None
        with selected_v3_gateway() as gateway:
            answer, hits, telemetry, _, _ = answer_question_v3(
                cached_retriever(), gateway,
                RetrievalRequest(query=body.question, top_k=body.top_k,
                                 **({"restaurant_id": key} if new_corpus_active() else
                                    {"restaurant_key": key})),
            )
        observe_rag(telemetry)
        return {"response": answer.model_dump(),
                "retrieved": [hit.model_dump() for hit in hits], "telemetry": telemetry}
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        observe_rag_error()
        raise HTTPException(status_code=503, detail=str(error)) from error


@api.get("/health")
def health():
    from src.foody.review_runtime import VECTOR, new_corpus_active

    vector = VECTOR if new_corpus_active() else Path("data/foody/vector_store")
    return {"status": "ready", "vector_index_available": Path(
        vector / "index_manifest.json").exists(),
        "qwen_weights_available": Path("models/qwen_v1/qwen2.5-1.5b-instruct-q4_k_m.gguf").exists()}


@api.post("/ask")
def ask(body: Question):
    try:
        from src.foody.review_runtime import new_corpus_active, review_identity
        from src.llm.gateway import LLMGateway
        from src.rag.cli import selected_retriever
        from src.rag.schemas import RetrievalRequest
        from src.rag.service import answer_question, resolve_restaurant

        key = (review_identity(body.restaurant)[0] if new_corpus_active() else
               resolve_restaurant(body.restaurant)[0]) if body.restaurant else None
        with LLMGateway() as gateway:
            answer, hits, telemetry = answer_question(
                selected_retriever(), gateway,
                RetrievalRequest(query=body.question, top_k=body.top_k,
                                 **({"restaurant_id": key} if new_corpus_active() else
                                    {"restaurant_key": key})),
            )
        return {"response": answer.model_dump(), "retrieved": [hit.model_dump() for hit in hits],
                "telemetry": telemetry}
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@api.get("/restaurants/{restaurant}/health-report")
def report(restaurant: str):
    try:
        from src.llm.gateway import LLMGateway
        from src.rag.cli import selected_retriever
        from src.rag.service import health_report

        with LLMGateway() as gateway:
            result, _, telemetry = health_report(selected_retriever(), gateway, restaurant)
        return {"report": result.model_dump(), "telemetry": telemetry}
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

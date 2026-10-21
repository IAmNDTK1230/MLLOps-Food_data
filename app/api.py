"""Optional backend for Docker/API users; all heavyweight services load per request."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

api = FastAPI(title="Foody grounded review API")


class Question(BaseModel):
    question: str = Field(min_length=1)
    restaurant: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


@api.get("/health")
def health():
    return {"status": "ready", "vector_index_available": Path(
        "data/foody/vector_store/index_manifest.json").exists(),
        "qwen_weights_available": Path("models/qwen_v1/qwen2.5-1.5b-instruct-q4_k_m.gguf").exists()}


@api.post("/ask")
def ask(body: Question):
    try:
        from src.llm.gateway import LLMGateway
        from src.rag.cli import selected_retriever
        from src.rag.schemas import RetrievalRequest
        from src.rag.service import answer_question, resolve_restaurant

        key = resolve_restaurant(body.restaurant)[0] if body.restaurant else None
        with LLMGateway() as gateway:
            answer, hits, telemetry = answer_question(
                selected_retriever(), gateway,
                RetrievalRequest(query=body.question, restaurant_key=key, top_k=body.top_k),
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

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.model_loader import load_model
from src.api.monitoring import InferenceMonitor
from src.api.schemas import PredictRequest, PredictResponse
from src.ml.config import LABELS


def create_app(loader=load_model, monitor=None) -> FastAPI:
    monitor = monitor or InferenceMonitor(
        Path(os.getenv("INFERENCE_LOG_PATH", "logs/inference.jsonl"))
    )

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.model = loader()
        except Exception as error:
            app.state.model = None
            app.state.load_error = type(error).__name__
            logging.getLogger("shopee.api").warning("Model unavailable (%s)", type(error).__name__)
        yield

    app = FastAPI(title="Shopee review sentiment", lifespan=lifespan)

    @app.get("/health")
    def health():
        model = app.state.model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model unavailable; import real reviews, train, and configure serving",
            )
        return {"status": "ready", "model_version": model.version}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        try:
            probabilities = model.pipeline.predict_proba([request.text])[0]
            values = dict(zip(model.pipeline.classes_, map(float, probabilities), strict=True))
            if not values or not set(values).issubset(LABELS):
                raise ValueError("Unsupported model classes")
            if (
                any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values())
                or abs(sum(values.values()) - 1) > 1e-6
            ):
                raise ValueError("Invalid probabilities")
            predicted = max(values, key=values.get)
            monitor.record(model.version, predicted, values[predicted], len(request.text))
            return PredictResponse(
                sentiment=predicted,
                probabilities={label: values.get(label, 0.0) for label in LABELS},
                model_version=model.version,
            )
        except Exception as error:
            logging.getLogger("shopee.api").error("Inference failed (%s)", type(error).__name__)
            raise HTTPException(status_code=503, detail="Inference unavailable") from None

    @app.get("/monitoring")
    def monitoring():
        return monitor.snapshot()

    return app


app = create_app()

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.timestamps import utc_now


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": utc_now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            **getattr(record, "context", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shopee")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = str((directory / "pipeline.jsonl").resolve())
    if any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == target
        for handler in logger.handlers
    ):
        return
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(target, encoding="utf-8")):
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)


def event(stage: str, **context: Any) -> None:
    logging.getLogger("shopee").info(stage, extra={"context": {"stage": stage, **context}})

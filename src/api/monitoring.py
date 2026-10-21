import json
import threading
from collections import Counter
from pathlib import Path

from src.utils.timestamps import utc_now


class InferenceMonitor:
    def __init__(self, path: Path, confidence_threshold: float = 0.6):
        self.path = path
        self.threshold = confidence_threshold
        self.counts = Counter()
        self.low_confidence = 0
        self.length_total = 0
        self.lock = threading.Lock()

    def record(self, model_version: str, sentiment: str, confidence: float, text_length: int):
        record = {
            "timestamp": utc_now().isoformat(),
            "model_version": model_version,
            "predicted_class": sentiment,
            "confidence": confidence,
            "text_length": text_length,
        }
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.counts[sentiment] += 1
            self.low_confidence += confidence < self.threshold
            self.length_total += text_length

    def snapshot(self) -> dict:
        with self.lock:
            total = sum(self.counts.values())
            return {
                "requests": total,
                "predicted_class_counts": dict(self.counts),
                "low_confidence_rate": self.low_confidence / total if total else None,
                "average_text_length": self.length_total / total if total else None,
                "drift_status": "NOT COMPUTED — window/reference comparison hook only",
            }

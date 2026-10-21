import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, default=str, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class UploadManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(path) + ".lock", timeout=120)

    def read(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def contains(self, remote_path: str, checksum: str) -> bool:
        return any(
            row["remote_path"] == remote_path
            and row["checksum"] == checksum
            and row["status"] == "UPLOADED"
            for row in self.read()
        )

    def add(self, record: dict[str, Any]) -> None:
        with self.lock:
            rows = self.read()
            rows.append(record)
            atomic_json(self.path, rows)

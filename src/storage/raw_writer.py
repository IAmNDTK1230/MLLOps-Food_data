import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.models.crawl_run import CrawlRun
from src.utils.timestamps import utc_now


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidental replacement of historical data.
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def write_raw(root: Path, kind: str, records: list[dict[str, Any]], run: CrawlRun) -> Path | None:
    if not records:
        return None
    stamp = utc_now()
    path = (
        root
        / "raw"
        / kind
        / f"crawl_date={stamp:%Y-%m-%d}"
        / f"crawl_hour={stamp:%H}"
        / f"{kind}_{stamp:%Y%m%d_%H%M%S}_{run.run_id}_{uuid4().hex[:8]}.jsonl"
    )
    return write_jsonl(path, [{**record, "_run_id": run.run_id} for record in records])

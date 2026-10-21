from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from src.models.product import Product
from src.models.review import Review
from src.processing.normalizer import normalize
from src.storage.raw_writer import write_jsonl
from src.utils.timestamps import utc_now


def validate_records(
    records: list[dict[str, Any]], kind: str, run_id: str, data_dir: Path
) -> tuple[list[dict[str, Any]], int]:
    model = Product if kind == "products" else Review
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in records:
        try:
            item = model.model_validate(normalize(raw)).model_dump(mode="json")
            key = (
                (item["product_id"], item["crawl_timestamp"])
                if kind == "products"
                else (
                    item.get("review_id")
                    or (
                        item["product_id"],
                        item["rating"],
                        item.get("comment"),
                        item.get("review_timestamp"),
                    ),
                )
            )
            if key not in seen:
                seen.add(key)
                valid.append({**item, "_run_id": run_id})
        except (ValidationError, ValueError, TypeError) as error:
            rejected.append(
                {
                    "record": raw,
                    "validation_error": str(error),
                    "run_id": run_id,
                    "timestamp": utc_now().isoformat(),
                }
            )
    if rejected:
        write_jsonl(data_dir / "rejected" / f"{kind}_{run_id}_{uuid4().hex}.jsonl", rejected)
    return valid, len(rejected)

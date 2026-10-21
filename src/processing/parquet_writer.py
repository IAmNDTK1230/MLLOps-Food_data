from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.models.product import Product
from src.models.review import Review
from src.utils.timestamps import utc_now


def arrow_schema(kind: str) -> pa.Schema:
    fields = (Product if kind == "products" else Review).model_fields
    floating = {"price", "original_price", "discount_percent", "rating"}
    integer = {"rating_count", "sold_count", "stock", "likes"}
    timestamps = {"crawl_timestamp", "review_timestamp"}
    return pa.schema(
        [
            pa.field(
                name,
                pa.float64()
                if name in floating
                else pa.int64()
                if name in integer
                else pa.timestamp("us", tz="UTC")
                if name in timestamps
                else pa.string(),
            )
            for name in fields
        ]
        + [pa.field("_run_id", pa.string())]
    )


def write_parquet(root: Path, kind: str, records: list[dict[str, Any]], run_id: str) -> Path | None:
    if not records:
        return None
    stamp = utc_now()
    path = (
        root
        / "staged"
        / kind
        / f"crawl_date={stamp:%Y-%m-%d}"
        / f"{kind}_{stamp:%Y%m%d_%H%M%S}_{run_id}_{uuid4().hex[:8]}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    for column in ("crawl_timestamp", "review_timestamp"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    table = pa.Table.from_pandas(frame, schema=arrow_schema(kind), preserve_index=False)
    with path.open("xb") as stream:
        pq.write_table(table, stream, compression="snappy")
    return path

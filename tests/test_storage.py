import json

import pyarrow.parquet as pq
import pytest

from src.models.crawl_run import CrawlRun
from src.processing.parquet_writer import write_parquet
from src.processing.validator import validate_records
from src.storage.raw_writer import write_jsonl, write_raw


def test_history_and_parquet_schema(tmp_path, product):
    run = CrawlRun(keyword="tai nghe", number_of_pages=1)
    raw = {**product, "_raw": {"html": "<span>original</span>"}}
    a = write_raw(tmp_path, "products", [raw], run)
    b = write_raw(tmp_path, "products", [raw], run)
    assert a != b and a.exists() and b.exists()
    assert json.loads(a.read_text(encoding="utf-8"))["_raw"] == raw["_raw"]
    valid, _ = validate_records([product], "products", run.run_id, tmp_path)
    parquet = write_parquet(tmp_path, "products", valid, run.run_id)
    table = pq.ParquetFile(parquet).read()
    assert str(table.schema.field("crawl_timestamp").type) == "timestamp[us, tz=UTC]"
    assert str(table.schema.field("stock").type) == "int64"
    assert table.to_pylist()[0]["price"] == 199000
    with pytest.raises(FileExistsError):
        write_jsonl(a, [raw])

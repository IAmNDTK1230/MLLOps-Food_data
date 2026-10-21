import json

from src.processing.validator import validate_records


def test_rejections_preserve_record_and_error(product, tmp_path):
    invalid = {**product, "price": "contact shop"}
    valid, rejected = validate_records([product, product, invalid], "products", "run1", tmp_path)
    assert len(valid) == 1
    assert valid[0]["price"] == 199000
    assert rejected == 1
    record = json.loads(next((tmp_path / "rejected").glob("*.jsonl")).read_text(encoding="utf-8"))
    assert record["record"] == invalid
    assert record["run_id"] == "run1"
    assert record["validation_error"]

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.config import Settings
from src.orchestration import flow as module


def unwrap_tasks(monkeypatch):
    # Execute the real task bodies without an orchestration server in unit tests.
    for name in (
        "crawl_products",
        "crawl_reviews",
        "write_raw_files",
        "validate_data",
        "convert_to_parquet",
        "upload_to_databricks",
        "trigger_databricks_job",
        "write_pipeline_summary",
    ):
        monkeypatch.setattr(module, name, getattr(module, name).fn)


def test_offline_flow_preserves_bad_rows(tmp_path, monkeypatch, product):
    unwrap_tasks(monkeypatch)
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(product) + "\n" + json.dumps({**product, "price": -1}) + "\n", encoding="utf-8"
    )
    settings = Settings(
        data_dir=tmp_path / "data", state_dir=tmp_path / "state", log_dir=tmp_path / "logs"
    )
    summary = module.shopee_data_pipeline.fn(
        "audio", 1, crawl_only=True, settings=settings, fixture=str(fixture)
    )
    assert summary["status"] == "SUCCESS"
    assert summary["products_collected"] == 1 and summary["rejected_records"] == 1
    assert len(summary["files"]) == 2
    assert list((settings.data_dir / "metadata").glob("*.jsonl"))


def test_blocked_keeps_partial_raw_and_skips_upload(tmp_path, monkeypatch, product):
    from src.crawler.base import CrawlResult

    unwrap_tasks(monkeypatch)
    monkeypatch.setattr(
        module, "crawl_products", lambda *args: CrawlResult([product], "CAPTCHA", ["stopped"])
    )
    monkeypatch.setattr(module, "crawl_reviews", lambda *args: CrawlResult())
    monkeypatch.setattr(
        module,
        "upload_to_databricks",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not upload")),
    )
    settings = Settings(
        data_dir=tmp_path / "data", state_dir=tmp_path / "state", log_dir=tmp_path / "logs"
    )
    with pytest.raises(module.PipelineFailed) as failure:
        module.shopee_data_pipeline.fn("audio", 1, settings=settings)
    summary = failure.value.summary
    assert summary["status"] == "CAPTCHA" and len(summary["files"]) == 2


def test_stage_failure_writes_summary(tmp_path, monkeypatch, product):
    unwrap_tasks(monkeypatch)
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps(product), encoding="utf-8")
    monkeypatch.setattr(
        module, "convert_to_parquet", lambda *args: (_ for _ in ()).throw(RuntimeError("broken"))
    )
    settings = Settings(
        data_dir=tmp_path / "data", state_dir=tmp_path / "state", log_dir=tmp_path / "logs"
    )
    with pytest.raises(module.PipelineFailed) as failure:
        module.shopee_data_pipeline.fn(
            "audio", 1, crawl_only=True, settings=settings, fixture=str(fixture)
        )
    summary = failure.value.summary
    assert summary["status"] == "FAILED" and summary["errors"]
    assert summary["stages"]["convert_to_parquet"]["status"] == "FAILED"
    assert len(summary["files"]) == 1


@pytest.mark.parametrize("catalog,allow", [("shopee", True), ("shopee_test", False)])
def test_direct_flow_cannot_bypass_fixture_safety(tmp_path, catalog, allow):
    settings = Settings(
        databricks_catalog=catalog,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
    )
    with pytest.raises(ValueError, match="shopee_test|local-only"):
        module.shopee_data_pipeline.fn(
            "cloud", 1, settings=settings, fixture="missing.jsonl", allow_synthetic_cloud_test=allow
        )
    assert not settings.data_dir.exists() and not settings.state_dir.exists()


def test_test_catalog_fixture_executes_upload_and_job_with_isolated_outputs(
    tmp_path, monkeypatch, product
):
    from src.databricks.uploader import Uploader

    unwrap_tasks(monkeypatch)
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps(product), encoding="utf-8")
    settings = Settings(
        databricks_catalog="shopee_test",
        databricks_job_id=99,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
    )
    destinations = []

    def upload(paths, configured, run):
        assert configured.databricks_catalog == "shopee_test"
        uploader = Uploader(Mock(), configured)
        destinations.extend(uploader.remote_path(path) for path in paths)
        return [{"status": "UPLOADED", "remote_path": destination} for destination in destinations]

    monkeypatch.setattr(module, "upload_to_databricks", upload)
    job = Mock(return_value={"status": "SUCCESS", "databricks_run_id": 123})
    monkeypatch.setattr(module, "trigger_databricks_job", job)
    summary = module.shopee_data_pipeline.fn(
        "cloud acceptance",
        1,
        settings=settings,
        fixture=str(fixture),
        allow_synthetic_cloud_test=True,
    )
    assert summary["status"] == "SUCCESS" and summary["products_collected"] == 1
    assert summary["stages"]["crawl_products"]["status"] == "FIXTURE"
    assert len(destinations) == 2
    assert all(
        path.startswith("/Volumes/shopee_test/raw/landing/shopee/products/")
        for path in destinations
    )
    assert all(
        Path(path).is_relative_to(settings.data_dir / "synthetic-cloud-test")
        for path in summary["files"]
    )
    assert not (settings.data_dir / "raw").exists()
    assert job.call_args.args[0].databricks_job_id == 99
    assert summary["stages"]["trigger_databricks_job"]["databricks_run_id"] == 123

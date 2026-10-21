import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from prefect import task

from src.config import Settings
from src.crawler.base import CrawlResult
from src.crawler.shopee_reviews import ShopeeReviewCrawler
from src.databricks.client import check_connection, workspace_client
from src.databricks.jobs import trigger_job
from src.databricks.uploader import Uploader
from src.models.crawl_run import CrawlRun
from src.processing.parquet_writer import write_parquet
from src.processing.validator import validate_records
from src.storage.raw_writer import write_jsonl, write_raw


@task(name="crawl_products", persist_result=False)
def crawl_products(settings: Settings, run: CrawlRun, resume: bool) -> CrawlResult:
    from src.crawler.shopee_products import ShopeeProductCrawler

    return asyncio.run(
        ShopeeProductCrawler(settings, run.run_id).crawl(
            run.keyword, run.number_of_pages, resume=resume
        )
    )


@task(name="crawl_reviews", persist_result=False)
def crawl_reviews(settings: Settings, run: CrawlRun, skip: bool) -> CrawlResult:
    if skip or not settings.reviews_enabled:
        return CrawlResult(status="DISABLED")
    return asyncio.run(ShopeeReviewCrawler().crawl(run.keyword, run.number_of_pages))


@task(name="validate_data", persist_result=False)
def validate_data(
    records: dict[str, list[dict[str, Any]]], settings: Settings, run: CrawlRun
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    normalized = {}
    failures = 0
    for kind, rows in records.items():
        normalized[kind], rejected = validate_records(rows, kind, run.run_id, settings.data_dir)
        failures += rejected
    return normalized, failures


@task(name="write_raw_files", persist_result=False)
def write_raw_files(
    records: dict[str, list[dict[str, Any]]], settings: Settings, run: CrawlRun
) -> list[Path]:
    return [
        path
        for kind, rows in records.items()
        if (path := write_raw(settings.data_dir, kind, rows, run))
    ]


@task(name="convert_to_parquet", persist_result=False)
def convert_to_parquet(
    records: dict[str, list[dict[str, Any]]], settings: Settings, run: CrawlRun
) -> list[Path]:
    return [
        path
        for kind, rows in records.items()
        if (path := write_parquet(settings.data_dir, kind, rows, run.run_id))
    ]


@task(name="upload_to_databricks", persist_result=False)
def upload_to_databricks(
    paths: list[Path], settings: Settings, run: CrawlRun
) -> list[dict[str, Any]]:
    client = workspace_client(settings)
    check_connection(client)
    uploader = Uploader(client, settings)
    return [uploader.upload(path, run.run_id) for path in paths]


@task(name="trigger_databricks_job", persist_result=False)
def trigger_databricks_job(settings: Settings, run: CrawlRun) -> dict[str, Any]:
    if not settings.databricks_job_id:
        return {"status": "SKIPPED", "reason": "No job configured; only landing files updated"}
    return trigger_job(workspace_client(settings), settings, run.run_id)


@task(name="write_pipeline_summary", persist_result=False)
def write_pipeline_summary(settings: Settings, run: CrawlRun) -> str:
    path = settings.data_dir / "metadata" / f"run_{run.run_id}_{run.status}_{uuid4().hex[:8]}.jsonl"
    run.metadata_file = str(path)
    write_jsonl(path, [run.model_dump(mode="json")])
    return str(path)

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from filelock import FileLock
from prefect import flow

from src.config import Settings
from src.crawler.base import CrawlResult
from src.models.crawl_run import CrawlRun
from src.orchestration.fixture_policy import validate_fixture_mode
from src.orchestration.tasks import (
    convert_to_parquet,
    crawl_products,
    crawl_reviews,
    trigger_databricks_job,
    upload_to_databricks,
    validate_data,
    write_pipeline_summary,
    write_raw_files,
)
from src.storage.manifest import atomic_json
from src.utils.logging import configure_logging, event
from src.utils.timestamps import utc_now

T = TypeVar("T")


class PipelineFailed(RuntimeError):
    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        super().__init__(
            f"Pipeline {summary['run_id']} ended with {summary['status']}; see run metadata"
        )


@flow(name="ShopeeDataPipeline", persist_result=False, log_prints=False)
def shopee_data_pipeline(
    keyword: str,
    pages: int = 2,
    skip_reviews: bool = False,
    crawl_only: bool = False,
    upload_only: bool = False,
    skip_databricks_job: bool = False,
    resume: bool = False,
    settings: Settings | None = None,
    fixture: str | None = None,
    allow_synthetic_cloud_test: bool = False,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    validate_fixture_mode(
        settings,
        fixture,
        crawl_only=crawl_only,
        upload_only=upload_only,
        allow_synthetic_cloud_test=allow_synthetic_cloud_test,
    )
    if fixture and not crawl_only:
        # Keep synthetic acceptance output outside normal raw/staged upload scans.
        settings = settings.model_copy(
            update={
                "data_dir": settings.data_dir / "synthetic-cloud-test",
                "state_dir": settings.state_dir / "synthetic-cloud-test",
                "log_dir": settings.log_dir / "synthetic-cloud-test",
            }
        )
    configure_logging(settings.log_dir)
    run = CrawlRun(keyword=keyword, number_of_pages=pages)
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def stage(name: str, action: Callable[[], T], count: int = 0) -> T:
        started = time.monotonic()
        try:
            value = action()
        except Exception as error:
            run.stages[name] = {"status": "FAILED", "duration": time.monotonic() - started}
            event(
                name,
                run_id=run.run_id,
                keyword=keyword,
                records_processed=count,
                records_success=0,
                records_failed=count,
                duration=time.monotonic() - started,
                status="FAILED",
                error=type(error).__name__,
            )
            raise
        status = (
            value.status
            if isinstance(value, CrawlResult)
            else value.get("status", "SUCCESS")
            if isinstance(value, dict)
            else "SUCCESS"
        )
        processed = len(value.records) if isinstance(value, CrawlResult) else count
        failed = value[1] if name == "validate_data" else 0
        succeeded = (
            sum(map(len, value[0].values())) if name == "validate_data" else processed - failed
        )
        run.stages[name] = {
            "status": status,
            "duration": time.monotonic() - started,
            "records_processed": processed,
            "records_success": succeeded,
            "records_failed": failed,
        }
        event(
            name,
            run_id=run.run_id,
            keyword=keyword,
            records_processed=processed,
            records_success=succeeded,
            records_failed=failed,
            duration=time.monotonic() - started,
            status=status,
        )
        return value

    # One local state owner prevents concurrent checkpoint, upload and summary races.
    with FileLock(str(settings.state_dir / "pipeline.lock"), timeout=1):
        atomic_json(settings.state_dir / f"run_{run.run_id}.json", run.model_dump(mode="json"))
        try:
            if crawl_only and upload_only:
                raise ValueError("--crawl-only and --upload-only are mutually exclusive")
            if upload_only:
                paths = sorted(settings.data_dir.glob("raw/**/*.jsonl")) + sorted(
                    settings.data_dir.glob("staged/**/*.parquet")
                )
                if not paths:
                    raise ValueError("No local raw or staged files available to upload")
            else:
                if fixture:
                    records = {
                        "products": [
                            json.loads(line)
                            for line in Path(fixture).read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ],
                        "reviews": [],
                    }
                    run.stages["crawl_products"] = {"status": "FIXTURE"}
                else:
                    products = stage(
                        "crawl_products", lambda: crawl_products(settings, run, resume)
                    )
                    if products.status == "SUCCESS":
                        reviews = stage(
                            "crawl_reviews", lambda: crawl_reviews(settings, run, skip_reviews)
                        )
                    else:
                        reviews = CrawlResult()
                        run.stages["crawl_reviews"] = {
                            "status": "SKIPPED",
                            "reason": "Product crawl stopped",
                        }
                    records = {"products": products.records, "reviews": reviews.records}
                    run.errors.extend(products.errors + reviews.errors)
                    if products.status != "SUCCESS":
                        run.status = products.status
                        run.stages["crawl_products"]["status"] = products.status
                    elif reviews.status not in {"SUCCESS", "DISABLED"}:
                        run.status = reviews.status
                        run.stages["crawl_reviews"]["status"] = reviews.status
                count = sum(map(len, records.values()))
                # Persist every observation before validation so unexpected parser failures cannot lose raw data.
                paths += stage(
                    "write_raw_files", lambda: write_raw_files(records, settings, run), count
                )
                valid, rejected = stage(
                    "validate_data", lambda: validate_data(records, settings, run), count
                )
                run.products_collected = len(valid["products"])
                run.reviews_collected = len(valid["reviews"])
                run.rejected_records = rejected
                run.stages["validate_data"].update(
                    records_processed=count,
                    records_success=run.products_collected + run.reviews_collected,
                    records_failed=rejected,
                    records_duplicates=count
                    - rejected
                    - run.products_collected
                    - run.reviews_collected,
                )
                if not run.products_collected and run.status == "RUNNING":
                    run.status = "NO_DATA"
                    run.errors.append("No valid products collected")
                paths += stage(
                    "convert_to_parquet",
                    lambda: convert_to_parquet(valid, settings, run),
                    count - rejected,
                )
            run.files = [str(path) for path in paths]
            if not crawl_only and run.status == "RUNNING":
                uploads = stage(
                    "upload_to_databricks",
                    lambda: upload_to_databricks(paths, settings, run),
                    len(paths),
                )
                run.stages["upload_to_databricks"]["files"] = uploads
                if not skip_databricks_job:
                    job = stage(
                        "trigger_databricks_job", lambda: trigger_databricks_job(settings, run)
                    )
                    run.stages["trigger_databricks_job"].update(job)
            if run.status == "RUNNING":
                run.status = "SUCCESS"
        except Exception as error:
            run.status = "FAILED"
            # SDK exceptions can include request context; do not serialize credentials.
            message = (
                str(error).replace(settings.databricks_token, "[REDACTED]")
                if settings.databricks_token
                else str(error)
            )
            run.errors.append(f"{type(error).__name__}: {message}")
        finally:
            run.crawl_finished_at = utc_now()
            run.files = [str(path) for path in paths]
            write_pipeline_summary(settings, run)
            atomic_json(settings.state_dir / f"run_{run.run_id}.json", run.model_dump(mode="json"))
            event(
                "summary",
                run_id=run.run_id,
                keyword=keyword,
                records_processed=run.products_collected
                + run.reviews_collected
                + run.rejected_records,
                records_success=run.products_collected + run.reviews_collected,
                records_failed=run.rejected_records,
                duration=(run.crawl_finished_at - run.crawl_started_at).total_seconds(),
                status=run.status,
            )
    summary = run.model_dump(mode="json")
    if run.status != "SUCCESS":
        raise PipelineFailed(summary)
    return summary

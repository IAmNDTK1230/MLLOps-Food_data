from datetime import timedelta
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunResultState

from src.config import Settings


def trigger_job(client: WorkspaceClient, settings: Settings, run_id: str) -> dict[str, Any]:
    if not settings.databricks_job_id:
        return {"status": "SKIPPED", "reason": "DATABRICKS_JOB_ID is not configured"}
    waiter = client.jobs.run_now(
        job_id=settings.databricks_job_id,
        idempotency_token=run_id,
        job_parameters={"run_id": run_id},
    )
    result = waiter.result(timeout=timedelta(minutes=settings.databricks_job_timeout_minutes))
    if not result.state or result.state.result_state != RunResultState.SUCCESS:
        raise RuntimeError(f"Databricks job failed: run_id={result.run_id}, state={result.state}")
    return {"status": "SUCCESS", "databricks_run_id": result.run_id, "url": result.run_page_url}

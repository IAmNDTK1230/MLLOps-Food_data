from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from databricks.sdk.service.jobs import RunResultState

from src.config import Settings
from src.databricks.jobs import trigger_job


def test_wait_for_job_success_and_idempotency():
    client = Mock()
    client.jobs.run_now.return_value.result.return_value = SimpleNamespace(
        run_id=10,
        run_page_url="https://workspace/jobs/1/runs/10",
        state=SimpleNamespace(result_state=RunResultState.SUCCESS),
    )
    result = trigger_job(client, Settings(databricks_job_id=1), "run-local")
    assert result["status"] == "SUCCESS" and result["databricks_run_id"] == 10
    client.jobs.run_now.assert_called_once_with(
        job_id=1, idempotency_token="run-local", job_parameters={"run_id": "run-local"}
    )
    client.jobs.run_now.return_value.result.assert_called_once()


def test_failed_remote_job_is_not_success():
    client = Mock()
    client.jobs.run_now.return_value.result.return_value = SimpleNamespace(
        run_id=10, state=SimpleNamespace(result_state=RunResultState.FAILED)
    )
    with pytest.raises(RuntimeError, match="job failed"):
        trigger_job(client, Settings(databricks_job_id=1), "run-local")


def test_missing_job_explicitly_skipped():
    client = Mock()
    assert trigger_job(client, Settings(), "run-local")["status"] == "SKIPPED"
    client.jobs.run_now.assert_not_called()

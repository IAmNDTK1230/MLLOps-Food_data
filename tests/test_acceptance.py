from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from src.config import Settings
from src.crawler.selectors import missing_cards_status
from src.databricks.preflight import bootstrap_sql, check_databricks
from src.databricks.uploader import Uploader
from src.orchestration.report import execution_report


def configured():
    return Settings(databricks_host="https://example.invalid", databricks_token="test-secret")


@pytest.mark.parametrize(
    "body,expected",
    [
        ("", "NO_DATA"),
        ("Không tìm thấy sản phẩm", "NO_DATA"),
        ("New public product layout", "DOM_CHANGED"),
    ],
)
def test_missing_cards_status(body, expected):
    assert missing_cards_status(body) == expected


def test_preflight_uses_profile_without_pat(monkeypatch):
    client = Mock()
    client.files.download.return_value = SimpleNamespace(
        contents=BytesIO(b"shopee preflight write/read probe\n")
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr("src.databricks.preflight.workspace_client", factory)
    settings = Settings(databricks_config_profile="shopee")
    assert check_databricks(settings)["status"] == "PASS"
    factory.assert_called_once_with(settings)
    client.current_user.me.assert_called_once_with()
    client.catalogs.get.assert_called_once_with("shopee")
    assert [call.args[0] for call in client.schemas.get.call_args_list] == [
        "shopee.raw",
        "shopee.bronze",
        "shopee.silver",
        "shopee.gold",
    ]
    client.volumes.read.assert_called_once_with("shopee.raw.landing")
    client.files.delete.assert_called_once_with(client.files.upload.call_args.args[0])


def test_preflight_missing_unified_auth_reports_safe_failure(monkeypatch):
    from src.databricks.client import DatabricksAuthenticationError

    monkeypatch.setattr(
        "src.databricks.preflight.workspace_client",
        Mock(side_effect=DatabricksAuthenticationError("configuration unavailable")),
    )
    report = check_databricks(Settings())
    assert report["status"] == "FAILED"
    assert report["checks"] == [
        {
            "name": "authentication",
            "status": "FAILED",
            "error_type": "DatabricksAuthenticationError",
        }
    ]


def test_preflight_write_verify_and_cleanup():
    client = Mock()
    client.files.download.return_value = SimpleNamespace(
        contents=BytesIO(b"shopee preflight write/read probe\n")
    )
    report = check_databricks(configured(), client)
    assert report["status"] == "PASS"
    assert client.schemas.get.call_count == 4
    probe = client.files.upload.call_args.args[0]
    assert probe.startswith(configured().volume_root + "/_preflight/probe_")
    assert client.files.upload.call_args.kwargs["overwrite"] is False
    client.files.delete.assert_called_once_with(probe)


def test_preflight_failure_redacts_exception_and_stops():
    client = Mock()
    client.current_user.me.side_effect = RuntimeError("test-secret")
    report = check_databricks(configured(), client)
    assert report["status"] == "FAILED" and "test-secret" not in str(report)
    client.catalogs.get.assert_not_called()


def test_preflight_read_failure_still_cleans_own_file():
    client = Mock()
    client.files.download.side_effect = RuntimeError("test-secret")
    assert check_databricks(configured(), client)["status"] == "FAILED"
    client.files.delete.assert_called_once_with(client.files.upload.call_args.args[0])


def test_bootstrap_uses_custom_configuration():
    sql = bootstrap_sql(
        Settings(
            databricks_catalog="custom", databricks_raw_schema="incoming", databricks_volume="files"
        )
    )
    assert "`custom`.`incoming`.`files`" in sql
    assert "`shopee`" not in sql and "IF NOT EXISTS" in sql


def test_report_never_invents_cloud_counts():
    report = execution_report(
        dict(
            run_id="run",
            status="SUCCESS",
            products_collected=1,
            reviews_collected=0,
            rejected_records=0,
            files=[],
        )
    )
    assert "NOT VERIFIED" in report and "Uploads: SKIPPED" in report


def test_uploader_rejects_shallow_and_outside_paths(tmp_path):
    uploader = Uploader(Mock(), Settings(data_dir=tmp_path, state_dir=tmp_path / "state"))
    for path in (tmp_path / "raw", tmp_path / "raw/products", tmp_path.parent / "outside.jsonl"):
        with pytest.raises(ValueError):
            uploader.remote_path(path)


def test_bundle_graph_and_python_files():
    root = Path(__file__).resolve().parents[1]
    bundle = yaml.safe_load((root / "databricks.yml").read_text())
    job = bundle["resources"]["jobs"]["data_pipeline"]
    assert bundle["variables"] == {
        name: {"default": value}
        for name, value in {
            "catalog": "shopee",
            "raw_schema": "raw",
            "bronze_schema": "bronze",
            "silver_schema": "silver",
            "gold_schema": "gold",
            "volume": "landing",
        }.items()
    }
    assert job["environments"] == [
        {"environment_key": "default", "spec": {"environment_version": "2"}}
    ]
    assert bundle["targets"] == {
        "dev": {"default": True, "mode": "development"},
        "test": {"mode": "development", "variables": {"catalog": "shopee_test"}},
    }
    classic_keys = {
        "cluster_id",
        "existing_cluster_id",
        "job_cluster_key",
        "job_clusters",
        "new_cluster",
    }

    def check_no_classic_compute(value):
        if isinstance(value, dict):
            assert not classic_keys.intersection(value)
            for nested in value.values():
                check_no_classic_compute(nested)
        elif isinstance(value, list):
            for nested in value:
                check_no_classic_compute(nested)

    check_no_classic_compute(bundle)
    tasks = {t["task_key"]: t for t in job["tasks"]}
    expected = {
        "setup": [],
        "bronze_products": ["setup"],
        "bronze_reviews": ["setup"],
        "silver_products": ["bronze_products"],
        "silver_reviews": ["bronze_reviews"],
        "gold_products": ["silver_products"],
        "gold_reviews": ["silver_reviews"],
    }
    assert job["max_concurrent_runs"] == 1 and set(tasks) == set(expected)
    for key, parents in expected.items():
        assert tasks[key]["environment_key"] == "default"
        assert (
            tasks[key]["spark_python_task"]["parameters"]
            == tasks["setup"]["spark_python_task"]["parameters"]
        )
        assert [d["task_key"] for d in tasks[key].get("depends_on", [])] == parents
        assert (root / tasks[key]["spark_python_task"]["python_file"]).is_file()

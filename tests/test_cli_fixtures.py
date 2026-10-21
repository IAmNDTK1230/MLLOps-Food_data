from unittest.mock import Mock

import pytest

from src import cli
from src.config import Settings
from src.models.crawl_run import CrawlRun
from src.orchestration import flow as module


@pytest.fixture
def invoke(monkeypatch):
    flow = Mock(
        return_value=CrawlRun(
            keyword="cloud acceptance", number_of_pages=1, status="SUCCESS"
        ).model_dump(mode="json")
    )
    monkeypatch.setattr(module, "shopee_data_pipeline", flow)

    def run(catalog, *options):
        monkeypatch.setattr(
            Settings, "from_env", classmethod(lambda cls: Settings(databricks_catalog=catalog))
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "cli",
                "pipeline",
                "--keyword",
                "cloud acceptance",
                "--pages",
                "1",
                "--fixture",
                "tests/fixtures/products.jsonl",
                *options,
            ],
        )
        return cli.main()

    return run, flow


def test_cloud_fixture_requires_explicit_allow_flag(invoke, capsys):
    run, flow = invoke
    with pytest.raises(SystemExit) as error:
        run("shopee_test")
    assert error.value.code == 2
    assert "--allow-synthetic-cloud-test" in capsys.readouterr().err
    flow.assert_not_called()


@pytest.mark.parametrize("catalog", ["shopee", "other", "SHOPEE_TEST", "shopee_test_extra"])
def test_cloud_fixture_rejects_any_catalog_other_than_exact_test(invoke, capsys, catalog):
    run, flow = invoke
    with pytest.raises(SystemExit) as error:
        run(catalog, "--allow-synthetic-cloud-test")
    assert error.value.code == 2
    assert (
        "Synthetic fixture cloud upload is only allowed when DATABRICKS_CATALOG=shopee_test"
        in capsys.readouterr().err
    )
    flow.assert_not_called()


def test_cloud_fixture_allowed_only_with_flag_and_test_catalog(invoke):
    run, flow = invoke
    assert run("shopee_test", "--allow-synthetic-cloud-test", "--skip-reviews") == 0
    args = flow.call_args.kwargs
    assert args["settings"].databricks_catalog == "shopee_test"
    assert args["allow_synthetic_cloud_test"] and not args["crawl_only"]
    assert args["fixture"] == "tests/fixtures/products.jsonl"
    assert not args["skip_databricks_job"]


def test_crawl_only_fixture_remains_allowed(invoke):
    run, flow = invoke
    assert run("shopee", "--crawl-only") == 0
    assert flow.call_args.kwargs["crawl_only"]
    assert not flow.call_args.kwargs["allow_synthetic_cloud_test"]


def test_fixture_upload_only_combination_rejected(invoke, capsys):
    run, flow = invoke
    with pytest.raises(SystemExit):
        run("shopee_test", "--allow-synthetic-cloud-test", "--upload-only")
    assert "--fixture cannot be combined with --upload-only" in capsys.readouterr().err
    flow.assert_not_called()

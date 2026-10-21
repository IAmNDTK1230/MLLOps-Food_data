"""One safety policy for both CLI and direct Prefect flow invocation."""

from src.config import Settings


def validate_fixture_mode(
    settings: Settings,
    fixture: str | None,
    *,
    crawl_only: bool,
    upload_only: bool,
    allow_synthetic_cloud_test: bool,
) -> None:
    if allow_synthetic_cloud_test and not fixture:
        raise ValueError("--allow-synthetic-cloud-test requires --fixture")
    if fixture and upload_only:
        raise ValueError("--fixture cannot be combined with --upload-only")
    if not fixture or crawl_only:
        return
    if settings.databricks_catalog != "shopee_test":
        raise ValueError(
            "Synthetic fixture cloud upload is only allowed when DATABRICKS_CATALOG=shopee_test"
        )
    if not allow_synthetic_cloud_test:
        raise ValueError(
            "--fixture is local-only unless --allow-synthetic-cloud-test is supplied "
            "with DATABRICKS_CATALOG=shopee_test"
        )

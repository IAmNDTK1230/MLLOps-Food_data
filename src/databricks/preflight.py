"""Small, bounded checks; never include SDK exception text or credentials in output."""

from io import BytesIO
from typing import Any
from uuid import uuid4

from src.config import Settings
from src.databricks.client import workspace_client


def check_databricks(settings: Settings, client: Any = None) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    probe = f"{settings.volume_root}/_preflight/probe_{uuid4().hex}.txt"
    uploaded = False
    operation = "authentication"
    try:
        client = workspace_client(settings) if client is None else client
        client.current_user.me()
        checks.append({"name": operation, "status": "PASS"})
        operation = "catalog"
        client.catalogs.get(settings.databricks_catalog)
        checks.append({"name": operation, "status": "PASS"})
        for schema in (
            settings.databricks_raw_schema,
            settings.databricks_bronze_schema,
            settings.databricks_silver_schema,
            settings.databricks_gold_schema,
        ):
            operation = f"schema:{schema}"
            client.schemas.get(f"{settings.databricks_catalog}.{schema}")
            checks.append({"name": operation, "status": "PASS"})
        operation = "volume"
        client.volumes.read(
            f"{settings.databricks_catalog}.{settings.databricks_raw_schema}.{settings.databricks_volume}"
        )
        checks.append({"name": operation, "status": "PASS"})
        operation = "volume_write_read"
        client.files.create_directory(probe.rsplit("/", 1)[0])
        payload = b"shopee preflight write/read probe\n"
        client.files.upload(probe, BytesIO(payload), overwrite=False)
        uploaded = True
        response = client.files.download(probe)
        try:
            if response.contents.read() != payload:
                raise ValueError("Probe content differs")
        finally:
            response.contents.close()
        checks.append({"name": operation, "status": "PASS"})
    except Exception as error:
        checks.append({"name": operation, "status": "FAILED", "error_type": type(error).__name__})
    finally:
        if uploaded:
            try:
                client.files.delete(probe)
                checks.append({"name": "probe_cleanup", "status": "PASS"})
            except Exception as error:
                checks.append(
                    {
                        "name": "probe_cleanup",
                        "status": "FAILED",
                        "error_type": type(error).__name__,
                    }
                )
    return {
        "status": "FAILED" if any(c["status"] == "FAILED" for c in checks) else "PASS",
        "checks": checks,
    }


def bootstrap_sql(settings: Settings) -> str:
    catalog = settings.databricks_catalog
    lines = [
        "-- Generated from Settings; regenerate with python -m src.cli databricks-bootstrap.",
        "-- Run as a Unity Catalog administrator or principal with CREATE CATALOG.",
        "-- Schema creation requires USE CATALOG and CREATE SCHEMA on the catalog.",
        "-- Volume creation requires USE SCHEMA and CREATE VOLUME on the raw schema.",
        "-- Runtime principal needs USE CATALOG, USE SCHEMA, READ/WRITE VOLUME,",
        "-- CREATE TABLE/MODIFY/SELECT on processing schemas and access to Serverless Jobs.",
        f"CREATE CATALOG IF NOT EXISTS `{catalog}`;",
    ]
    for schema in (
        settings.databricks_raw_schema,
        settings.databricks_bronze_schema,
        settings.databricks_silver_schema,
        settings.databricks_gold_schema,
    ):
        lines.append(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`;")
    lines.append(
        f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{settings.databricks_raw_schema}`.`{settings.databricks_volume}`;"
    )
    return "\n".join(lines) + "\n"

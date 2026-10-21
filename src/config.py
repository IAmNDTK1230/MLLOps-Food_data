import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    state_dir: Path = Field(default_factory=lambda: Path("state"))
    log_dir: Path = Field(default_factory=lambda: Path("logs"))
    crawl_concurrency: int = Field(default=1, ge=1, le=4)
    crawl_delay_seconds: float = Field(default=5, ge=1)
    crawl_timeout_ms: int = Field(default=30000, ge=1000)
    crawl_headless: bool = True
    reviews_enabled: bool = False
    databricks_config_profile: str = ""
    databricks_host: str = ""
    databricks_token: str = Field(default="", repr=False, exclude=True)
    databricks_catalog: str = "shopee"
    databricks_raw_schema: str = "raw"
    databricks_bronze_schema: str = "bronze"
    databricks_silver_schema: str = "silver"
    databricks_gold_schema: str = "gold"
    databricks_volume: str = "landing"
    databricks_job_id: int | None = Field(default=None, gt=0)
    databricks_job_timeout_minutes: int = Field(default=60, gt=0)
    crawl_cron: str = "0 2 * * *"
    crawl_timezone: str = "Asia/Ho_Chi_Minh"
    crawl_keyword: str = "tai nghe bluetooth"
    crawl_pages: int = Field(default=2, ge=1)

    @field_validator("databricks_config_profile")
    @classmethod
    def clean_profile(cls, value: str) -> str:
        return value.strip()

    @field_validator(
        "databricks_catalog",
        "databricks_raw_schema",
        "databricks_bronze_schema",
        "databricks_silver_schema",
        "databricks_gold_schema",
        "databricks_volume",
    )
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("Use simple SQL identifiers (letters, digits, underscore)")
        return value

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.databricks_catalog}/{self.databricks_raw_schema}/{self.databricks_volume}/shopee"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        # Prefect's event client treats an empty URL as a configured remote server.
        # Normalize optional blank configuration to truly unset for manual runs.
        if not os.environ.get("PREFECT_API_URL", "").strip():
            os.environ.pop("PREFECT_API_URL", None)
        return cls(
            **{
                name: os.environ[name.upper()]
                for name in cls.model_fields
                if os.environ.get(name.upper(), "") != ""
            }
        )

from typing import Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, Field

from src.utils.timestamps import utc_now

Status = Literal[
    "RUNNING", "SUCCESS", "FAILED", "BLOCKED", "CAPTCHA", "RATE_LIMITED", "NO_DATA", "DOM_CHANGED"
]


class CrawlRun(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    crawl_started_at: AwareDatetime = Field(default_factory=utc_now)
    crawl_finished_at: AwareDatetime | None = None
    keyword: str
    number_of_pages: int = Field(ge=1)
    products_collected: int = 0
    reviews_collected: int = 0
    rejected_records: int = 0
    errors: list[str] = Field(default_factory=list)
    status: Status = "RUNNING"
    stages: dict[str, Any] = Field(default_factory=dict)
    files: list[str] = Field(default_factory=list)
    metadata_file: str | None = None

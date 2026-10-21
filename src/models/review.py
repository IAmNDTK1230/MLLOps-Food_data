from datetime import UTC, datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


class Review(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, allow_inf_nan=False)
    review_id: str | None = None
    product_id: str = Field(min_length=1)
    shop_id: str | None = None
    rating: float = Field(ge=0, le=5)
    comment: str | None = None
    review_timestamp: AwareDatetime | None = None
    likes: int | None = Field(default=None, ge=0)
    crawl_timestamp: AwareDatetime

    @field_validator("crawl_timestamp", "review_timestamp")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value else None

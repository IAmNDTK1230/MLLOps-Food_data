from datetime import UTC, datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


class Product(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, allow_inf_nan=False)
    product_id: str = Field(min_length=1)
    shop_id: str | None = None
    product_name: str = Field(min_length=1)
    product_url: str | None = None
    shop_name: str | None = None
    shop_location: str | None = None
    category: str | None = None
    price: float | None = Field(default=None, ge=0)
    original_price: float | None = Field(default=None, ge=0)
    discount_percent: float | None = Field(default=None, ge=0, le=100)
    rating: float | None = Field(default=None, ge=0, le=5)
    rating_count: int | None = Field(default=None, ge=0)
    sold_count: int | None = Field(default=None, ge=0)
    stock: int | None = Field(default=None, ge=0)
    image_url: str | None = None
    keyword: str
    crawl_timestamp: AwareDatetime
    source: str = "shopee.vn"

    @field_validator("crawl_timestamp")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

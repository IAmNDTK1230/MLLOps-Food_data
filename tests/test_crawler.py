import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.config import Settings
from src.crawler.base import CrawlStopped
from src.crawler.selectors import enrich_card, restriction
from src.crawler.shopee_products import ShopeeProductCrawler
from src.utils.hashing import text_hash


@pytest.mark.parametrize(
    "code,url,text,captcha,expected",
    [
        (429, "https://shopee.vn/search", "", False, "RATE_LIMITED"),
        (403, "https://shopee.vn/search", "", False, "BLOCKED"),
        (200, "https://shopee.vn/buyer/login", "", False, "BLOCKED"),
        (200, "https://shopee.vn/search", "verify you are human", False, "CAPTCHA"),
        (200, "https://shopee.vn/search", "Products", True, "CAPTCHA"),
        (200, "https://shopee.vn/search", "Products", False, None),
    ],
)
def test_restrictions(code, url, text, captcha, expected):
    assert restriction(code, url, text, captcha) == expected


def test_card_identity():
    result = enrich_card(
        {"product_url": "/headphones-i.42.123", "_raw": {}}, "audio", "2026-10-03T00:00:00Z"
    )
    assert result["shop_id"] == "42" and result["product_id"] == "123"
    assert result["source"] == "shopee.vn"


def test_stop_prevents_next_navigation(tmp_path):
    async def check():
        crawler = ShopeeProductCrawler(Settings(state_dir=tmp_path), "run")
        crawler.stop.set()
        with pytest.raises(CrawlStopped):
            await crawler.pace()

    asyncio.run(check())


def test_auth_redirect_stops_before_dom_wait(tmp_path):
    async def check():
        crawler = ShopeeProductCrawler(Settings(state_dir=tmp_path), "run")
        page = SimpleNamespace(url="https://shopee.vn/buyer/login")
        with pytest.raises(CrawlStopped) as failure:
            await crawler.check_page(page)
        assert failure.value.status == "BLOCKED"
        assert crawler.stop.is_set()

    asyncio.run(check())


def test_resume_only_fetches_missing_pages(monkeypatch, tmp_path, product):
    from src.crawler import shopee_products

    checkpoint = tmp_path / "checkpoints" / f"{text_hash('audio:2')}.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(json.dumps({"pages": {"0": [product]}}), encoding="utf-8")
    context = SimpleNamespace(
        on=lambda *args: None,
        request=SimpleNamespace(
            get=AsyncMock(
                return_value=SimpleNamespace(
                    status=200,
                    url="https://shopee.vn/robots.txt",
                    text=AsyncMock(return_value="User-agent: *\nAllow: /"),
                )
            )
        ),
    )
    browser = SimpleNamespace(new_context=AsyncMock(return_value=context), close=AsyncMock())
    manager = AsyncMock()
    manager.__aenter__.return_value = SimpleNamespace(
        chromium=SimpleNamespace(launch=AsyncMock(return_value=browser))
    )
    monkeypatch.setattr(shopee_products, "async_playwright", lambda: manager)
    crawler = ShopeeProductCrawler(Settings(state_dir=tmp_path), "run")
    fetch = AsyncMock(return_value=[{**product, "product_id": "456"}])
    monkeypatch.setattr(crawler, "fetch_page", fetch)
    result = asyncio.run(crawler.crawl("audio", 2, resume=True))
    assert result.status == "SUCCESS" and len(result.records) == 2
    assert fetch.await_count == 1
    assert "page=1" in fetch.await_args.args[1]
    assert len(json.loads(checkpoint.read_text(encoding="utf-8"))["pages"]) == 2

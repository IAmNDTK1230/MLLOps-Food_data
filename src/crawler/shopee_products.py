import asyncio
import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.robotparser import RobotFileParser

from playwright.async_api import BrowserContext, Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings
from src.crawler.base import BaseCrawler, CrawlResult, CrawlStopped, TransientCrawlError
from src.crawler.selectors import (
    BASE_URL,
    CAPTCHA_SELECTOR,
    CARD_SELECTOR,
    CRAWLER_AGENT,
    EXTRACT_CARDS,
    enrich_card,
    missing_cards_status,
    restriction,
)
from src.storage.manifest import atomic_json
from src.utils.hashing import text_hash
from src.utils.logging import event
from src.utils.timestamps import utc_now


class ShopeeProductCrawler(BaseCrawler):
    def __init__(self, settings: Settings, run_id: str) -> None:
        self.settings = settings
        self.run_id = run_id
        self.stop = asyncio.Event()
        self.rate_lock = asyncio.Lock()
        self.next_request = 0.0
        self.network_restriction: str | None = None

    async def pace(self) -> None:
        async with self.rate_lock:
            await asyncio.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + self.settings.crawl_delay_seconds
        if self.stop.is_set():
            raise CrawlStopped(self.network_restriction or "BLOCKED", "Crawl halted")

    async def check_page(self, page: Page, code: int | None = None) -> None:
        immediate = self.network_restriction or restriction(code, page.url, "", False)
        if immediate:
            self.stop.set()
            raise CrawlStopped(immediate, f"Public search stopped: {immediate}")
        text = await page.locator("body").inner_text(timeout=self.settings.crawl_timeout_ms)
        captcha = page.locator(CAPTCHA_SELECTOR)
        visible = any([await captcha.nth(i).is_visible() for i in range(await captcha.count())])
        status = self.network_restriction or restriction(code, page.url, text, visible)
        if status:
            self.stop.set()
            raise CrawlStopped(status, f"Public search stopped: {status}")

    @retry(
        retry=retry_if_exception_type((PlaywrightTimeoutError, TransientCrawlError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=20),
        reraise=True,
    )
    async def fetch_page(
        self, context: BrowserContext, url: str, keyword: str
    ) -> list[dict[str, Any]]:
        await self.pace()
        page = await context.new_page()
        try:
            response = await page.goto(
                url, wait_until="domcontentloaded", timeout=self.settings.crawl_timeout_ms
            )
            await self.check_page(page, response.status if response else None)
            if response and response.status in {500, 502, 503, 504}:
                raise TransientCrawlError(f"Temporary search HTTP {response.status}")
            if response and response.status >= 400:
                raise RuntimeError(f"Search HTTP {response.status}")
            try:
                await page.locator(CARD_SELECTOR).first.wait_for(
                    timeout=self.settings.crawl_timeout_ms
                )
            except PlaywrightTimeoutError:
                await self.check_page(page)
                body = await page.locator("body").inner_text()
                diagnostic = self.settings.state_dir / "public_dom" / f"{self.run_id}.json"
                atomic_json(
                    diagnostic,
                    {
                        "url": page.url,
                        "http_status": response.status if response else None,
                        "title": await page.title(),
                        "body_text": body[:12000],
                        "status": missing_cards_status(body),
                    },
                )
                raise CrawlStopped(
                    missing_cards_status(body),
                    f"No public product cards; rendered DOM evidence: {diagnostic}",
                ) from None
            await self.check_page(page)
            cards = await page.evaluate(EXTRACT_CARDS, CARD_SELECTOR)
            timestamp = utc_now().isoformat()
            return [enrich_card(card, keyword, timestamp) for card in cards]
        except PlaywrightTimeoutError:
            if page.url != "about:blank":
                try:
                    await self.check_page(page)
                except PlaywrightTimeoutError:
                    pass
            # A restriction arriving during navigation must not trigger a navigation retry.
            if self.stop.is_set():
                raise CrawlStopped(
                    self.network_restriction or "BLOCKED", "Blocked during navigation"
                ) from None
            raise
        finally:
            await page.close()

    async def crawl(self, keyword: str, pages: int, *, resume: bool = False) -> CrawlResult:
        checkpoint = (
            self.settings.state_dir / "checkpoints" / f"{text_hash(f'{keyword}:{pages}')}.json"
        )
        saved = (
            json.loads(checkpoint.read_text(encoding="utf-8"))
            if resume and checkpoint.exists()
            else {"pages": {}}
        )
        observations = saved["pages"]
        result = CrawlResult()
        checkpoint_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.settings.crawl_concurrency)

        def monitor(response: object) -> None:
            code = getattr(response, "status", None)
            url = getattr(response, "url", "")
            if "shopee.vn" in url and code in {401, 403, 429}:
                self.network_restriction = "RATE_LIMITED" if code == 429 else "BLOCKED"
                self.stop.set()

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=self.settings.crawl_headless)
            try:
                context = await browser.new_context(
                    locale="vi-VN", timezone_id="UTC", user_agent=CRAWLER_AGENT
                )
                context.on("response", monitor)
                robots_response = await context.request.get(
                    BASE_URL + "/robots.txt", timeout=self.settings.crawl_timeout_ms
                )
                if robots_response.status != 200:
                    status = "RATE_LIMITED" if robots_response.status == 429 else "BLOCKED"
                    return CrawlResult(
                        status=status,
                        errors=[
                            f"Cannot verify robots.txt policy: HTTP {robots_response.status} at {robots_response.url}; stopped"
                        ],
                    )
                robots_text = await robots_response.text()
                robots_block = restriction(
                    robots_response.status, robots_response.url, robots_text, False
                )
                if robots_block or "<html" in robots_text.lower():
                    return CrawlResult(
                        status=robots_block or "BLOCKED",
                        errors=["robots.txt returned a challenge or HTML"],
                    )
                robots = RobotFileParser()
                robots.parse(robots_text.splitlines())
                self.next_request = time.monotonic() + self.settings.crawl_delay_seconds
                robot_delay = robots.crawl_delay(CRAWLER_AGENT)
                if robot_delay:
                    self.settings.crawl_delay_seconds = max(
                        self.settings.crawl_delay_seconds, robot_delay
                    )

                async def worker(index: int) -> None:
                    if str(index) in observations:
                        return
                    async with semaphore:
                        if self.stop.is_set():
                            return
                        url = BASE_URL + "/search?" + urlencode({"keyword": keyword, "page": index})
                        try:
                            if not robots.can_fetch(CRAWLER_AGENT, url):
                                raise CrawlStopped("BLOCKED", "robots.txt disallows public search")
                            records = await self.fetch_page(context, url, keyword)
                            async with checkpoint_lock:
                                observations[str(index)] = records
                                atomic_json(
                                    checkpoint,
                                    {
                                        "keyword": keyword,
                                        "requested_pages": pages,
                                        "pages": observations,
                                    },
                                )
                            event(
                                "crawler_page",
                                run_id=self.run_id,
                                keyword=keyword,
                                page=index + 1,
                                records_processed=len(records),
                                status="SUCCESS",
                            )
                        except CrawlStopped as error:
                            self.stop.set()
                            if result.status == "SUCCESS":
                                result.status = error.status
                            result.errors.append(str(error))
                        except Exception as error:
                            self.stop.set()
                            result.status = "FAILED"
                            result.errors.append(
                                f"Page {index + 1}: {type(error).__name__}: {error}"
                            )

                await asyncio.gather(*(worker(index) for index in range(pages)))
                if self.network_restriction:
                    result.status = self.network_restriction
                    result.errors.append("Public site returned a restricted response")
            finally:
                await browser.close()
        seen: set[tuple[str, str]] = set()
        for index in sorted(observations, key=int):
            for record in observations[index]:
                key = (record.get("shop_id"), record.get("product_id"))
                if key not in seen or key[1] is None:
                    result.records.append(record)
                    seen.add(key)
        return result

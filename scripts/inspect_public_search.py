"""One rendered DOM inspection after a selector miss; no private APIs or bypass."""

import asyncio
import json
import sys

from playwright.async_api import async_playwright

from src.config import Settings
from src.crawler.selectors import BASE_URL, CARD_SELECTOR, CRAWLER_AGENT
from src.crawler.shopee_products import ShopeeProductCrawler
from src.storage.manifest import atomic_json


async def inspect() -> None:
    settings = Settings.from_env()
    destination = settings.state_dir / "public_dom"
    destination.mkdir(parents=True, exist_ok=True)
    crawler = ShopeeProductCrawler(settings, "public-dom-inspection")
    evidence = {}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(user_agent=CRAWLER_AGENT, locale="vi-VN")
            robots = await context.request.get(BASE_URL + "/robots.txt")
            evidence["robots_http_status"] = robots.status
            if robots.status != 200:
                evidence["status"] = "BLOCKED"
                return
            from urllib.robotparser import RobotFileParser

            policy = RobotFileParser()
            robots_text = await robots.text()
            policy.parse(robots_text.splitlines())
            url = BASE_URL + "/search?keyword=tai+nghe+bluetooth&page=0"
            if "<html" in robots_text.lower() or not policy.can_fetch(CRAWLER_AGENT, url):
                evidence["status"] = "BLOCKED"
                return
            await asyncio.sleep(
                max(5, settings.crawl_delay_seconds, policy.crawl_delay(CRAWLER_AGENT) or 0)
            )
            page = await context.new_page()
            response = await page.goto(url, wait_until="domcontentloaded")
            evidence["http_status"] = response.status if response else None
            # Check immediately, then after the ordinary rendering wait. No navigation retries.
            await crawler.check_page(page, evidence["http_status"])
            await page.wait_for_timeout(5000)
            await crawler.check_page(page)
            evidence.update(
                url=page.url,
                title=await page.title(),
                product_cards=await page.locator(CARD_SELECTOR).count(),
                body_text=(await page.locator("body").inner_text())[:12000],
                links=await page.locator("a[href]").evaluate_all(
                    "els => els.map(e => ({text:e.innerText, href:e.href})).slice(0,100)"
                ),
                status="INSPECTED",
            )
            (destination / "search.html").write_text(await page.content(), encoding="utf-8")
            await page.screenshot(path=str(destination / "search.png"), full_page=True)
        except Exception as error:
            evidence.update(status=getattr(error, "status", "FAILED"), error=str(error))
            if "page" in locals():
                evidence.update(
                    url=page.url,
                    title=await page.title(),
                    body_text=(await page.locator("body").inner_text())[:12000],
                )
                (destination / "search.html").write_text(await page.content(), encoding="utf-8")
                await page.screenshot(path=str(destination / "search.png"), full_page=True)
        finally:
            await browser.close()
            atomic_json(destination / "evidence.json", evidence)
            print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(inspect())

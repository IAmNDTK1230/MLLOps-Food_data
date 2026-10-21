"""One anonymous visible-browser visit; public DOM/static-resource metadata only."""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from uuid import uuid4

from playwright.async_api import async_playwright

from src.config import Settings
from src.crawler.selectors import (
    BASE_URL,
    CAPTCHA_SELECTOR,
    CARD_SELECTOR,
    CRAWLER_AGENT,
    restriction,
)
from src.storage.manifest import atomic_json
from src.utils.timestamps import utc_now

STATIC_TYPES = {"document", "script", "stylesheet", "image", "font", "media"}
OUTCOMES = {
    "SUCCESS",
    "DOM_CHANGED",
    "PUBLIC_PAGE_EMPTY",
    "LOGIN_REQUIRED",
    "CAPTCHA",
    "BLOCKED",
    "RATE_LIMITED",
    "NETWORK_FAILURE",
    "JAVASCRIPT_FAILURE",
    "UNKNOWN",
}


def public_resource(resource_type: str, url: str) -> bool:
    return resource_type in STATIC_TYPES and "/api/" not in urlsplit(url).path.lower()


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    if "/api/" in parts.path.lower():
        return "<excluded-dynamic-request>"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def safe_message(message: str) -> str:
    message = re.sub(r"https?://[^\s)]+", lambda match: safe_url(match.group()), message)
    return re.sub(
        r"(?i)(token|authorization|password|secret)\s*[:=]\s*\S+", r"\1=[REDACTED]", message
    )[:2000]


def classify(evidence: dict) -> str:
    snapshots = evidence.get("snapshots", [])
    for snapshot in snapshots:
        if "/buyer/login" in snapshot.get("url", "").lower():
            return "LOGIN_REQUIRED"
        restricted = restriction(
            snapshot.get("main_document_status"),
            snapshot.get("url", ""),
            snapshot.get("body_text", ""),
            snapshot.get("captcha_visible", False),
        )
        if restricted:
            return restricted
    if evidence.get("stop_status"):
        return evidence["stop_status"]
    if evidence.get("navigation_failed"):
        return "NETWORK_FAILURE"
    if not snapshots:
        return "UNKNOWN"
    final = snapshots[-1]
    if final.get("visible_product_cards", 0):
        return "SUCCESS"
    if final.get("visible_product_links", 0):
        return "DOM_CHANGED"
    if evidence.get("page_errors"):
        return "JAVASCRIPT_FAILURE"
    if not final.get("body_text", "").strip() and not final.get("visible_images", 0):
        return "PUBLIC_PAGE_EMPTY"
    return "UNKNOWN"


async def diagnose(keyword: str, destination: Path) -> dict:
    settings = Settings.from_env()
    destination.mkdir(parents=True, exist_ok=False)
    evidence = {
        "started_at": utc_now().isoformat(),
        "headless": False,
        "concurrency": 1,
        "delay_seconds": max(5, settings.crawl_delay_seconds),
        "keyword": keyword,
        "snapshots": [],
        "console": [],
        "page_errors": [],
        "failed_requests": [],
        "responses": [],
        "capture_errors": [],
        "outcome": "UNKNOWN",
        "products_collected": 0,
        "excluded": "XHR/fetch/WebSocket requests, private API URLs, response bodies, headers and cookies",
    }
    main_status = None
    navigation_start = None
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        evidence["browser_version"] = browser.version
        try:
            context = await browser.new_context(
                user_agent=CRAWLER_AGENT, locale="vi-VN", viewport={"width": 1365, "height": 900}
            )
            robots = await context.request.get(
                BASE_URL + "/robots.txt", timeout=settings.crawl_timeout_ms
            )
            evidence["robots_http_status"] = robots.status
            robots_text = await robots.text()
            (destination / "robots.txt").write_text(robots_text, encoding="utf-8")
            if robots.status != 200 or "<html" in robots_text.lower():
                evidence["stop_status"] = "RATE_LIMITED" if robots.status == 429 else "BLOCKED"
                return evidence
            policy = RobotFileParser()
            policy.parse(robots_text.splitlines())
            url = BASE_URL + "/search?" + urlencode({"keyword": keyword, "page": 0})
            if not policy.can_fetch(CRAWLER_AGENT, url):
                evidence["stop_status"] = "BLOCKED"
                evidence["reason"] = "robots.txt disallows this public search"
                return evidence
            await asyncio.sleep(
                max(evidence["delay_seconds"], policy.crawl_delay(CRAWLER_AGENT) or 0)
            )
            page = await context.new_page()

            def response_event(response):
                nonlocal main_status
                request = response.request
                if not public_resource(request.resource_type, response.url):
                    return
                evidence["responses"].append(
                    {
                        "url": safe_url(response.url),
                        "resource_type": request.resource_type,
                        "status": response.status,
                    }
                )
                if request.resource_type == "document" and request.frame == page.main_frame:
                    main_status = response.status

            page.on("response", response_event)
            page.on(
                "requestfailed",
                lambda request: evidence["failed_requests"].append(
                    {
                        "url": safe_url(request.url),
                        "resource_type": request.resource_type,
                        "failure": safe_message(request.failure or "unknown"),
                    }
                )
                if public_resource(request.resource_type, request.url)
                else None,
            )
            page.on(
                "console",
                lambda message: evidence["console"].append(
                    {
                        "type": message.type,
                        "text": safe_message(message.text),
                        "location": safe_url(message.location.get("url", "")),
                    }
                ),
            )
            page.on(
                "pageerror",
                lambda error: evidence["page_errors"].append(
                    {"message": safe_message(str(error)), "at": utc_now().isoformat()}
                ),
            )

            async def capture(label):
                metrics = await page.evaluate("""() => {
                    const visible = e => !!(e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden' && getComputedStyle(e).display !== 'none');
                    return {ready_state: document.readyState, body_html_length: document.body?.innerHTML.length || 0,
                        body_text_length: document.body?.innerText.length || 0, body_text: (document.body?.innerText || '').slice(0,12000),
                        links: document.querySelectorAll('a[href]').length, divs: document.querySelectorAll('div').length,
                        scripts: document.querySelectorAll('script').length, iframes: document.querySelectorAll('iframe').length,
                        visible_images: [...document.images].filter(visible).length,
                        visible_product_links: [...document.querySelectorAll('a[href]')].filter(e => visible(e) && /-i\\.\\d+\\.\\d+|\\/product\\/\\d+\\/\\d+/.test(e.href)).length};
                }""")
                cards = page.locator(CARD_SELECTOR)
                captchas = page.locator(CAPTCHA_SELECTOR)
                metrics.update(
                    label=label,
                    captured_at=utc_now().isoformat(),
                    seconds_since_navigation=round(time.monotonic() - navigation_start, 3),
                    url=page.url,
                    title=await page.title(),
                    main_document_status=main_status,
                    visible_product_cards=sum(
                        [await cards.nth(i).is_visible() for i in range(await cards.count())]
                    ),
                    captcha_visible=any(
                        [await captchas.nth(i).is_visible() for i in range(await captchas.count())]
                    ),
                )
                evidence["snapshots"].append(metrics)
                (destination / f"{label}.html").write_text(await page.content(), encoding="utf-8")
                try:
                    await page.screenshot(
                        path=str(destination / f"{label}.png"), full_page=True, timeout=10000
                    )
                except Exception as error:
                    evidence["capture_errors"].append(
                        {"point": label, "error_type": type(error).__name__}
                    )
                status = classify({"snapshots": [metrics]})
                if status in {"LOGIN_REQUIRED", "CAPTCHA", "BLOCKED", "RATE_LIMITED"}:
                    evidence["stop_status"] = status
                    return False
                return True

            navigation_start = time.monotonic()
            try:
                await page.goto(url, wait_until="commit", timeout=settings.crawl_timeout_ms)
            except Exception as error:
                evidence["navigation_failed"] = True
                evidence["navigation_error"] = {
                    "error_type": type(error).__name__,
                    "message": safe_message(str(error)),
                }
                return evidence
            if not await capture("A_navigation"):
                return evidence
            for state, label in (("domcontentloaded", "B_domcontentloaded"), ("load", "C_load")):
                try:
                    await page.wait_for_load_state(state, timeout=settings.crawl_timeout_ms)
                except Exception as error:
                    evidence["capture_errors"].append(
                        {"point": label, "wait_error": type(error).__name__}
                    )
                if not await capture(label):
                    return evidence
            # D/E are five/ten seconds after the completed C capture. Actual timings are saved.
            after_load = time.monotonic()
            for seconds, label in ((5, "D_after_5s"), (10, "E_after_10s")):
                await asyncio.sleep(max(0, after_load + seconds - time.monotonic()))
                if not await capture(label):
                    return evidence
        except Exception as error:
            evidence["diagnostic_error"] = {
                "error_type": type(error).__name__,
                "message": safe_message(str(error)),
            }
        finally:
            await browser.close()
            evidence["browser_closed"] = not browser.is_connected()
            evidence["finished_at"] = utc_now().isoformat()
            evidence["outcome"] = classify(evidence)
            atomic_json(destination / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyword", default="tai nghe bluetooth")
    parser.add_argument("--output", default=f"state/visible-diagnostic/{uuid4().hex}")
    args = parser.parse_args()
    result = asyncio.run(diagnose(args.keyword, Path(args.output)))
    print(
        json.dumps(
            {
                "outcome": result["outcome"],
                "evidence": str(Path(args.output) / "evidence.json"),
                "snapshots": len(result["snapshots"]),
                "page_errors": len(result["page_errors"]),
                "failed_public_requests": len(result["failed_requests"]),
                "products_collected": result["products_collected"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

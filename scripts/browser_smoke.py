"""Launch, navigate and close a real browser without authentication or workarounds."""

import asyncio
import json
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

from src.storage.manifest import atomic_json


async def smoke() -> dict[str, Any]:
    result: dict[str, Any] = {"launched": False, "navigation": False, "closed": False}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        result.update(launched=True, version=browser.version)
        try:
            page = await browser.new_page()
            response = await page.goto("https://example.com", wait_until="domcontentloaded")
            result.update(
                url=page.url,
                http_status=response.status if response else None,
                title=await page.title(),
            )
            result["navigation"] = bool(
                response and response.ok and result["title"] == "Example Domain"
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            await browser.close()
            result["closed"] = not browser.is_connected()
    return result


if __name__ == "__main__":
    outcome = asyncio.run(smoke())
    atomic_json(Path("state/browser_smoke.json"), outcome)
    print(json.dumps(outcome, indent=2))
    raise SystemExit(
        0 if all(outcome.get(key) for key in ("launched", "navigation", "closed")) else 1
    )

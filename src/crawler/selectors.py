"""All Shopee HTML assumptions live here; no hidden API or scaled API prices."""

import re
from typing import Any
from urllib.parse import urljoin

BASE_URL = "https://shopee.vn"
CRAWLER_AGENT = "ShopeeDataPlatform/0.1"
CARD_SELECTOR = '[data-sqe="item"], .shopee-search-item-result__item'
CAPTCHA_SELECTOR = 'iframe[src*="captcha"], [id*="captcha"], [class*="captcha"]'
PRODUCT_LINK = re.compile(r"(?:-i\.|/product/)(\d+)[./](\d+)")
CAPTCHA_TEXT = ("verify you are human", "xác minh bạn là con người", "complete the captcha")
BLOCK_TEXT = ("access denied", "unusual traffic", "tài khoản của bạn", "security verification")
EMPTY_RESULTS_TEXT = ("không tìm thấy kết quả", "không tìm thấy sản phẩm", "no results found")


def missing_cards_status(body: str) -> str:
    if not body.strip() or any(text in body.lower() for text in EMPTY_RESULTS_TEXT):
        return "NO_DATA"
    return "DOM_CHANGED"


# DOM reads only. Missing optional fields remain null; do not infer stock/shop/rating.
EXTRACT_CARDS = r"""selector => [...document.querySelectorAll(selector)].map(card => {
  const text = s => card.querySelector(s)?.textContent?.trim() || null;
  const a = [...card.querySelectorAll('a[href]')].find(x => /-i\.\d+\.\d+|\/product\/\d+\/\d+/.test(x.href));
  return {
    product_url: a?.href || null,
    product_name: text('[data-sqe="name"], .line-clamp-2, .shopee-item-card__text-name'),
    price: text('[data-sqe="price"], .shopee-item-card__current-price'),
    original_price: text('.shopee-item-card__original-price'),
    sold_count: text('.shopee-item-card__sold, [data-sqe="sold"]'),
    shop_location: text('.shopee-item-card__location'),
    image_url: card.querySelector('img')?.src || null,
    _raw: {html: card.outerHTML, text: card.innerText}
  };
})"""


def restriction(status_code: int | None, url: str, text: str, captcha_visible: bool) -> str | None:
    if status_code == 429:
        return "RATE_LIMITED"
    if captcha_visible or "captcha" in url.lower() or any(t in text.lower() for t in CAPTCHA_TEXT):
        return "CAPTCHA"
    if (
        status_code in {401, 403}
        or "/buyer/login" in url
        or any(t in text.lower() for t in BLOCK_TEXT)
    ):
        return "BLOCKED"
    return None


def enrich_card(card: dict[str, Any], keyword: str, timestamp: str) -> dict[str, Any]:
    match = PRODUCT_LINK.search(card.get("product_url") or "")
    return {
        **card,
        "product_id": match.group(2) if match else None,
        "shop_id": match.group(1) if match else None,
        "product_url": urljoin(BASE_URL, card["product_url"]) if card.get("product_url") else None,
        "keyword": keyword,
        "crawl_timestamp": timestamp,
        "source": "shopee.vn",
    }

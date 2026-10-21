import pytest

from scripts.shopee_visible_diagnostic import OUTCOMES, classify, public_resource


def snapshot(**values):
    return {
        "url": "https://shopee.vn/search",
        "main_document_status": 200,
        "body_text": "",
        "visible_product_cards": 0,
        "visible_product_links": 0,
        "visible_images": 0,
        **values,
    }


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, "PUBLIC_PAGE_EMPTY"),
        ({"body_text": "Navigation without products"}, "UNKNOWN"),
        ({"visible_product_cards": 1}, "SUCCESS"),
        ({"visible_product_links": 1, "body_text": "Public product"}, "DOM_CHANGED"),
        ({"url": "https://shopee.vn/buyer/login"}, "LOGIN_REQUIRED"),
        ({"captcha_visible": True}, "CAPTCHA"),
        ({"main_document_status": 403}, "BLOCKED"),
        ({"main_document_status": 429}, "RATE_LIMITED"),
    ],
)
def test_outcome_requires_evidence(changes, expected):
    result = classify({"snapshots": [snapshot(**changes)]})
    assert result == expected and result in OUTCOMES


def test_caught_console_error_does_not_establish_javascript_failure():
    assert (
        classify({"snapshots": [snapshot()], "console": [{"text": "TypeError"}], "page_errors": []})
        == "PUBLIC_PAGE_EMPTY"
    )
    assert (
        classify({"snapshots": [snapshot()], "page_errors": [{"message": "Uncaught error"}]})
        == "JAVASCRIPT_FAILURE"
    )


def test_navigation_failure():
    assert classify({"navigation_failed": True}) == "NETWORK_FAILURE"


def test_dynamic_and_private_resources_excluded():
    assert public_resource("script", "https://cdn.example/public.js")
    assert not public_resource("fetch", "https://shopee.vn/search")
    assert not public_resource("xhr", "https://shopee.vn/api/private")
    assert not public_resource("script", "https://shopee.vn/api/private")

"""Transparent Vietnamese restaurant query intent and limited expansion."""

import re
from dataclasses import dataclass

ASPECT_TERMS = {
    "service": ("phục vụ", "nhân viên", "thái độ", "dịch vụ", "chờ", "đợi"),
    "food": ("món", "đồ ăn", "thức ăn", "hương vị", "đồ uống"),
    "price": ("giá", "đắt", "mắc", "rẻ", "tiền"),
    "ambiance": ("không gian", "chỗ ngồi", "ồn", "đông", "bàn ghế"),
    "cleanliness": ("sạch", "bẩn", "vệ sinh"),
}
NEGATIVE = ("phàn nàn", "vấn đề", "kém", "tệ", "chậm", "đắt", "mắc", "dở",
            "không ngon", "không tốt", "ồn", "bẩn", "thất vọng", "khó chịu",
            "giá cao", "giá quá cao")
POSITIVE = ("khen", "ngon", "tốt", "nhiệt tình", "hài lòng", "rẻ", "sạch")
EXPANSIONS = {
    "service": {"complaint": ("nhân viên", "thái độ", "chậm", "chờ"),
                "general": ("nhân viên", "dịch vụ")},
    "food": {"complaint": ("món", "dở", "vị"),
             "general": ("món", "thức ăn", "hương vị")},
    "price": {"complaint": ("giá", "mắc", "đắt"),
              "general": ("giá", "tiền")},
    "ambiance": {"complaint": ("ồn", "chật", "chỗ ngồi"),
                 "general": ("không gian", "chỗ ngồi")},
    "cleanliness": {"complaint": ("bẩn", "vệ sinh"),
                    "general": ("sạch", "vệ sinh")},
}
ISSUE_TERMS = (
    (("chậm", "chờ", "đợi"), ("chậm", "lâu", "chờ", "đợi")),
    (("ồn",), ("ồn", "la hét", "ồn ào")),
    (("không ngon", "dở"), ("không ngon", "dở", "tệ", "mặn", "nhạt")),
    (("thái độ",), ("thái độ", "nạt", "hỗn", "khinh")),
    (("giá cao", "giá quá cao", "đắt", "mắc"), ("giá cao", "đắt", "mắc")),
)
PRICE_COMPLAINT = re.compile(
    r"(?:giá(?:\s+\S+){0,3}\s+(?:cao|đắt|mắc|chát|không\s+rẻ|quá\s+cao)"
    r"|(?:đắt|mắc|chát)\s+(?:đỏ|quá|so\s+với)?\s*(?:tiền|giá)?"
    r"|không\s+đáng\s+(?:giá|tiền))",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryIntent:
    aspect: str | None
    intent: str
    original_query: str
    expanded_query_terms: tuple[str, ...]


def detect(query):
    text = re.sub(r"\s+", " ", query.casefold()).strip()
    scores = {aspect: sum(term in text for term in terms)
              for aspect, terms in ASPECT_TERMS.items()}
    best = max(scores.values())
    winners = [aspect for aspect, score in scores.items() if score == best and score > 0]
    aspect = winners[0] if len(winners) == 1 else None
    if "tại sao" in text and ("giảm" in text or "xuống" in text):
        intent = "causal"
    elif any(term in text for term in NEGATIVE):
        intent = "complaint"
    elif any(term in text for term in POSITIVE):
        intent = "praise"
    else:
        intent = "general"
    category = "complaint" if intent == "complaint" else "general"
    expanded = tuple(term for term in EXPANSIONS.get(aspect, {}).get(category, ())
                     if term not in text)[:4]
    return QueryIntent(aspect, intent, query, expanded)


def evidence_compatible(hit, intent, aspect=None):
    """Conservative pre-generation gate; metadata alone is not semantic proof."""
    text = hit.text.casefold()
    aspects = hit.metadata.get("predicted_aspects", "[]")
    if isinstance(aspects, str):
        import json

        aspects = json.loads(aspects)
    if aspect and aspect not in aspects:
        return False
    if aspect and not any(term in text for term in ASPECT_TERMS.get(aspect, ())):
        return False
    negative_text = any(term in text for term in NEGATIVE) or bool(
        aspect == "price" and PRICE_COMPLAINT.search(text)
    )
    positive_text = any(term in text for term in POSITIVE)
    if intent == "complaint":
        if aspect == "price" and not PRICE_COMPLAINT.search(text):
            return False
        return negative_text and (hit.metadata.get("predicted_sentiment") == "negative"
                                  or not positive_text)
    if intent == "praise":
        return positive_text and hit.metadata.get("predicted_sentiment") != "negative"
    return True


def issue_matches(text, query):
    query, text = query.casefold(), text.casefold()
    selected = [terms for triggers, terms in ISSUE_TERMS
                if any(trigger in query for trigger in triggers)]
    if not selected:
        return None
    return all(any(term in text for term in terms) for terms in selected)

"""Fixed Vietnamese lexicon baseline, authored before held-out evaluation.

NFC/whitespace then casefold; longest non-overlapping lexicon phrases win.
Each sentiment hit contributes +/-1. An odd count of negators in the previous
three tokens flips it. Punctuation/adversatives end scope. Intensifiers multiply
the hit by 1.5. Positive total -> positive, negative -> negative, zero -> neutral.
No labels, aspects, rating scores, training, or automatic lexicon mining are used.
"""

import re

from src.ml.features import normalize_text

POSITIVE = (
    "ngon",
    "tốt",
    "tuyệt",
    "tuyệt vời",
    "thích",
    "sạch",
    "sạch sẽ",
    "nhanh",
    "thân thiện",
    "hài lòng",
    "hấp dẫn",
    "đậm đà",
    "vừa miệng",
    "ngon miệng",
    "ngon lành",
    "tươi",
    "tươi ngon",
    "nhiệt tình",
    "chu đáo",
    "dễ chịu",
    "lịch sự",
    "rẻ",
    "hợp lý",
    "phải chăng",
    "ấm cúng",
    "thoáng",
    "thoải mái",
    "đẹp",
    "ổn",
    "chất lượng",
    "xuất sắc",
    "ưng ý",
    "đáng tiền",
)
NEGATIVE = (
    "dở",
    "tệ",
    "chậm",
    "bẩn",
    "đắt",
    "nguội",
    "khó chịu",
    "thất vọng",
    "chán",
    "kém",
    "nhạt",
    "mặn",
    "hôi",
    "ôi",
    "thiu",
    "khô",
    "cứng",
    "tanh",
    "sống",
    "cháy",
    "lạnh",
    "mắc",
    "cau có",
    "bất lịch sự",
    "vô duyên",
    "ồn",
    "ồn ào",
    "chật",
    "tồi",
    "tồi tệ",
    "khó ăn",
    "phí tiền",
)
NEGATORS = ("không", "chưa", "chẳng", "chả", "đừng", "không hề")
INTENSIFIERS = ("rất", "quá", "cực", "siêu", "cực kỳ", "vô cùng")
BOUNDARIES = {".", ",", ";", "!", "?", ":", "nhưng", "song", "tuy", "mà"}


def rule_definition():
    return {
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "negators": NEGATORS,
        "intensifiers": INTENSIFIERS,
        "negation_window_tokens": 3,
        "intensifier_multiplier": 1.5,
        "threshold": 0,
        "boundaries": sorted(BOUNDARIES),
        "features": ["text"],
        "scope": "simple local negation only; no sarcasm/idiom parser",
        "provenance": "manual domain lexicon; no test-driven tuning",
    }


class VietnameseRuleBased:
    def __init__(self):
        self.lexicon = {
            tuple(term.split()): score
            for terms, score in ((POSITIVE, 1), (NEGATIVE, -1))
            for term in terms
        }
        self.maximum = max(map(len, self.lexicon))

    def score(self, text):
        tokens = re.findall(r"\w+|[.,;!?:]", normalize_text(text).casefold(), flags=re.UNICODE)
        score, index = 0.0, 0
        while index < len(tokens):
            match = next(
                (
                    tuple(tokens[index : index + size])
                    for size in range(self.maximum, 0, -1)
                    if tuple(tokens[index : index + size]) in self.lexicon
                ),
                None,
            )
            if match is None:
                index += 1
                continue
            window = tokens[max(0, index - 3) : index]
            boundaries = [offset for offset, token in enumerate(window) if token in BOUNDARIES]
            if boundaries:
                window = window[boundaries[-1] + 1 :]
            # Token negators count once (không hề is the same không scope).
            value = self.lexicon[match] * (
                -1
                if sum(token in {"không", "chưa", "chẳng", "chả", "đừng"} for token in window) % 2
                else 1
            )
            if any(
                tuple(window[-len(term.split()) :]) == tuple(term.split()) for term in INTENSIFIERS
            ):
                value *= 1.5
            score += value
            index += len(match)
        return score

    def predict(self, texts):
        scores = [self.score(text) for text in texts]
        return [
            "positive" if score > 0 else "negative" if score < 0 else "neutral" for score in scores
        ]

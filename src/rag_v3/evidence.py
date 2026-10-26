"""Deterministic evidence selection precedes every V3 generation call."""

import json
import re
from dataclasses import dataclass

from src.rag_v2.intent import ASPECT_TERMS, NEGATIVE, POSITIVE, evidence_compatible, issue_matches
from src.rag_v2.service import support_gate


@dataclass(frozen=True)
class Packet:
    question_intent: str
    aspect: str | None
    restaurant: str
    evidence: tuple[dict, ...]
    aggregates: dict
    support_gate_status: str

    def generation_payload(self, question):
        # The model cites stable E-refs. Opaque review IDs stay in this packet
        # for validation and the UI, but are not needed in the LLM prompt.
        generation_evidence = [
            {key: value for key, value in item.items() if key != "review_id"}
            for item in self.evidence
        ]
        return {"question": question, "question_intent": self.question_intent,
                "restaurant": self.restaurant, "evidence": generation_evidence,
                "aggregates": self.aggregates}


def _metadata_list(value):
    return json.loads(value) if isinstance(value, str) else list(value or [])


def _packet_text(text, intent):
    if intent == "complaint":
        match = re.search(
            r"\bdở\b|\btệ\b|thất vọng|chờ lâu|thái độ|\bnguội\b|"
            r"\bđắt\b|\bmặn\b|không hợp|không hỗ trợ|không kèm|không cho|"
            r"\bthiếu\b|đi lại thì không|hơi kì",
            text, flags=re.IGNORECASE,
        )
        if match and match.start() >= 170:
            boundary = text.rfind(". ", max(0, match.start() - 150), match.start())
            start = boundary + 2 if boundary >= 0 else max(0, match.start() - 35)
            return text[start:start + 350].strip()
    return text[:350]


def build_packet(request, hits, route, *, max_evidence=3):
    """Select only compatible, deduplicated reviews and expose no frequency claim."""
    if route.category == "TREND_CAUSATION":
        selected = []
    elif route.intent in ("complaint", "praise"):
        selected = support_gate(hits, route.intent, route.aspect, request.query)
        if route.intent == "praise":
            # Mixed reviews can contain a local complaint despite a positive
            # overall prediction; do not turn that complaint into a strength.
            selected = [hit for hit in selected if not any(
                phrase in hit.text.casefold() for phrase in
                ("bị nguội", "không ngon", "hơi mặn", "hơi nhạt", "phục vụ tệ",
                 "thái độ tệ", "chờ lâu", "không hài lòng")
            )]
    elif route.aspect:
        selected = [hit for hit in hits if any(
            term in hit.text.casefold() for term in ASPECT_TERMS[route.aspect]
        )]
    else:
        selected = hits
    unique = {}
    for hit in selected:
        normalized = re.sub(r"\s+", " ", hit.text.casefold()).strip()
        unique.setdefault(normalized, hit)
    selected = list(unique.values())[:max_evidence]
    packet_evidence = tuple({
        "ref": f"E{index}", "review_id": hit.review_id,
        "sentiment": hit.metadata.get("predicted_sentiment"),
        "aspects": _metadata_list(hit.metadata.get("predicted_aspects")),
        "issues": _metadata_list(hit.metadata.get("detected_issues")),
        "rating": hit.metadata.get("raw_rating"),
        "date": hit.metadata.get("review_timestamp"),
        "source": hit.metadata.get("source"),
        "text": _packet_text(hit.text, route.intent),
    } for index, hit in enumerate(selected, 1))
    sentiments = {item["sentiment"] for item in packet_evidence}
    aggregates = {"accepted_review_count": len(packet_evidence),
                  "mixed_sentiment": {"positive", "negative"} <= sentiments,
                  "frequency_supported": False}
    return Packet(route.category, route.aspect,
                  selected[0].restaurant_name if selected else "",
                  packet_evidence, aggregates,
                  "ACCEPTED" if packet_evidence else "INSUFFICIENT_EVIDENCE")


def evidence_for_refs(packet, refs, hits):
    mapped = {item["ref"]: item["review_id"] for item in packet.evidence}
    by_id = {hit.review_id: hit for hit in hits}
    if not refs or len(refs) != len(set(refs)) or any(ref not in mapped for ref in refs):
        raise ValueError("Citation placeholder is missing, repeated, or unknown")
    return [by_id[mapped[ref]] for ref in refs]


def validate_packet_citations(answer, refs, packet, hits, route, question):
    """Map short placeholders to exact IDs, then recheck topic and polarity."""
    inline = re.findall(r"\[(E\d+)\]", answer)
    if not inline or set(inline) != set(refs):
        raise ValueError("Answer and citation list disagree")
    cited = evidence_for_refs(packet, list(dict.fromkeys(inline)), hits)
    answer_aspect_terms = ASPECT_TERMS[route.aspect] if route.aspect else ()
    if route.aspect == "food":
        answer_aspect_terms += ("phần ăn", "khẩu phần")
    if route.aspect and not any(term in answer.casefold()
                                for term in answer_aspect_terms):
        raise ValueError("Answer does not address requested aspect")
    if route.intent == "praise":
        if (not any(term in answer.casefold() for term in POSITIVE)
                or any(term in answer.casefold() for term in
                       (*NEGATIVE, "nguội", "mặn", "nhạt"))):
            raise ValueError("Praise answer contains complaint or lacks praise")
    if route.intent == "complaint" and not any(
        term in answer.casefold() for term in (*NEGATIVE, "cần phải xem lại",
                                              "không hỗ trợ", "không kèm")
    ):
        raise ValueError("Complaint answer lacks complaint wording")
    if route.aspect == "food" and any(term in answer.casefold() for term in (
        "nhân viên", "phục vụ", "không gian", "chỗ ngồi",
    )):
        raise ValueError("Answer adds off-topic restaurant facts")
    # "phần ăn nhiều" describes a portion, not the prevalence of an opinion.
    frequency_claim = re.search(
        r"đa số|phần lớn|thường xuyên|phổ biến|chủ yếu|nhiều nhất|"
        r"\bnhiều\s+(?:khách|người|review|đánh giá|phản hồi)",
        answer, flags=re.IGNORECASE,
    )
    if frequency_claim and not packet.aggregates["frequency_supported"]:
        raise ValueError("Unsupported frequency claim")
    if re.search(r"nguyên nhân|dẫn đến|khiến|\bvì\b.{0,60}\bnên\b", answer,
                 flags=re.IGNORECASE):
        raise ValueError("Unsupported causal claim")
    if packet.aggregates["mixed_sentiment"] and route.intent == "general" and not re.search(
        r"trái chiều|khác nhau|có người khen.*có người chê|vừa khen.*vừa chê|"
        r"tuy nhiên|trong khi",
        answer, flags=re.IGNORECASE,
    ):
        raise ValueError("Mixed evidence requires a mixed answer")
    for hit in cited:
        if route.intent in ("complaint", "praise") and not evidence_compatible(
            hit, route.intent, route.aspect,
        ):
            raise ValueError("Cited review contradicts question intent")
        if route.aspect and not any(term in hit.text.casefold()
                                    for term in ASPECT_TERMS[route.aspect]):
            raise ValueError("Cited review misses requested aspect")
        if issue_matches(hit.text, question) is False:
            raise ValueError("Cited review misses requested issue")
    ignored = {"khách", "đánh", "giá", "quán", "một", "các", "về", "được",
               "trong", "những", "điều", "này", "rất", "cho", "với", "nhân", "viên",
               "nói", "rằng", "của", "theo", "còn", "tuy", "nhiên", "số",
               "review", "truy", "xuất", "hiện", "bằng", "chứng", "liên",
               "quan", "thấy", "nhận", "xét"}
    ignored.update({"đồ", "ăn", "món", "không", "và", "khi", "nên", "đều",
                    "chất", "lượng", "cần", "phải"})
    by_ref = {item["ref"]: item for item in packet.evidence}
    # A trailing citation applies to every clause in that sentence. A single
    # cited review must therefore support every independent factual assertion.
    for sentence in re.split(r"(?<=[.!?])\s+|;", answer):
        sentence_refs = re.findall(r"\[(E\d+)\]", sentence)
        if not sentence_refs:
            if re.search(r"[^\W_]{3,}", sentence):
                raise ValueError("Answer clause lacks its own citation")
            continue
        if (re.search(r"một số khách|một số (?:review|đánh giá|phản hồi)",
                      sentence, flags=re.IGNORECASE)
                and len(set(sentence_refs)) < 2):
            raise ValueError("Plural-review claim has only one cited review")
        for clause in re.split(r",|\bnhưng\b|\btuy nhiên\b|\bvà\b", sentence):
            clean = re.sub(r"\[E\d+\]", "", clause).strip(" .!?")
            terms = set(re.findall(r"[^\W_]{3,}", clean.casefold())) - ignored
            if len(terms) < 2:
                continue
            local_refs = re.findall(r"\[(E\d+)\]", clause) or sentence_refs
            if not any(_clause_supported(clean, by_ref[ref], terms, ignored)
                       for ref in local_refs):
                raise ValueError("Answer clause lacks cited support")
    return cited


_ISSUE_CUES = {
    "waiting": ("chậm", "chờ lâu", "đợi lâu", "đợi", "lâu"),
    "attitude": ("thái độ", "hỗn", "nạt", "khinh", "khó chịu", "thân thiện"),
    "price_high": ("giá cao", "đắt", "mắc", "chặt chém"),
    "price_low": ("giá rẻ", "phải chăng", "hợp lý", "đáng tiền"),
    "food_cold": ("nguội", "lạnh"),
    "food_salty": ("mặn",),
    "food_good": ("ngon", "vừa miệng", "dễ ăn"),
    "order_missing": ("không kèm", "không cho", "thiếu", "không có"),
}


def _clause_supported(clause, item, terms, ignored):
    claim = clause.casefold()
    source = item["text"].casefold()
    # Numeric and frequency details are especially easy to misattribute. Keep
    # such generated claims out of accepted synthesis; the source is still
    # available verbatim through the extractive fallback.
    if re.search(r"\d", claim):
        return False
    if (re.search(r"nhân viên.{0,30}(?:chờ|đợi)", claim)
            and not re.search(r"nhân viên.{0,30}(?:chờ|đợi)", source)):
        return False
    source_terms = set(re.findall(r"[^\W_]{3,}", source)) - ignored
    overlap = len(terms & source_terms)
    # A bounded paraphrase of an explicit taste statement: "nêm nếm rất vừa"
    # can support "vừa miệng". Keep all other claims on the lexical gate.
    taste_paraphrase = ("vừa miệng" in claim
                        and "nêm nếm rất vừa" in source and overlap >= 1)
    if overlap < min(2, len(terms)) and not taste_paraphrase:
        return False
    for issue, cues in _ISSUE_CUES.items():
        if any(cue in claim for cue in cues):
            if not any(cue in source for cue in cues):
                return False
            if issue == "attitude":
                # Positive staff praise cannot substantiate negative attitude
                # or vice versa, even when the word 'thái độ' overlaps.
                negative = ("tệ", "kém", "không tốt", "chưa tốt", "hỗn",
                            "khinh", "nạt", "khó chịu")
                positive = ("tốt", "thân thiện", "nhiệt tình", "lịch sự")
                if any(word in claim for word in negative) and not any(
                    word in source for word in negative
                ):
                    return False
                if any(word in claim for word in positive) and not any(
                    word in source for word in positive
                ):
                    return False
    if any(cue in claim for cue in _ISSUE_CUES["price_high"]) and not any(
        cue in source for cue in _ISSUE_CUES["price_high"]
    ):
        return False
    return True


def extractive_answer(packet):
    if not packet.evidence:
        raise ValueError("Extractive answer requires accepted evidence")
    item = packet.evidence[0]
    excerpt = item["text"].strip()
    if len(excerpt) > 180:
        excerpt = excerpt[:180].rsplit(" ", 1)[0] + "…"
    return f"Một đánh giá viết: “{excerpt}” [{item['ref']}].", [item["ref"]]

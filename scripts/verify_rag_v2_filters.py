"""Check V2 metadata filtering against the full persistent indices."""

import json
from datetime import datetime
from pathlib import Path

from src.foody.schema import save_json
from src.rag.schemas import RetrievalRequest
from src.rag.service import resolve_restaurant
from src.rag_v2.cli import selected_v2_retriever


def main():
    key, _ = resolve_restaurant("Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu")
    retriever = selected_v2_retriever()
    checks = {}
    cases = {
        "restaurant": RetrievalRequest(query="món ăn", restaurant_key=key, top_k=5),
        "sentiment": RetrievalRequest(query="phục vụ chậm", restaurant_key=key,
                                      sentiment="negative", top_k=5),
        "aspect": RetrievalRequest(query="phục vụ chậm", restaurant_key=key,
                                   aspect="service", top_k=5),
        "date": RetrievalRequest(query="món ăn", restaurant_key=key,
                                 date_from="2015-01-01", date_to="2030-12-31", top_k=5),
    }
    for name, request in cases.items():
        hits, _, debug = retriever.search_debug(request)
        checks[name] = {"returned": len(hits), "review_ids": [hit.review_id for hit in hits],
                        "effective_filters": debug["effective_filters"],
                        "all_match": all(
                            hit.metadata["restaurant_key"] == key
                            and (request.sentiment is None or
                                 hit.metadata["predicted_sentiment"] == request.sentiment)
                            and (request.aspect is None or request.aspect in json.loads(
                                hit.metadata["predicted_aspects"]))
                            and (request.date_from is None or
                                 request.date_from <= datetime.fromisoformat(
                                     hit.metadata["review_timestamp"])
                                 <= request.date_to)
                            for hit in hits
                        )}
    result = {"checks": checks, "pass": all(
        item["all_match"] and item["returned"] > 0 for item in checks.values()
    )}
    Path("reports/rag_v2").mkdir(parents=True, exist_ok=True)
    save_json("reports/rag_v2/filter_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit("RAG V2 filter verification failed")


if __name__ == "__main__":
    main()

"""Verify persisted Foody metadata filters against returned reviews."""

import json
from pathlib import Path

from src.foody.schema import restaurant_key, save_json
from src.rag.cli import selected_retriever
from src.rag.schemas import RetrievalRequest
from src.rag.vector_store import epoch


def main():
    key = restaurant_key("Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu")
    retriever = selected_retriever()
    cases = {
        "restaurant_only": {"restaurant_key": key},
        "restaurant_service": {"restaurant_key": key, "aspect": "service"},
        "restaurant_negative": {"restaurant_key": key, "sentiment": "negative"},
        "restaurant_date_range": {
            "restaurant_key": key,
            "date_from": "2021-01-01",
            "date_to": "2021-12-31",
        },
    }
    results = {}
    for name, filters in cases.items():
        request = RetrievalRequest(query="chất lượng món ăn và phục vụ", top_k=5, **filters)
        hits, latency_ms = retriever.search(request)
        checks = []
        for hit in hits:
            meta = hit.metadata
            ok = meta["restaurant_key"] == key
            if "aspect" in filters:
                ok &= filters["aspect"] in json.loads(meta["predicted_aspects"])
            if "sentiment" in filters:
                ok &= meta["predicted_sentiment"] == filters["sentiment"]
            if "date_from" in filters:
                ok &= epoch(filters["date_from"]) <= meta["timestamp_epoch"]
                ok &= meta["timestamp_epoch"] < epoch("2022-01-01")
            checks.append(bool(ok))
        results[name] = {
            "filters": filters,
            "result_count": len(hits),
            "all_results_match": bool(hits) and all(checks),
            "review_ids": [hit.review_id for hit in hits],
            "retrieval_latency_ms": latency_ms,
        }
    output = Path("reports/rag/filter_verification.json")
    save_json(output, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(result["all_results_match"] for result in results.values()):
        raise SystemExit("At least one metadata filter failed")


if __name__ == "__main__":
    main()

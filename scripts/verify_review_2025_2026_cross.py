"""Live Sales/Review ID and period alignment checks before corpus cutover."""

import json
import os

import pandas as pd

from src.foody.review_migration import DATA_VERSION, ENRICHED
from src.sales.service import cross_answer
from src.warehouse.loader import connection

RESTAURANTS = ("10", "14", "83")
PERIODS = ("2025-05", "2026-05")


def main():
    if os.getenv("FOODY_REVIEW_CORPUS") != DATA_VERSION:
        raise ValueError("Explicit side-by-side corpus selection is required")
    frame = pd.read_parquet(ENRICHED, columns=["review_id", "restaurant_id", "review_time"])
    owners = frame.set_index("review_id")[["restaurant_id", "review_time"]].to_dict("index")
    with connection() as conn:
        for restaurant_id in RESTAURANTS:
            for period in PERIODS:
                question = (f"Doanh thu tháng {period[5:7]}/{period[:4]} thay đổi thế nào "
                            "và review cùng kỳ phản ánh điều gì?")
                result = cross_answer(conn, restaurant_id, question)
                if result["sql_facts"] is None:
                    raise ValueError("Expected observed SQL sales facts")
                for item in result["review_evidence"]:
                    row = owners[item["review_id"]]
                    if str(row["restaurant_id"]) != restaurant_id or \
                            row["review_time"].strftime("%Y-%m") != period:
                        raise ValueError("Cross answer cited wrong restaurant or period")
                if result["review_evidence"] and "không chứng minh quan hệ nhân quả" not in result["answer"]:
                    raise ValueError("Cross answer omitted the noncausal limitation")
                print(json.dumps({"restaurant_id": restaurant_id, "period": period,
                                  "status": result["status"],
                                  "aligned_reviews": result.get("aligned_review_count"),
                                  "cited_reviews": len(result["review_evidence"])},
                                 ensure_ascii=True), flush=True)
        result = cross_answer(conn, "1072",
                              "Doanh thu tháng 9/2026 thay đổi thế nào và review cùng kỳ phản ánh điều gì?")
        if result["status"] != "no_aligned_reviews" or result["review_evidence"]:
            raise ValueError("Out-of-range September 2026 reviews were substituted")
        print(json.dumps({"restaurant_id": "1072", "period": "2026-09",
                          "status": result["status"], "sales_facts": bool(result["sql_facts"]),
                          "cited_reviews": 0}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

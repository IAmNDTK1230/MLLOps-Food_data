"""Live ID-isolated Review RAG smoke against three new-corpus restaurants."""

import json
import os

from src.foody.review_migration import DATA_VERSION
from src.rag.schemas import RetrievalRequest
from src.rag_v3.cli import selected_v3_gateway
from src.rag_v3.retriever import RoutedRetriever
from src.rag_v3.service import answer_question_v3

CASES = {
    "14": ("Khách đánh giá quán thế nào?", "Món nào được khen?",
           "Khách nói gì về phục vụ?"),
    "32": ("Khách đánh giá quán thế nào?", "Đồ uống nào được khen?",
           "Khách nói gì về phục vụ?"),
    "83": ("Khách đánh giá quán thế nào?", "Món nào được khen?",
           "Khách nói gì về nhân viên?"),
}


def main():
    if os.getenv("FOODY_REVIEW_CORPUS") != DATA_VERSION:
        raise ValueError("Explicit side-by-side corpus selection is required")
    retriever = RoutedRetriever()
    with selected_v3_gateway() as gateway:
        for restaurant_id, questions in CASES.items():
            for question in questions:
                answer, hits, telemetry, _, packet = answer_question_v3(
                    retriever, gateway, RetrievalRequest(
                        query=question, restaurant_id=restaurant_id, top_k=5))
                if any(hit.metadata.get("restaurant_id") != restaurant_id for hit in hits):
                    raise ValueError("Retrieved a review from another restaurant ID")
                by_id = {hit.review_id: hit for hit in hits}
                if any(by_id[item.review_id].metadata.get("restaurant_id") != restaurant_id
                       for item in answer.evidence):
                    raise ValueError("Cited a review from another restaurant ID")
                if any(item["review_id"] not in by_id for item in packet.evidence):
                    raise ValueError("Evidence packet contains an unknown review")
                print(json.dumps({"restaurant_id": restaurant_id, "question": question,
                                  "answer": answer.answer,
                                  "mode": answer.generation_mode,
                                  "status": telemetry["status"],
                                  "cited_count": len(answer.evidence)}, ensure_ascii=True),
                      flush=True)


if __name__ == "__main__":
    main()

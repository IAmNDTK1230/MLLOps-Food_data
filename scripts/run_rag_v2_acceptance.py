"""Run the same five real questions through frozen V1 and selected V2."""

from pathlib import Path

from src.foody.schema import save_json
from src.llm.gateway import LLMGateway
from src.rag.cli import selected_retriever
from src.rag.schemas import RetrievalRequest
from src.rag.service import answer_question, resolve_restaurant
from src.rag_v2.cli import selected_v2_retriever
from src.rag_v2.service import answer_question_v2

RESTAURANT = "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu"
QUESTIONS = {
    "complaints": "Khách hàng phàn nàn điều gì nhiều nhất?",
    "food": "Khách nói gì về chất lượng món ăn?",
    "service": "Nhân viên phục vụ có vấn đề gì không?",
    "price": "Khách có phàn nàn về giá không?",
    "rating_drop": "Tại sao đánh giá của nhà hàng này giảm?",
}


def main():
    root = Path("reports/rag_v2/examples")
    root.mkdir(parents=True, exist_ok=True)
    key, _ = resolve_restaurant(RESTAURANT)
    v1 = selected_retriever()
    v2 = selected_v2_retriever()
    with LLMGateway() as gateway:
        for name, question in QUESTIONS.items():
            request = RetrievalRequest(query=question, restaurant_key=key, top_k=5)
            old, old_hits, old_telemetry = answer_question(v1, gateway, request)
            new, new_hits, new_telemetry, debug = answer_question_v2(v2, gateway, request)
            result = {"restaurant": RESTAURANT, "question": question,
                      "v1": {"response": old.model_dump(),
                             "retrieved_reviews": [hit.model_dump() for hit in old_hits],
                             "telemetry": old_telemetry},
                      "v2": {"response": new.model_dump(),
                             "retrieved_reviews": [hit.model_dump() for hit in new_hits],
                             "telemetry": new_telemetry, "retrieval_debug": debug}}
            save_json(root / f"{name}_comparison.json", result)
            print(name, "V1", old_telemetry["status"], "V2", new_telemetry["status"])


if __name__ == "__main__":
    main()

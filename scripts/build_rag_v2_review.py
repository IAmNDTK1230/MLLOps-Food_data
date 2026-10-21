"""Save inspectable review text for every frozen benchmark query."""

from src.foody.schema import save_json
from src.rag.cli import selected_retriever
from src.rag.evaluation import development
from src.rag.schemas import RetrievalRequest
from src.rag_v2.hybrid import HYBRID_EVIDENCE, HybridRetriever
from src.rag_v2.lexical import BM25Retriever

REVIEWER_NOTES = {
    "khách phàn nàn về phục vụ chậm": "Both systems return service complaints; V2 prioritizes waiting complaints and avoids one near-duplicate V1 pair. Some V2 hits discuss staff attitude rather than delay.",
    "đồ ăn không ngon": "V2's first four results describe disappointing food; V1 includes an unrelated service complaint in its top five. Predicted sentiment is wrong for one relevant V1 review.",
    "giá quá cao": "Both top lists contain explicit high-price complaints; V2 places an exact high-price review first.",
    "không gian ồn": "V2 includes a cramped-space and other ambiance review without a clear noise complaint. This query remains a semantic weakness despite negative/ambiance metadata filters.",
    "nhân viên thái độ kém": "Both top lists contain explicit staff-attitude complaints. One relevant V1 review is incorrectly tagged positive; V2's strict metadata filter drops it.",
    "món ăn ngon": "Both top lists contain clear food praise. V2 first result has more descriptive food evidence; short generic praise remains common.",
}


def describe(hits):
    return [{"rank": rank, "review_id": hit.review_id, "text": hit.text,
             "score": hit.score,
             "predicted_sentiment": hit.metadata["predicted_sentiment"],
             "predicted_aspects": hit.metadata["predicted_aspects"],
             "restaurant_name": hit.restaurant_name}
            for rank, hit in enumerate(hits, 1)]


def main():
    dense = selected_retriever()
    dense.monitor = False
    hybrid = HybridRetriever(dense, BM25Retriever(), HYBRID_EVIDENCE)
    rows = []
    for case in development()["cases"]:
        request = RetrievalRequest(query=case["query"], top_k=10)
        baseline, _ = dense.search(request)
        selected, _, debug = hybrid.search_debug(request)
        rows.append({"query": case["query"],
                     "known_relevant_review_ids": case["known_relevant_review_ids"],
                     "manual_observation": REVIEWER_NOTES[case["query"]],
                     "retrieval_debug": debug,
                     "v1_dense_top_10": describe(baseline),
                     "v2_hybrid_top_10": describe(selected)})
    save_json("reports/rag_v2/manual_retrieval_review.json", {
        "scope": "Six frozen manually known-relevant queries; labels incomplete; text included for human review",
        "selected_candidate": HYBRID_EVIDENCE.name, "cases": rows,
    })
    print("Saved six V1/V2 top-10 review lists for manual semantic inspection")


if __name__ == "__main__":
    main()

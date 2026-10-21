"""Run the four Streamlit pages and one real RAG interaction with AppTest."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"),
                            default_timeout=120).run()
    checks = {}
    for page in ("Restaurant Intelligence", "Review Explorer",
                 "AI Review Analyst", "LLMOps Monitor"):
        app.sidebar.radio[0].set_value(page).run()
        checks[page] = {"exception_count": len(app.exception),
                        "titles": [item.value for item in app.title]}
    app.sidebar.radio[0].set_value("AI Review Analyst").run()
    app.selectbox[0].set_value("Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu").run()
    app.button[0].click().run()
    rendered = [str(item.value) for item in app.markdown]
    answer_text = [value for value in rendered if "[R1]" in value
                   or "Không đủ bằng chứng" in value]
    checks["AI Review Analyst query"] = {
        "exception_count": len(app.exception),
        "warnings": [item.message for item in app.warning],
        "evidence_visible": any("Evidence IDs" in value for value in rendered),
        "review_metadata_visible": any("predicted sentiment" in str(item.value)
                                       for item in app.caption),
        "answer_visible": bool(answer_text),
    }
    path = Path("reports/rag_v2/streamlit_smoke.json")
    path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if any(item["exception_count"] for item in checks.values()):
        raise SystemExit("Streamlit page error")
    if checks["AI Review Analyst query"]["warnings"] or not all(
        checks["AI Review Analyst query"][field]
        for field in ("evidence_visible", "review_metadata_visible", "answer_visible")
    ):
        raise SystemExit("RAG V2 interaction did not render answer and review evidence")


if __name__ == "__main__":
    main()

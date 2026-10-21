"""Exercise four Streamlit pages and a real V3 answer with mode disclosure."""

import json
import re
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
    app.sidebar.radio[0].set_value("Restaurant Intelligence").run()
    app.button[0].click().run()
    checks["Restaurant Intelligence health"] = {
        "exception_count": len(app.exception),
        "warnings": [item.message for item in app.warning],
        "v3_mode_visible": any("RAG V3 section modes" in str(item.value)
                               for item in app.caption),
    }
    app.sidebar.radio[0].set_value("AI Review Analyst").run()
    app.selectbox[0].set_value("Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu").run()
    app.button[0].click().run()
    rendered = [str(item.value) for item in app.markdown]
    captions = [str(item.value) for item in app.caption]
    checks["AI Review Analyst query"] = {
        "exception_count": len(app.exception),
        "warnings": [item.message for item in app.warning],
        "answer_visible": any(re.search(r"\[E\d+\]", item) or "Không đủ bằng chứng" in item
                              for item in rendered),
        "evidence_visible": any("Evidence IDs" in item for item in rendered),
        "review_metadata_visible": any("predicted sentiment" in item for item in captions),
        "generation_mode_visible": any("Generation mode:" in item for item in captions),
    }
    path = Path("reports/rag_v3/streamlit_smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if any(item["exception_count"] for item in checks.values()):
        raise SystemExit("Streamlit page error")
    if checks["AI Review Analyst query"]["warnings"] or not all(
        checks["AI Review Analyst query"][field] for field in
        ("answer_visible", "evidence_visible", "review_metadata_visible",
         "generation_mode_visible")
    ):
        raise SystemExit("RAG V3 interaction lacks answer, evidence, or mode disclosure")
    if (checks["Restaurant Intelligence health"]["warnings"]
            or not checks["Restaurant Intelligence health"]["v3_mode_visible"]):
        raise SystemExit("RAG V3 health action did not show section modes")


if __name__ == "__main__":
    main()

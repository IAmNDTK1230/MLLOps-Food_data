"""Save a traceable V3 health report using the selected generator."""

from pathlib import Path

from src.foody.schema import save_json
from src.rag_v3.cli import selected_v3_gateway
from src.rag_v3.health import health_report_v3
from src.rag_v3.retriever import RoutedRetriever


def main():
    retriever = RoutedRetriever()
    with selected_v3_gateway() as gateway:
        report, telemetry, verification = health_report_v3(
            retriever, gateway, "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu",
        )
    path = Path("reports/rag_v3/health_report.json")
    save_json(path, {"report": report.model_dump(), "telemetry": telemetry,
                     "verification": verification})
    print(path)


if __name__ == "__main__":
    main()

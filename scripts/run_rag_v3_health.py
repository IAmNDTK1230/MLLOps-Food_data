"""Run one traceable V3 health report for the frozen Hanuri acceptance case."""

from src.foody.schema import save_json
from src.rag_v3.cli import selected_v3_gateway
from src.rag_v3.health import health_report_v3
from src.rag_v3.retriever import RoutedRetriever


def main():
    with selected_v3_gateway() as gateway:
        report, telemetry, verification = health_report_v3(
            RoutedRetriever(), gateway,
            "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu",
        )
    result = {"report": report.model_dump(), "telemetry": telemetry,
              "verification": verification}
    save_json("reports/rag_v3/health_report.json", result)
    print(telemetry["status"], telemetry["section_modes"])


if __name__ == "__main__":
    main()

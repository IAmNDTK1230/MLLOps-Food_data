from datetime import datetime
from typing import Any


def execution_report(summary: dict[str, Any]) -> str:
    stages = summary.get("stages", {})
    uploads = stages.get("upload_to_databricks", {}).get("files")
    duration = "unknown"
    if summary.get("crawl_finished_at") and summary.get("crawl_started_at"):
        duration = f"{(datetime.fromisoformat(summary['crawl_finished_at']) - datetime.fromisoformat(summary['crawl_started_at'])).total_seconds():.2f}s"
    lines = [
        f"Run {summary['run_id']}: {summary['status']}",
        f"Keyword: {summary.get('keyword', 'unknown')}; crawler: {stages.get('crawl_products', {}).get('status', 'SKIPPED')}",
        f"Products valid: {summary['products_collected']}; reviews valid: {summary['reviews_collected']}; rejected: {summary['rejected_records']}",
        f"Duration: {duration}",
        "Uploads: "
        + (
            f"{sum(u['status'] == 'UPLOADED' for u in uploads)} uploaded; {sum(u['status'] == 'SKIPPED' for u in uploads)} skipped"
            if uploads is not None
            else "SKIPPED — upload stage not executed"
        ),
        "Bronze / Silver / Gold counts: NOT VERIFIED — no cloud SQL verification executed",
        "Raw: "
        + (", ".join(p for p in summary.get("files", []) if p.endswith(".jsonl")) or "none"),
        "Parquet: "
        + (", ".join(p for p in summary.get("files", []) if p.endswith(".parquet")) or "none"),
        "Metadata: " + (summary.get("metadata_file") or "unknown"),
    ]
    lines.extend(f"Error: {error}" for error in summary.get("errors", []))
    return "\n".join(lines)

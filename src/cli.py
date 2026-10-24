import argparse
import json
import sys
from pathlib import Path

from src.config import Settings
from src.orchestration.fixture_policy import validate_fixture_mode


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Polite public Shopee data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("crawl", "pipeline"):
        sub = commands.add_parser(command)
        sub.add_argument("--keyword", required=True)
        sub.add_argument("--pages", type=int, default=2)
        sub.add_argument("--skip-reviews", action="store_true")
        sub.add_argument("--resume", action="store_true")
        sub.add_argument(
            "--fixture", help="Synthetic product JSONL fixture (local-only by default)"
        )
        if command == "pipeline":
            sub.add_argument(
                "--allow-synthetic-cloud-test",
                action="store_true",
                help="Permit --fixture cloud execution only in catalog shopee_test",
            )
        group = sub.add_mutually_exclusive_group()
        group.add_argument("--crawl-only", action="store_true")
        group.add_argument("--upload-only", action="store_true")
        sub.add_argument("--skip-databricks-job", action="store_true")
        sub.add_argument("--json", action="store_true", help="Machine-readable summary")
    check = commands.add_parser(
        "databricks-check",
        help="Check SDK unified authentication, Unity Catalog and a unique volume probe",
    )
    check.add_argument("--json", action="store_true")
    bootstrap = commands.add_parser(
        "databricks-bootstrap", help="Generate SQL using configured Unity Catalog names"
    )
    bootstrap.add_argument("--output", default="scripts/databricks_bootstrap.sql")
    commands.add_parser("demo-preflight", help="Read-only frozen demo artifact checks")
    commands.add_parser("warehouse-load", help="Idempotently load frozen Foody marts")
    commands.add_parser("sales-load", help="Audit and idempotently load Foody order CSV")
    sales_query = commands.add_parser("sales-query", help="Ask a bounded SQL/cross sales question")
    sales_query.add_argument("--restaurant-id", required=True)
    sales_query.add_argument("--question", required=True)
    replay = commands.add_parser("kafka-produce", help="Replay a bounded Foody event sample")
    replay.add_argument("--limit", type=int, default=20)
    commands.add_parser("serve", help="Serve the configurable Prefect deployment")
    from src.ml.cli import add_commands, run_command

    ml_commands = add_commands(commands)
    from src.foody import pipeline as foody_pipeline

    foody_pipeline.add_commands(commands)
    from src.rag import cli as rag_cli
    from src.rag_v2 import cli as rag_v2_cli
    from src.rag_v3 import cli as rag_v3_cli

    rag_cli.add_commands(commands)
    rag_v2_cli.add_commands(commands)
    rag_v3_cli.add_commands(commands)
    args = parser.parse_args()
    if args.command == "sales-load":
        from src.sales.loader import main as sales_main

        return sales_main()
    if args.command == "sales-query":
        from src.sales.service import answer_question
        from src.warehouse.loader import connection

        with connection() as conn:
            result = answer_question(conn, args.restaurant_id, args.question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "warehouse-load":
        from src.warehouse.loader import main as warehouse_main

        return warehouse_main()
    if args.command == "kafka-produce":
        from src.streaming.producer import main as producer_main

        return producer_main(args.limit)
    if args.command == "demo-preflight":
        from src.demo import demo_preflight, print_preflight

        result = demo_preflight()
        print_preflight(result)
        return 0 if result["status"] == "PASS" else 1
    if args.command in rag_v3_cli.COMMANDS:
        return rag_v3_cli.execute(args)
    if args.command in rag_v2_cli.COMMANDS:
        return rag_v2_cli.execute(args)
    if args.command in rag_cli.COMMANDS:
        return rag_cli.execute(args)
    if args.command in foody_pipeline.COMMANDS:
        return foody_pipeline.run_command(args)
    if args.command in ml_commands:
        return run_command(args)
    settings = Settings.from_env()
    if args.command == "databricks-bootstrap":
        from src.databricks.preflight import bootstrap_sql

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(bootstrap_sql(settings), encoding="utf-8")
        print(f"Generated {path}")
        return 0
    if args.command == "databricks-check":
        from src.databricks.preflight import check_databricks

        result = check_databricks(settings)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result["status"] + (" — " + result["reason"] if "reason" in result else ""))
            for check in result["checks"]:
                print(
                    f"{check['name']}: {check['status']}"
                    + (f" ({check['error_type']})" if "error_type" in check else "")
                )
        return 1 if result["status"] == "FAILED" else 0
    if args.command != "serve":
        if args.pages < 1:
            parser.error("--pages must be positive")
        local = args.command == "crawl" or args.crawl_only
        if args.command == "crawl" and args.upload_only:
            parser.error("crawl cannot use --upload-only")
        try:
            validate_fixture_mode(
                settings,
                args.fixture,
                crawl_only=local,
                upload_only=args.upload_only,
                allow_synthetic_cloud_test=getattr(args, "allow_synthetic_cloud_test", False),
            )
        except ValueError as error:
            parser.error(str(error))
    from src.orchestration.flow import PipelineFailed, shopee_data_pipeline

    if args.command == "serve":
        from prefect.schedules import Cron

        shopee_data_pipeline.serve(
            name="shopee-daily",
            schedule=Cron(settings.crawl_cron, timezone=settings.crawl_timezone),
            parameters={"keyword": settings.crawl_keyword, "pages": settings.crawl_pages},
            limit=1,
        )
        return 0
    try:
        summary = shopee_data_pipeline(
            keyword=args.keyword,
            pages=args.pages,
            skip_reviews=args.skip_reviews,
            crawl_only=local,
            upload_only=args.upload_only,
            skip_databricks_job=args.skip_databricks_job,
            resume=args.resume,
            settings=settings,
            fixture=args.fixture,
            allow_synthetic_cloud_test=getattr(args, "allow_synthetic_cloud_test", False),
        )
    except PipelineFailed as error:
        summary = error.summary
    from src.orchestration.report import execution_report

    print(
        json.dumps(summary, ensure_ascii=False, indent=2)
        if args.json
        else execution_report(summary)
    )
    if summary["status"] in {"BLOCKED", "CAPTCHA", "RATE_LIMITED"}:
        return 2  # normal safety stop, distinct from a successful crawl
    return 0 if summary["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

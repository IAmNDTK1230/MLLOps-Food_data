import os

from common import Runtime


def main() -> None:
    runtime = Runtime()
    runtime.spark.sql(f"CREATE CATALOG IF NOT EXISTS {runtime.args.catalog}")
    for layer in ("raw", "bronze", "silver", "gold"):
        runtime.spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS {runtime.args.catalog}.{getattr(runtime.args, layer + '_schema')}"
        )
    runtime.spark.sql(
        f"CREATE VOLUME IF NOT EXISTS {runtime.args.catalog}.{runtime.args.raw_schema}.{runtime.args.volume}"
    )
    # Create landing directories even when reviews are disabled.
    for kind in ("products", "reviews"):
        os.makedirs(runtime.landing(kind), exist_ok=True)


if __name__ == "__main__":
    main()

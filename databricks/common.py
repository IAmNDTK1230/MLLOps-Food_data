"""Databricks Runtime-only transformations; UTC, explicit schemas, Delta transactions."""

import argparse
import os
import re

from delta.tables import DeltaTable
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


class Runtime:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--catalog", default=os.getenv("DATABRICKS_CATALOG", "shopee"))
        for layer in ("raw", "bronze", "silver", "gold"):
            parser.add_argument(
                f"--{layer}-schema", default=os.getenv(f"DATABRICKS_{layer.upper()}_SCHEMA", layer)
            )
        parser.add_argument("--volume", default=os.getenv("DATABRICKS_VOLUME", "landing"))
        parser.add_argument("--run-id", default="manual")
        self.args = parser.parse_args()
        for key, value in vars(self.args).items():
            if key != "run_id" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                raise ValueError(f"Invalid SQL identifier: {key}")
        self.spark = SparkSession.builder.getOrCreate()
        self.spark.conf.set("spark.sql.session.timeZone", "UTC")

    def table(self, layer: str, name: str) -> str:
        return f"{self.args.catalog}.{getattr(self.args, layer + '_schema')}.{name}"

    def landing(self, kind: str) -> str:
        return (
            f"/Volumes/{self.args.catalog}/{self.args.raw_schema}/{self.args.volume}/shopee/{kind}"
        )


PRODUCT_FIELDS = {
    "product_id": "string",
    "shop_id": "string",
    "product_name": "string",
    "product_url": "string",
    "shop_name": "string",
    "shop_location": "string",
    "category": "string",
    "price": "double",
    "original_price": "double",
    "discount_percent": "double",
    "rating": "double",
    "rating_count": "long",
    "sold_count": "long",
    "stock": "long",
    "image_url": "string",
    "keyword": "string",
    "crawl_timestamp": "timestamp",
    "source": "string",
    "_run_id": "string",
}
REVIEW_FIELDS = {
    "review_id": "string",
    "product_id": "string",
    "shop_id": "string",
    "rating": "double",
    "comment": "string",
    "review_timestamp": "timestamp",
    "likes": "long",
    "crawl_timestamp": "timestamp",
    "_run_id": "string",
}


def schema(kind: str) -> StructType:
    types = {
        "string": StringType(),
        "double": DoubleType(),
        "long": LongType(),
        "timestamp": TimestampType(),
    }
    return StructType(
        [
            StructField(key, types[value], True)
            for key, value in (PRODUCT_FIELDS if kind == "products" else REVIEW_FIELDS).items()
        ]
    )


def bronze(runtime: Runtime, kind: str) -> None:
    """Auto Loader JSONL only: retain raw payload, reject rows, and ingest each file once.

    Parquet is an adjacent normalized export; reading both would duplicate observations.
    The durable checkpoint is the idempotency mechanism. Do not delete/reset it.
    """
    target = runtime.table("bronze", kind + "_raw")
    root = runtime.landing(kind)
    checkpoint = f"/Volumes/{runtime.args.catalog}/{runtime.args.raw_schema}/{runtime.args.volume}/_checkpoints/{kind}"
    # Precreate every expected column so empty/disabled review feeds are valid inputs downstream.
    runtime.spark.sql(f"""CREATE TABLE IF NOT EXISTS {target} (
        raw_payload STRING, _source_file STRING, _run_id STRING,
        _ingestion_timestamp TIMESTAMP, _ingestion_date DATE
    ) USING DELTA""")
    stream = (
        runtime.spark.readStream.format("cloudFiles")
        .schema(StructType([StructField("value", StringType(), True)]))
        .option("cloudFiles.format", "text")
        .option("cloudFiles.schemaLocation", checkpoint + "/schema")
        .option("cloudFiles.partitionColumns", "")
        .option("pathGlobFilter", "*.jsonl")
        .option("recursiveFileLookup", "true")
        .load(root)
    )
    frame = stream.select(
        F.col("value").alias("raw_payload"), F.col("_metadata.file_path").alias("_source_file")
    )
    frame = (
        frame.withColumn("_run_id", F.get_json_object("raw_payload", "$._run_id"))
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_ingestion_date", F.current_date())
    )
    query = (
        frame.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint)
        .trigger(availableNow=True)
        .toTable(target)
    )
    query.awaitTermination()


def parsed(runtime: Runtime, kind: str) -> DataFrame:
    source = runtime.spark.table(runtime.table("bronze", kind + "_raw"))
    # Bronze stores visible text. JSON strings are parsed first, then numbers normalized explicitly.
    string_schema = StructType(
        [
            StructField(key, StringType(), True)
            for key in (PRODUCT_FIELDS if kind == "products" else REVIEW_FIELDS)
        ]
    )
    frame = source.select(
        F.from_json("raw_payload", string_schema).alias("record"),
        "_source_file",
        "_ingestion_timestamp",
        "raw_payload",
    )
    frame = frame.select("record.*", "_source_file", "_ingestion_timestamp", "raw_payload")
    for key, dtype in (PRODUCT_FIELDS if kind == "products" else REVIEW_FIELDS).items():
        if dtype == "string":
            frame = frame.withColumn(
                key, F.when(F.length(F.trim(F.col(key))) > 0, F.trim(F.col(key)))
            )
        elif dtype == "timestamp":
            frame = frame.withColumn(key, F.expr(f"try_cast(`{key}` AS TIMESTAMP)"))
        else:
            text = F.lower(F.regexp_replace(F.col(key), r"(?i)^(đã bán|sold)\s*", ""))
            text = F.regexp_replace(text, r"[₫đ%\s+]", "")
            suffix = F.regexp_extract(text, r"([km])$", 1)
            grouped = text.rlike(r"^-?\d{1,3}(\.\d{3})+$")
            clean = F.when(grouped & (suffix == ""), F.regexp_replace(text, r"\.", ""))
            clean = clean.otherwise(F.regexp_replace(F.regexp_replace(text, "[km]$", ""), ",", "."))
            temp = f"__{key}"
            frame = frame.withColumn(temp, clean)
            number = F.expr(f"try_cast(`{temp}` AS DOUBLE)") * F.when(suffix == "k", 1000).when(
                suffix == "m", 1000000
            ).otherwise(1)
            if dtype == "long":
                number = F.when(number == F.floor(number), number).cast("long")
            frame = frame.withColumn(key, number).drop(temp)
    return frame


def merge_insert(runtime: Runtime, frame: DataFrame, table: str, keys: list[str]) -> None:
    if not runtime.spark.catalog.tableExists(table):
        frame.write.format("delta").mode("append").saveAsTable(table)
        return
    condition = " AND ".join(f"target.`{key}` <=> incoming.`{key}`" for key in keys)
    (
        DeltaTable.forName(runtime.spark, table)
        .alias("target")
        .merge(frame.alias("incoming"), condition)
        .whenNotMatchedInsertAll()
        .execute()
    )


def validity(frame: DataFrame, kind: str) -> Column:
    raw_schema = StructType(
        [
            StructField(key, StringType())
            for key in (PRODUCT_FIELDS if kind == "products" else REVIEW_FIELDS)
        ]
    )
    original = F.from_json(F.col("raw_payload"), raw_schema)
    valid = F.col("product_id").isNotNull() & F.col("crawl_timestamp").isNotNull()
    if kind == "products":
        valid = valid & F.col("product_name").isNotNull()
    fields = (
        {
            "rating": (0, 5),
            "price": (0, None),
            "original_price": (0, None),
            "discount_percent": (0, 100),
            "rating_count": (0, None),
            "sold_count": (0, None),
            "stock": (0, None),
        }
        if kind == "products"
        else {"rating": (0, 5), "likes": (0, None)}
    )
    for key, (low, high) in fields.items():
        present = original[key].isNotNull() & (F.trim(original[key]) != "")
        numeric = (
            F.col(key).isNotNull()
            & ~F.isnan(F.col(key))
            & (F.abs(F.col(key)) != float("inf"))
            & (F.col(key) >= low)
        )
        if high is not None:
            numeric = numeric & (F.col(key) <= high)
        valid = valid & (~present | numeric)
    if kind == "reviews":
        valid = valid & F.col("rating").isNotNull()
    return F.coalesce(valid, F.lit(False))


def silver(runtime: Runtime, kind: str) -> None:
    frame = parsed(runtime, kind)
    valid = validity(frame, kind)
    rejected = (
        frame.filter(~valid)
        .select("raw_payload", "_source_file", "_run_id")
        .withColumn("validation_error", F.lit("Invalid identity, timestamp or numeric value"))
        .withColumn("_record_hash", F.sha2("raw_payload", 256))
        .dropDuplicates(["_record_hash", "_source_file"])
    )
    merge_insert(
        runtime,
        rejected,
        runtime.table("silver", kind + "_rejected"),
        ["_record_hash", "_source_file"],
    )
    clean = frame.filter(valid).drop("raw_payload")
    keys = ["product_id", "crawl_timestamp"]
    if kind == "reviews":
        # JSON serialization preserves nulls and boundaries, unlike ambiguous string concatenation.
        digest = F.sha2(
            F.to_json(
                F.struct("product_id", "rating", "comment", "review_timestamp"),
                {"ignoreNullFields": "false"},
            ),
            256,
        )
        clean = clean.withColumn(
            "review_key",
            F.when(
                F.col("review_id").isNotNull(), F.concat(F.lit("id:"), F.col("review_id"))
            ).otherwise(F.concat(F.lit("hash:"), digest)),
        )
        keys = ["review_key"]
    window = Window.partitionBy(*keys).orderBy(
        F.col("_ingestion_timestamp").desc(), F.col("_source_file").desc()
    )
    clean = clean.withColumn("_rn", F.row_number().over(window)).filter("_rn = 1").drop("_rn")
    merge_insert(runtime, clean, runtime.table("silver", kind), keys)
    if kind == "products":
        full = runtime.spark.table(runtime.table("silver", "products"))
        latest = Window.partitionBy("product_id").orderBy(
            F.col("crawl_timestamp").desc(), F.col("_source_file").desc()
        )
        current = full.withColumn("_rn", F.row_number().over(latest)).filter("_rn = 1").drop("_rn")
        current.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(runtime.table("silver", "products_current"))


def gold_products(runtime: Runtime) -> None:
    products = runtime.spark.table(runtime.table("silver", "products")).withColumn(
        "date", F.to_date("crawl_timestamp")
    )
    window = Window.partitionBy("date", "product_id").orderBy(
        F.col("crawl_timestamp").desc(), F.col("_source_file").desc()
    )
    daily = (
        products.withColumn("_rn", F.row_number().over(window))
        .filter("_rn = 1")
        .select(
            "date",
            "product_id",
            "product_name",
            "shop_id",
            "shop_name",
            "category",
            "price",
            "rating",
            "rating_count",
            "sold_count",
            "discount_percent",
        )
    )
    categories = daily.groupBy("date", "category").agg(
        F.countDistinct("product_id").alias("product_count"),
        F.avg("price").alias("average_price"),
        F.avg("rating").alias("average_rating"),
        F.sum("sold_count").alias("total_sold"),
    )
    shops = daily.groupBy("date", "shop_id").agg(
        F.max("shop_name").alias("shop_name"),
        F.countDistinct("product_id").alias("product_count"),
        F.avg("rating").alias("average_rating"),
        F.sum("sold_count").alias("total_sold"),
    )
    for name, frame in (
        ("daily_product_metrics", daily),
        ("category_summary", categories),
        ("shop_summary", shops),
    ):
        frame.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            runtime.table("gold", name)
        )


def gold_reviews(runtime: Runtime) -> None:
    frame = runtime.spark.table(runtime.table("silver", "reviews"))
    frame = frame.withColumn("clean_text", F.trim(F.regexp_replace("comment", r"\s+", " ")))
    frame = frame.filter(F.length("clean_text") > 0).select(
        "review_key",
        "review_id",
        "product_id",
        "shop_id",
        "clean_text",
        "rating",
        "review_timestamp",
        "crawl_timestamp",
    )
    frame.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        runtime.table("gold", "review_ml_dataset")
    )

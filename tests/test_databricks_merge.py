"""Exercise the actual helper with SDK doubles, without installing PySpark/Delta."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def helper():
    source = Path(__file__).resolve().parents[1] / "databricks" / "common.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "merge_insert"
    )
    # Load the real function body in isolation; cloud imports stay runtime-provided.
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, function], type_ignores=[])
    delta = Mock()
    namespace = {"DeltaTable": delta}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["merge_insert"], delta


@pytest.mark.parametrize("name", ["products", "reviews", "products_rejected", "reviews_rejected"])
def test_first_write_uses_append_and_never_merges(helper, name):
    merge_insert, delta = helper
    table = f"shopee.silver.{name}"
    runtime = SimpleNamespace(spark=Mock())
    runtime.spark.catalog.tableExists.return_value = False
    writer = Mock()
    writer.format.return_value = writer

    def serverless_mode(mode):
        if mode != "append":
            raise AssertionError(f"First write must preserve data with append, received {mode}")
        return writer

    writer.mode.side_effect = serverless_mode
    frame = SimpleNamespace(write=writer)
    merge_insert(runtime, frame, table, ["product_id", "crawl_timestamp"])
    runtime.spark.catalog.tableExists.assert_called_once_with(table)
    writer.format.assert_called_once_with("delta")
    writer.mode.assert_called_once_with("append")
    writer.saveAsTable.assert_called_once_with(table)
    delta.forName.assert_not_called()


@pytest.mark.parametrize("keys", [["product_id", "crawl_timestamp"], ["review_key"]])
def test_existing_table_preserves_insert_only_merge(helper, keys):
    merge_insert, delta = helper
    table = "shopee.silver.existing"
    runtime = SimpleNamespace(spark=Mock())
    runtime.spark.catalog.tableExists.return_value = True
    # No .write attribute: any attempt to append/overwrite an existing table fails.
    frame = SimpleNamespace(alias=Mock(return_value="incoming-frame"))
    target = Mock()
    target.alias.return_value = target
    target.merge.return_value = target
    target.whenNotMatchedInsertAll.return_value = target
    delta.forName.return_value = target
    merge_insert(runtime, frame, table, keys)
    runtime.spark.catalog.tableExists.assert_called_once_with(table)
    delta.forName.assert_called_once_with(runtime.spark, table)
    target.alias.assert_called_once_with("target")
    frame.alias.assert_called_once_with("incoming")
    target.merge.assert_called_once_with(
        "incoming-frame", " AND ".join(f"target.`{key}` <=> incoming.`{key}`" for key in keys)
    )
    target.whenNotMatchedInsertAll.assert_called_once_with()
    target.execute.assert_called_once_with()

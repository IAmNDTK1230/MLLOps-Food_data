"""Transactional, idempotent PostgreSQL load of the canonical order CSV."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from src.sales.audit import SOURCE, audit_orders, read_orders, require_loadable
from src.warehouse.loader import connection, migrate


def source_hash(path=SOURCE):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_names(orders):
    names = defaultdict(Counter)
    for restaurant_id, name in zip(orders.restaurant_id, orders.restaurant_name, strict=True):
        if isinstance(name, str) and name.strip():
            names[str(restaurant_id)][name.strip()] += 1
    return {str(restaurant_id): sorted(names[str(restaurant_id)].items(),
                                      key=lambda pair: (-pair[1], pair[0]))[0][0]
            if names[str(restaurant_id)] else None
            for restaurant_id in orders.restaurant_id.unique()}


def load_sales(path=SOURCE, *, connect=connection):
    audit = audit_orders(path)
    require_loadable(audit)
    sha = source_hash(path)
    with connect() as conn:
        applied = migrate(conn)
        prior = conn.execute(
            "SELECT source_sha256,row_count FROM sales_source_state WHERE source_name=%s",
            (Path(path).name,),
        ).fetchone()
        order_count = conn.execute("SELECT COUNT(*) FROM fact_order").fetchone()[0]
        if prior == (sha, audit["rows"]) and order_count == audit["rows"]:
            return {"status": "unchanged", "orders": order_count,
                    "matched_restaurants": audit["matched_count"],
                    "migrations_applied": applied, "source_sha256": sha}

        orders = read_orders(path)
        names = canonical_names(orders)
        with conn.cursor() as cursor:
            cursor.executemany("""INSERT INTO dim_sales_restaurant(restaurant_id,restaurant_name)
                VALUES (%s,%s) ON CONFLICT (restaurant_id) DO UPDATE
                SET restaurant_name=EXCLUDED.restaurant_name""", names.items())
            conn.execute("DELETE FROM sales_review_restaurant_map")
            cursor.executemany("""INSERT INTO sales_review_restaurant_map
                (restaurant_id,restaurant_key,match_method) VALUES (%s,%s,%s)""",
                [(item["restaurant_id"], item["restaurant_key"], item["match_method"])
                 for item in audit["matched"]])
            conn.execute("""CREATE TEMP TABLE sales_order_stage (
                order_id TEXT PRIMARY KEY, restaurant_id TEXT NOT NULL,
                order_date DATE NOT NULL, date_key INTEGER NOT NULL,
                sales_amount NUMERIC(14,2) NOT NULL, order_status TEXT NOT NULL
            ) ON COMMIT DROP""")
            with cursor.copy("""COPY sales_order_stage
                (order_id,restaurant_id,order_date,date_key,sales_amount,order_status)
                FROM STDIN""") as copy:
                with Path(path).open(encoding="utf-8-sig", newline="") as source:
                    for row in csv.DictReader(source):
                        when = date.fromisoformat(row["order_date"])
                        copy.write_row((row["order_id"], row["restaurant_id"], when,
                                        int(when.strftime("%Y%m%d")),
                                        Decimal(row["sales_amount"]), row["order_status"]))
            conn.execute("""INSERT INTO dim_date(date_key,calendar_date,day,month,year)
                SELECT DISTINCT date_key,order_date,EXTRACT(DAY FROM order_date)::SMALLINT,
                       EXTRACT(MONTH FROM order_date)::SMALLINT,
                       EXTRACT(YEAR FROM order_date)::SMALLINT
                FROM sales_order_stage WHERE TRUE ON CONFLICT (date_key) DO NOTHING""")
            conn.execute("""INSERT INTO fact_order
                (order_id,restaurant_id,date_key,sales_amount,order_status)
                SELECT order_id,restaurant_id,date_key,sales_amount,order_status
                FROM sales_order_stage WHERE TRUE
                ON CONFLICT (order_id) DO UPDATE SET
                  restaurant_id=EXCLUDED.restaurant_id, date_key=EXCLUDED.date_key,
                  sales_amount=EXCLUDED.sales_amount, order_status=EXCLUDED.order_status
                WHERE (fact_order.restaurant_id,fact_order.date_key,fact_order.sales_amount,
                       fact_order.order_status) IS DISTINCT FROM
                      (EXCLUDED.restaurant_id,EXCLUDED.date_key,EXCLUDED.sales_amount,
                       EXCLUDED.order_status)""")
            conn.execute("""INSERT INTO sales_source_state(source_name,source_sha256,row_count)
                VALUES (%s,%s,%s) ON CONFLICT (source_name) DO UPDATE SET
                source_sha256=EXCLUDED.source_sha256,row_count=EXCLUDED.row_count,
                loaded_at=now()""", (Path(path).name, sha, audit["rows"]))
        loaded = conn.execute("SELECT COUNT(*) FROM fact_order").fetchone()[0]
        mappings = conn.execute("SELECT COUNT(*) FROM sales_review_restaurant_map").fetchone()[0]
    return {"status": "loaded", "orders": loaded, "matched_restaurants": mappings,
            "migrations_applied": applied, "source_sha256": sha}


def main():
    print(json.dumps(load_sales(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

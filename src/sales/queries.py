"""Whitelisted, parameterized sales analytics. No model-generated SQL."""

import calendar
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo


def vietnam_today():
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()


def _row(cursor):
    value = cursor.fetchone()
    if value is None:
        return None
    return dict(zip((item.name for item in cursor.description), value, strict=True))


def _rows(cursor):
    columns = [item.name for item in cursor.description]
    return [dict(zip(columns, value, strict=True)) for value in cursor.fetchall()]


def _date(month):
    if not isinstance(month, str) or len(month) != 7 or month[4] != "-":
        raise ValueError("Month must use YYYY-MM")
    return datetime.strptime(month + "-01", "%Y-%m-%d").date()


def restaurant(conn, restaurant_id):
    return _row(conn.execute("""SELECT d.restaurant_id,d.restaurant_name,
        m.restaurant_key,r.restaurant_name AS review_restaurant_name
        FROM dim_sales_restaurant d
        LEFT JOIN sales_review_restaurant_map m USING (restaurant_id)
        LEFT JOIN dim_restaurant r ON r.restaurant_key=m.restaurant_key
        WHERE d.restaurant_id=%s""", (str(restaurant_id),)))


def list_restaurants(conn, *, matched_only=False):
    where = "WHERE m.restaurant_key IS NOT NULL" if matched_only else ""
    return _rows(conn.execute(f"""SELECT d.restaurant_id,d.restaurant_name,m.restaurant_key
        FROM dim_sales_restaurant d LEFT JOIN sales_review_restaurant_map m USING (restaurant_id)
        {where} ORDER BY d.restaurant_name NULLS LAST,d.restaurant_id"""))


def monthly_sales(conn, restaurant_id, month=None, *, today=None):
    today = today or vietnam_today()
    current_month = today.replace(day=1)
    if month is None:
        return _row(conn.execute("""SELECT * FROM restaurant_monthly_sales
            WHERE restaurant_id=%s AND period_start < %s
            ORDER BY period_start DESC LIMIT 1""", (str(restaurant_id), current_month)))
    requested = _date(month)
    if requested > current_month:
        raise ValueError("Future months are not available as observed sales")
    return _row(conn.execute("""SELECT * FROM restaurant_monthly_sales
        WHERE restaurant_id=%s AND period_start=%s""", (str(restaurant_id), requested)))


def weekly_sales(conn, restaurant_id, week_start=None, *, today=None):
    today = today or vietnam_today()
    current_week = today.fromordinal(today.toordinal() - today.weekday())
    if week_start is None:
        return _row(conn.execute("""SELECT * FROM restaurant_weekly_sales
            WHERE restaurant_id=%s AND period_start < %s
            ORDER BY period_start DESC LIMIT 1""", (str(restaurant_id), current_week)))
    if week_start > current_week:
        raise ValueError("Future weeks are not available as observed sales")
    return _row(conn.execute("""SELECT * FROM restaurant_weekly_sales
        WHERE restaurant_id=%s AND period_start=%s""", (str(restaurant_id), week_start)))


def monthly_trend(conn, restaurant_id, *, limit=24):
    if not 1 <= limit <= 36:
        raise ValueError("Trend limit must be 1..36")
    return list(reversed(_rows(conn.execute("""SELECT * FROM restaurant_monthly_sales
        WHERE restaurant_id=%s ORDER BY period_start DESC LIMIT %s""",
        (str(restaurant_id), limit)))))


def monthly_comparison(conn, restaurant_id, month=None, *, today=None):
    today = today or vietnam_today()
    row = monthly_sales(conn, restaurant_id, month, today=today)
    if row is None:
        return None
    current_month = today.replace(day=1)
    if row["period_start"] != current_month:
        return {**row, "comparison_revenue": row["previous_delivered_sales"],
                "comparison_growth": row["mom_growth"], "comparison_basis": "full_prior_month"}
    previous = current_month.fromordinal(current_month.toordinal() - 1).replace(day=1)
    cutoff = previous.replace(day=min(today.day, calendar.monthrange(
        previous.year, previous.month)[1]))
    prior = conn.execute("""SELECT COALESCE(SUM(delivered_sales),0)
        FROM restaurant_daily_sales WHERE restaurant_id=%s
        AND period_start BETWEEN %s AND %s""", (str(restaurant_id), previous, cutoff)).fetchone()[0]
    growth = (row["delivered_sales"] - prior) / prior if prior else None
    return {**row, "comparison_revenue": prior, "comparison_growth": growth,
            "comparison_basis": "same_day_of_month_partial"}


def weekly_trend(conn, restaurant_id, *, limit=16):
    if not 1 <= limit <= 52:
        raise ValueError("Trend limit must be 1..52")
    return list(reversed(_rows(conn.execute("""SELECT * FROM restaurant_weekly_sales
        WHERE restaurant_id=%s ORDER BY period_start DESC LIMIT %s""",
        (str(restaurant_id), limit)))))


def sales_summary(conn, restaurant_id):
    return _row(conn.execute("""SELECT * FROM restaurant_sales_summary
        WHERE restaurant_id=%s""", (str(restaurant_id),)))


def order_status_summary(conn, restaurant_id):
    return _rows(conn.execute("""SELECT * FROM restaurant_order_status_summary
        WHERE restaurant_id=%s ORDER BY order_status""", (str(restaurant_id),)))


def top_sales_restaurants(conn, *, limit=10):
    if not 1 <= limit <= 100:
        raise ValueError("Top restaurant limit must be 1..100")
    return _rows(conn.execute("""SELECT d.restaurant_id,d.restaurant_name,s.delivered_sales
        FROM restaurant_sales_summary s JOIN dim_sales_restaurant d USING (restaurant_id)
        ORDER BY s.delivered_sales DESC LIMIT %s""", (limit,)))


def lowest_growth_month(conn, restaurant_id, *, today=None):
    current_month = (today or vietnam_today()).replace(day=1)
    return _row(conn.execute("""SELECT * FROM restaurant_monthly_sales
        WHERE restaurant_id=%s AND period_start<%s AND mom_growth<0
        ORDER BY mom_growth ASC,period_start DESC LIMIT 1""",
        (str(restaurant_id), current_month)))


def highest_sales_month(conn, restaurant_id, *, today=None):
    current_month = (today or vietnam_today()).replace(day=1)
    return _row(conn.execute("""SELECT * FROM restaurant_monthly_sales
        WHERE restaurant_id=%s AND period_start<%s
        ORDER BY delivered_sales DESC,period_start DESC LIMIT 1""",
        (str(restaurant_id), current_month)))


def monthly_summary_document(conn, restaurant_id, month):
    """Create one deterministic evidence document from verified SQL aggregates."""
    identity = restaurant(conn, restaurant_id)
    metric = monthly_sales(conn, restaurant_id, month)
    if not identity or not metric:
        return None
    def value(key):
        return str(metric[key] if metric[key] is not None else "N/A")
    text = (f"Restaurant ID: {restaurant_id}; name: {identity['restaurant_name'] or 'unknown'}; "
            f"month: {month}; delivered revenue VND: {value('delivered_sales')}; "
            f"gross sales VND: {value('gross_sales')}; orders: {value('order_count')}; "
            f"delivered: {value('delivered_count')}; cancelled: {value('cancelled_count')}; "
            f"refunded: {value('refunded_count')}; delivered AOV VND: {value('aov')}; "
            f"MoM growth: {value('mom_growth')}.")
    return {"text": text, "metadata": {"document_type": "sales_summary",
            "restaurant_id": str(restaurant_id),
            "restaurant_name": identity["restaurant_name"],
            "period_type": "month", "period_start": month + "-01",
            "source": "postgres_sales_mart", "schema_version": "003"}}


def summary_document_count(conn):
    return conn.execute("SELECT COUNT(*) FROM restaurant_monthly_sales").fetchone()[0]


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

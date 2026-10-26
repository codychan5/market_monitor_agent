"""Exact queries over the fact store. Brokers are never blended: comparisons report each figure and the spread."""
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import pairwise

import psycopg

from .ingest.boe import MacroPoint
from .vocab import MACRO_SERIES, Broker, Metric, Submarket

# A macro query covers at most five years: enough for the brief, small enough to keep tool results readable.
MAX_MACRO_RANGE_DAYS = 1830
PERCENT_PLACES = Decimal("0.01")
QUARTER = re.compile(r"^(\d{4})Q([1-4])$")
MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")


class NotInData(Exception):
    """The store holds no such figure, document or citation. The message says what is missing."""


@dataclass(frozen=True)
class FactRow:
    fact_id: int
    broker: str
    metric: str
    submarket: str
    broker_label: str
    area_fit: str | None
    period: str
    value: Decimal
    unit: str
    measure_note: str
    locator: str | None
    citation: str
    definition: str | None


@dataclass(frozen=True)
class Comparison:
    rows: list[FactRow]
    minimum: Decimal
    maximum: Decimal
    spread: Decimal
    brokers: int
    period_note: str


@dataclass(frozen=True)
class Change:
    from_value: Decimal
    to_value: Decimal
    change: Decimal
    change_pct: Decimal | None
    unit: str
    citations: list[str]


def period_key(period: str) -> tuple[int, int]:
    """Sort key putting quarters and months on one scale: a quarter sorts at its last month."""
    if m := QUARTER.match(period):
        return int(m.group(1)), int(m.group(2)) * 3
    if m := MONTH.match(period):
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"period must look like 2026Q3 or 2026-08, got {period!r}")


def format_value(value: Decimal, unit: str) -> str:
    """A figure written the way a report prints it, e.g. £110.00 psf, 7.1%, 3.1m sq ft, £2.6bn."""
    return {"GBP psf pa": f"£{value} psf", "%": f"{value}%", "m sq ft": f"{value}m sq ft",
            "GBP bn": f"£{value}bn"}.get(unit, f"{value} {unit}")


FACT_SELECT = (
    "SELECT m.id, m.broker, m.metric, m.submarket, m.broker_label, a.fit, m.period, m.value, m.unit, m.measure_note,"
    " m.locator, d.definition FROM broker_metrics m"
    " LEFT JOIN broker_area_map a ON a.broker = m.broker AND a.broker_label = m.broker_label"
    " LEFT JOIN metric_definitions d ON d.broker = m.broker AND d.metric = m.metric")


def _fact(row) -> FactRow:
    return FactRow(fact_id=row[0], broker=row[1], metric=row[2], submarket=row[3], broker_label=row[4],
                   area_fit=row[5], period=row[6], value=row[7], unit=row[8], measure_note=row[9], locator=row[10],
                   citation=f"F{row[0]}", definition=row[11])


def get_metric(conn: psycopg.Connection, metric: Metric, submarket: Submarket, period_from: str | None = None,
               period_to: str | None = None, brokers: list[Broker] | None = None) -> list[FactRow]:
    """Every broker's figure for a metric in a submarket, sorted by broker then period."""
    rows = [_fact(r) for r in conn.execute(f"{FACT_SELECT} WHERE m.metric = %s AND m.submarket = %s",
                                           (Metric(metric).value, Submarket(submarket).value))]
    low = period_key(period_from) if period_from else None
    high = period_key(period_to) if period_to else None
    wanted = {Broker(b).value for b in brokers} if brokers else None
    rows = [r for r in rows if (low is None or period_key(r.period) >= low)
            and (high is None or period_key(r.period) <= high) and (wanted is None or r.broker in wanted)]
    return sorted(rows, key=lambda r: (r.broker, period_key(r.period)))


def compare_brokers(conn: psycopg.Connection, metric: Metric, submarket: Submarket, period: str) -> Comparison:
    """Each broker's figure for one period, with the spread. A quarter also accepts a broker's monthly reading from
    within it (the latest one) when that broker has no quarterly figure."""
    rows = get_metric(conn, metric, submarket)
    notes = []
    if m := QUARTER.match(period):
        year, quarter = int(m.group(1)), int(m.group(2))
        months = {f"{year}-{month:02d}" for month in range(quarter * 3 - 2, quarter * 3 + 1)}
        chosen: dict[str, FactRow] = {r.broker: r for r in rows if r.period == period}
        for r in sorted((r for r in rows if r.period in months), key=lambda r: period_key(r.period)):
            if r.broker not in chosen or chosen[r.broker].period != period:
                chosen[r.broker] = r
        for r in chosen.values():
            if r.period != period:
                notes.append(f"{r.broker}: {MONTH_NAMES[int(r.period[5:]) - 1]} {r.period[:4]} monthly reading")
        selected = sorted(chosen.values(), key=lambda r: r.broker)
    else:
        period_key(period)  # validates the format
        selected = [r for r in rows if r.period == period]
    if not selected:
        raise NotInData(f"No {Metric(metric).value} figures for {Submarket(submarket).value} in {period}.")
    values = [r.value for r in selected]
    return Comparison(rows=selected, minimum=min(values), maximum=max(values), spread=max(values) - min(values),
                      brokers=len(selected), period_note="; ".join(notes))


def percent_change(start: Decimal, end: Decimal) -> Decimal | None:
    """The change from start to end as a percentage of start, to two decimal places; None when start is zero."""
    return None if start == 0 else ((end - start) / start * 100).quantize(PERCENT_PLACES)


def metric_change(conn: psycopg.Connection, metric: Metric, submarket: Submarket, broker: Broker, from_period: str,
                  to_period: str) -> Change:
    """One broker's change between two periods, computed here so the model never does arithmetic."""
    rows = {r.period: r for r in get_metric(conn, metric, submarket, brokers=[broker])}
    missing = [p for p in (from_period, to_period) if p not in rows]
    if missing:
        raise NotInData(f"{Broker(broker).value} has no {Metric(metric).value} figure for {Submarket(submarket).value} "
                        f"in {', '.join(missing)}; it has {', '.join(sorted(rows, key=period_key)) or 'none'}.")
    start, end = rows[from_period], rows[to_period]
    return Change(from_value=start.value, to_value=end.value, change=end.value - start.value,
                  change_pct=percent_change(start.value, end.value), unit=end.unit,
                  citations=[start.citation, end.citation])


@dataclass(frozen=True)
class TrendStep:
    from_period: str
    to_period: str
    change: Decimal
    change_pct: Decimal | None


@dataclass(frozen=True)
class BrokerTrend:
    broker: str
    points: list[FactRow]
    steps: list[TrendStep]


def metric_trend(conn: psycopg.Connection, metric: Metric, submarket: Submarket,
                 broker: Broker | None = None) -> list[BrokerTrend]:
    """Each broker's figures for a metric in a submarket, oldest period first, with the change between consecutive
    periods computed here so the model never does arithmetic. Brokers are kept apart: they measure different
    things."""
    rows = get_metric(conn, metric, submarket, brokers=[broker] if broker else None)
    if not rows:
        raise NotInData(f"No {Metric(metric).value} figures for {Submarket(submarket).value}"
                        + (f" from {Broker(broker).value}." if broker else "."))
    by_broker: dict[str, list[FactRow]] = {}
    for row in rows:
        by_broker.setdefault(row.broker, []).append(row)
    trends = []
    for name in sorted(by_broker):
        points = sorted(by_broker[name], key=lambda r: period_key(r.period))
        steps = [TrendStep(from_period=a.period, to_period=b.period, change=b.value - a.value,
                           change_pct=percent_change(a.value, b.value)) for a, b in pairwise(points)]
        trends.append(BrokerTrend(broker=name, points=points, steps=steps))
    return trends


def get_macro(conn: psycopg.Connection, series: str, date_from: date, date_to: date) -> list[MacroPoint]:
    if series not in MACRO_SERIES:
        raise ValueError(f"series must be one of {', '.join(MACRO_SERIES)}, got {series!r}")
    if date_to < date_from or (date_to - date_from).days > MAX_MACRO_RANGE_DAYS:
        raise ValueError(f"date range must run forwards and span at most {MAX_MACRO_RANGE_DAYS} days")
    return [MacroPoint(*row) for row in conn.execute(
        "SELECT series, day, value FROM macro_series WHERE series = %s AND day BETWEEN %s AND %s ORDER BY day",
        (series, date_from, date_to))]


def get_coverage(conn: psycopg.Connection) -> dict:
    """What the store can answer: brokers, metric periods per submarket, macro series and documents."""
    metrics: dict[str, dict[str, list[str]]] = {}
    for metric, submarket, period in conn.execute("SELECT DISTINCT metric, submarket, period FROM broker_metrics"):
        metrics.setdefault(metric, {}).setdefault(submarket, []).append(period)
    for submarkets in metrics.values():
        for periods in submarkets.values():
            periods.sort(key=period_key)
    macro = [{"series": s, "name": MACRO_SERIES.get(s, s), "first_day": first.isoformat(), "last_day": last.isoformat()}
             for s, first, last in conn.execute(
                 "SELECT series, min(day), max(day) FROM macro_series GROUP BY series ORDER BY series")]
    count, newest = conn.execute("SELECT count(*), max(published_at) FROM documents").fetchone()
    return {"brokers": [r[0] for r in conn.execute("SELECT DISTINCT broker FROM broker_metrics ORDER BY broker")],
            "metrics": metrics, "macro_series": macro,
            "documents": {"count": count, "newest_published_at": newest.isoformat() if newest else None}}


def get_portfolio(conn: psycopg.Connection) -> list[dict]:
    return [{"name": r[0], "address": r[1], "submarket": r[2], "size_sqft": r[3], "status": r[4], "notes": r[5]}
            for r in conn.execute("SELECT name, address, submarket, size_sqft, status, notes FROM portfolio_assets"
                                  " ORDER BY name")]


def get_mandate(conn: psycopg.Connection) -> dict:
    row = conn.execute("SELECT body FROM mandate ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else {}

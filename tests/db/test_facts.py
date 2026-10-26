from datetime import date
from decimal import Decimal

import pytest

from london_office_monitor.facts import (NotInData, compare_brokers, format_value, get_coverage, get_macro,
                                         get_mandate, get_metric, get_portfolio, metric_change, metric_trend)
from london_office_monitor.vocab import Broker, Metric, Submarket


@pytest.mark.db
def test_get_metric_returns_cited_rows_with_definitions(seeded):
    rows = get_metric(seeded, Metric.PRIME_RENT, Submarket.CITY_CORE)
    assert [(r.broker, r.period, r.value) for r in rows] == [
        ("BNP Paribas", "2026Q2", Decimal("95.00")), ("JLL", "2026Q2", Decimal("100.00")),
        ("JLL", "2026Q3", Decimal("110.00"))]
    assert all(r.citation == f"F{r.fact_id}" for r in rows)
    assert rows[0].area_fit == "approximate" and rows[0].broker_label == "City" and rows[1].locator == "p. 3"


@pytest.mark.db
def test_get_metric_filters_periods_across_quarter_and_month_formats(seeded):
    rows = get_metric(seeded, Metric.VACANCY_RATE, Submarket.CENTRAL_LONDON, period_from="2026-07")
    assert [(r.broker, r.period) for r in rows] == [("CBRE", "2026-08")]


@pytest.mark.db
def test_compare_brokers_reports_spread_not_average(seeded):
    c = compare_brokers(seeded, Metric.VACANCY_RATE, Submarket.CENTRAL_LONDON, "2026Q2")
    assert (c.minimum, c.maximum, c.spread, c.brokers) == (Decimal("6.3"), Decimal("8.4"), Decimal("2.1"), 3)
    assert not hasattr(c, "average")


@pytest.mark.db
def test_quarter_matches_monthly_rows(seeded):
    c = compare_brokers(seeded, Metric.VACANCY_RATE, Submarket.CENTRAL_LONDON, "2026Q3")
    assert [r.broker for r in c.rows] == ["CBRE"] and "August 2026" in c.period_note


@pytest.mark.db
def test_compare_brokers_with_no_rows_raises_not_in_data(seeded):
    with pytest.raises(NotInData):
        compare_brokers(seeded, Metric.PRIME_YIELD, Submarket.SOUTHBANK, "2026Q3")


@pytest.mark.db
def test_metric_change_values(seeded):
    ch = metric_change(seeded, Metric.PRIME_RENT, Submarket.CITY_CORE, Broker.JLL, "2026Q2", "2026Q3")
    assert (ch.from_value, ch.to_value, ch.change, ch.change_pct) == (Decimal("100.00"), Decimal("110.00"),
                                                                      Decimal("10.00"), Decimal("10.00"))
    assert len(ch.citations) == 2 and all(c.startswith("F") for c in ch.citations)


@pytest.mark.db
def test_metric_change_missing_row_raises_not_in_data(seeded):
    with pytest.raises(NotInData):
        metric_change(seeded, Metric.PRIME_RENT, Submarket.CITY_CORE, Broker.SAVILLS, "2026Q2", "2026Q3")


@pytest.mark.db
def test_coverage_lists_what_exists(seeded):
    coverage = get_coverage(seeded)
    assert coverage["metrics"]["vacancy_rate"]["Central London"] == ["2026Q2", "2026-08"]
    assert "Knight Frank" not in coverage["brokers"] and "JLL" in coverage["brokers"]


@pytest.mark.db
def test_get_macro_returns_points_in_order_and_bounds_the_range(seeded):
    seeded.execute("INSERT INTO macro_series VALUES ('IUDBEDR', '2026-10-22', 3.75), ('IUDBEDR', '2026-10-21', 3.75)")
    points = get_macro(seeded, "IUDBEDR", date(2026, 10, 1), date(2026, 10, 24))
    assert [p.day for p in points] == [date(2026, 10, 21), date(2026, 10, 22)]
    with pytest.raises(ValueError):
        get_macro(seeded, "IUDBEDR", date(2020, 1, 1), date(2026, 10, 24))
    with pytest.raises(ValueError):
        get_macro(seeded, "NOT_A_SERIES", date(2026, 10, 1), date(2026, 10, 24))


@pytest.mark.db
def test_portfolio_and_mandate(seeded):
    assert {a["name"] for a in get_portfolio(seeded)} == {"Regent Quarter", "99 City Road"}
    assert get_mandate(seeded)["label"] == "illustrative"


def test_format_value_per_unit():
    assert format_value(Decimal("110.00"), "GBP psf pa") == "£110.00 psf"
    assert format_value(Decimal("7.1"), "%") == "7.1%"
    assert format_value(Decimal("3.1"), "m sq ft") == "3.1m sq ft"
    assert format_value(Decimal("2.6"), "GBP bn") == "£2.6bn"


@pytest.mark.db
def test_metric_trend_orders_each_brokers_figures_and_computes_the_changes(seeded):
    # The owner asked for trends in rents and vacancy, drawn as charts; every change is computed here, in code.
    series = {s.broker: s for s in metric_trend(seeded, Metric.PRIME_RENT, Submarket.CITY_CORE)}
    jll = series["JLL"]
    assert [p.period for p in jll.points] == ["2026Q2", "2026Q3"]
    assert [p.value for p in jll.points] == [Decimal("100.00"), Decimal("110.00")]
    assert [(s.from_period, s.to_period, s.change, s.change_pct) for s in jll.steps] == [
        ("2026Q2", "2026Q3", Decimal("10.00"), Decimal("10.00"))]
    assert len(series["BNP Paribas"].points) == 1 and series["BNP Paribas"].steps == []


@pytest.mark.db
def test_metric_trend_without_figures_is_not_in_the_data(seeded):
    with pytest.raises(NotInData):
        metric_trend(seeded, Metric.TAKE_UP, Submarket.SOUTHBANK)

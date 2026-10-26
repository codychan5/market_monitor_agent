from london_office_monitor.paths import SEED_DIR
from london_office_monitor.seed.load import read_seed
from london_office_monitor.vocab import Broker, Metric, Submarket


def test_seed_meets_spec_minimums():
    b = read_seed(SEED_DIR)
    assert len(b.metrics) >= 60
    assert len({r.broker for r in b.metrics} - {Broker.DELOITTE}) >= 5
    assert all("knightfrank" not in str(s.url) for s in b.sources)
    assert len({r.broker for r in b.metrics
                if r.metric == Metric.VACANCY_RATE and r.submarket == Submarket.CENTRAL_LONDON}) >= 3
    assert len({r.broker for r in b.metrics
                if r.metric == Metric.PRIME_RENT and r.submarket == Submarket.CITY_CORE}) >= 2
    assert {a.name for a in b.assets} >= {"Regent Quarter", "99 City Road", "108 Cannon Street", "138 Cheapside",
                                           "16 Old Bailey"}
    assert b.mandate.label == "illustrative"


def test_seed_has_a_figure_with_two_periods_for_metric_change():
    b = read_seed(SEED_DIR)
    periods: dict[tuple, set[str]] = {}
    for r in b.metrics:
        periods.setdefault((r.broker, r.metric, r.submarket), set()).add(r.period)
    assert any(len(p) >= 2 for p in periods.values())


def test_pdf_sourced_rows_carry_a_page_locator():
    b = read_seed(SEED_DIR)
    pdf_sources = {s.source_key for s in b.sources if str(s.url).lower().endswith(".pdf")}
    assert all(r.locator for r in b.metrics if r.source_key in pdf_sources)

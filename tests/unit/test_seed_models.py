import shutil
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from london_office_monitor.seed.load import SeedError, read_seed
from london_office_monitor.seed.models import BrokerMetricRow, SourceRow
from tests.conftest import SEED_MIN

VALID_ROW = {"broker": "JLL", "metric": "prime_rent", "submarket": "City Core", "broker_label": "City Core",
             "period": "2026Q3", "value": Decimal("110.00"), "unit": "GBP psf pa", "measure_note": "headline prime",
             "source_key": "t-a", "checked_at": date(2026, 10, 24), "check_method": "manual"}


def test_valid_row_is_accepted():
    assert BrokerMetricRow(**VALID_ROW).value == Decimal("110.00")


def test_knight_frank_row_is_rejected():
    with pytest.raises(ValidationError, match="excluded"):
        BrokerMetricRow(**{**VALID_ROW, "broker": "Knight Frank"})


def test_knight_frank_source_is_rejected():
    with pytest.raises(ValidationError, match="excluded"):
        SourceRow(source_key="kf", publisher="Knight Frank", title="Spotlight", licence_note="n",
                  url="https://www.knightfrank.co.uk/site-assets/research/spotlight.pdf", published=date(2026, 10, 1))


def test_source_without_publication_date_is_accepted():
    row = SourceRow(source_key="nf", publisher="Nan Fung Group", title="International property", licence_note="n",
                    url="https://www.nanfung.com/en/property/international/", published="")
    assert row.published is None


def test_unit_must_match_metric():
    with pytest.raises(ValidationError, match="unit"):
        BrokerMetricRow(**{**VALID_ROW, "unit": "%"})


@pytest.mark.parametrize("period", ["2026Q5", "2026-13", "Q3 2026", "2026"])
def test_bad_period_is_rejected(period):
    with pytest.raises(ValidationError):
        BrokerMetricRow(**{**VALID_ROW, "period": period})


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        BrokerMetricRow(**VALID_ROW, colour="blue")


def test_read_seed_loads_the_minimal_seed():
    bundle = read_seed(SEED_MIN)
    assert len(bundle.metrics) == 7 and len(bundle.sources) == 3 and bundle.mandate.label == "illustrative"
    assert {a.name for a in bundle.assets} == {"Regent Quarter", "99 City Road"}


def test_read_seed_reports_every_bad_row(tmp_path):
    shutil.copytree(SEED_MIN, tmp_path, dirs_exist_ok=True)
    lines = (tmp_path / "broker_metrics.csv").read_text().splitlines()
    lines[2] = lines[2].replace("Savills", "Knight Frank")      # file line 3
    lines[3] = lines[3].replace("2026Q2", "2026Q9")              # file line 4
    (tmp_path / "broker_metrics.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(SeedError) as e:
        read_seed(tmp_path)
    assert "broker_metrics.csv:3" in str(e.value) and "broker_metrics.csv:4" in str(e.value)


def test_read_seed_reports_unknown_source_key(tmp_path):
    shutil.copytree(SEED_MIN, tmp_path, dirs_exist_ok=True)
    text = (tmp_path / "broker_metrics.csv").read_text().replace(",t-c,", ",t-missing,")
    (tmp_path / "broker_metrics.csv").write_text(text)
    with pytest.raises(SeedError, match="t-missing"):
        read_seed(tmp_path)


def test_read_seed_reports_submarket_that_contradicts_area_map(tmp_path):
    shutil.copytree(SEED_MIN, tmp_path, dirs_exist_ok=True)
    text = (tmp_path / "broker_metrics.csv").read_text().replace(
        "JLL,prime_rent,City Core,City Core,2026Q3", "JLL,prime_rent,Midtown,City Core,2026Q3")
    (tmp_path / "broker_metrics.csv").write_text(text)
    with pytest.raises(SeedError, match="maps to City Core"):
        read_seed(tmp_path)


def test_read_seed_reports_label_missing_from_area_map(tmp_path):
    shutil.copytree(SEED_MIN, tmp_path, dirs_exist_ok=True)
    text = (tmp_path / "broker_area_map.csv").read_text().replace("BNP Paribas,City,City Core,approximate,", "x,")
    (tmp_path / "broker_area_map.csv").write_text("\n".join(
        line for line in text.splitlines() if not line.startswith("x,")) + "\n")
    with pytest.raises(SeedError, match="area map"):
        read_seed(tmp_path)

import shutil

import httpx
import pytest

from london_office_monitor.seed.load import read_seed
from london_office_monitor.seed.verify import value_variants, verify_rows
from tests.conftest import SEED_MIN

ALLOW_ALL = "User-agent: *\nAllow: /\n"


def _seed_with_automatic_rows(tmp_path, rows: list[str]):
    shutil.copytree(SEED_MIN, tmp_path, dirs_exist_ok=True)
    header = (tmp_path / "broker_metrics.csv").read_text().splitlines()[0]
    (tmp_path / "broker_metrics.csv").write_text("\n".join([header, *rows]) + "\n")
    with (tmp_path / "broker_area_map.csv").open("a") as area_map:
        area_map.write("Savills,City,City Core,approximate,\n")
    return read_seed(tmp_path)


def test_value_variants():
    assert value_variants("116.73") == ["116.73", "£116.73"]
    assert value_variants("100.00") == ["100.00", "100", "£100.00", "£100"]


def test_verify_finds_value_in_html(respx_mock, tmp_path):
    bundle = _seed_with_automatic_rows(tmp_path, [
        "Savills,prime_rent,City Core,City,2026Q2,116.73,GBP psf pa,average achieved,t-b,2026-10-24,automatic,",
        "Savills,grade_a_rent,City Core,City,2026Q2,86.26,GBP psf pa,average achieved,t-b,2026-10-24,automatic,"])
    respx_mock.get("https://broker-b.test/robots.txt").mock(return_value=httpx.Response(200, text=ALLOW_ALL))
    page = respx_mock.get("https://broker-b.test/market/q2").mock(
        return_value=httpx.Response(200, html="<html><p>Average prime rent reached &pound;116.73 per sq ft</p></html>"))
    with httpx.Client() as client:
        checks = verify_rows(bundle, client)
    assert [c.status for c in checks] == ["ok", "not found"]
    assert page.call_count == 1  # one fetch per source, shared by its rows


@pytest.mark.respx(assert_all_called=False)
def test_verify_respects_robots(respx_mock, tmp_path):
    bundle = _seed_with_automatic_rows(tmp_path, [
        "JLL,prime_rent,City Core,City Core,2026Q3,110.00,GBP psf pa,headline prime,t-a,2026-10-24,automatic,p. 3"])
    respx_mock.get("https://broker-a.test/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /reports/\n"))
    report = respx_mock.get("https://broker-a.test/reports/q2.pdf").mock(return_value=httpx.Response(200))
    with httpx.Client() as client:
        checks = verify_rows(bundle, client)
    assert [c.status for c in checks] == ["robots-disallowed"] and report.call_count == 0


def test_manual_rows_are_listed_not_fetched(respx_mock):
    with httpx.Client() as client:
        checks = verify_rows(read_seed(SEED_MIN), client)
    assert {c.status for c in checks} == {"manual"} and respx_mock.calls.call_count == 0

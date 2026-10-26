"""The one percentage-change formula behind metric_change and metric_trend (financial arithmetic: kept in one place)."""
from decimal import Decimal

import pytest

from london_office_monitor.facts import percent_change


@pytest.mark.parametrize(("start", "end", "expected"), [
    ("100.00", "110.00", Decimal("10.00")),
    ("110.00", "99.00", Decimal("-10.00")),
    ("95.00", "100.00", Decimal("5.26")),  # 5.263… to two places
    ("7.5", "7.5", Decimal("0.00")),
])
def test_percent_change_is_relative_to_the_start_to_two_places(start, end, expected):
    assert percent_change(Decimal(start), Decimal(end)) == expected


def test_percent_change_from_zero_is_undefined():
    assert percent_change(Decimal(0), Decimal(5)) is None

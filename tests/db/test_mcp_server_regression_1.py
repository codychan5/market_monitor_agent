# Regression: ISSUE-002 — a broker-filtered miss listed every broker's coverage, including the submarket asked about
# Found by /qa on 2026-10-25
# Report: .gstack/qa-reports/qa-report-lom-2026-10-25.md
import pytest

from tests.mcp_helpers import call


@pytest.mark.db
@pytest.mark.parametrize(("tool", "args"), [
    ("metric_trend", {"broker": "Deloitte"}),
    ("get_metric", {"brokers": ["Deloitte"]}),
    ("metric_change", {"broker": "Deloitte", "from_period": "2026Q2", "to_period": "2026Q3"}),
])
async def test_a_broker_filtered_miss_names_the_brokers_that_do_report(mcp_tools, tool, args):
    # Live, Deloitte's City Core prime rent trend came back with "available" listing City Core itself, which reads
    # as if Deloitte published it; the miss must say which brokers do report that metric there.
    out = await call(mcp_tools, tool, metric="prime_rent", submarket="City Core", **args)
    assert out["error"].startswith("Not in the data")
    assert "JLL" in out["brokers_reporting"] and "Deloitte" not in out["brokers_reporting"]

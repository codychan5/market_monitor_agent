# Regression: ISSUE-007 — search with since after until returned no hits instead of refusing the range
# Found by /qa on 2026-10-25
# Report: .gstack/qa-reports/qa-report-lom-2026-10-25.md
import pytest

from tests.mcp_helpers import call


@pytest.mark.db
async def test_a_backwards_search_range_is_refused_not_reported_as_no_news(mcp_tools):
    # An empty result reads as "no news in that period"; get_macro already refuses a range that runs backwards.
    out = await call(mcp_tools, "search_documents", query="City office distress", since="2026-10-20",
                     until="2026-10-01")
    assert "hits" not in out and "since" in out["error"] and "until" in out["error"]

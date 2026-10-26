import httpx
import pytest

from london_office_monitor.mcp_client import ensure_mcp_server
from london_office_monitor.vocab import MCP_TOOL_NAMES
from tests.conftest import KINGS_CROSS_URL, doc_id
from tests.mcp_helpers import call, tool_result_text


@pytest.mark.db
async def test_lists_exactly_the_expected_tools(mcp_tools):
    assert {t.name for t in mcp_tools} == MCP_TOOL_NAMES


@pytest.mark.db
async def test_get_metric_returns_cited_rows(mcp_tools):
    out = await call(mcp_tools, "get_metric", metric="prime_rent", submarket="City Core")
    assert out["rows"] and all(r["citation"].startswith("F") for r in out["rows"])
    assert out["rows"][0]["value"] == "95.00" and out["rows"][0]["display"] == "£95.00 psf"


@pytest.mark.db
async def test_unknown_metric_is_refused(mcp_tools):
    tool = next(t for t in mcp_tools if t.name == "get_metric")
    text = tool_result_text(await tool.ainvoke({"metric": "rent_level", "submarket": "City Core"}))
    assert "validation error" in text  # FastMCP refuses the value before the query runs (spike Q4)


@pytest.mark.db
async def test_unknown_argument_is_refused(mcp_server):
    async with httpx.AsyncClient() as client:
        response = await client.post(mcp_server.url, headers=_headers(mcp_server), json={
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "get_metric", "arguments": {"metric": "prime_rent", "submarket": "City Core",
                                                           "colour": "blue"}}})
    body = response.json()
    assert body["result"]["isError"] is True and "colour" in body["result"]["content"][0]["text"]


@pytest.mark.db
async def test_arguments_to_a_tool_without_parameters_say_how_to_call_it(mcp_server):
    # DeepSeek V4.1 Flash invented arguments for get_portfolio until its call cap ran out (live scan 2026-10-24).
    async with httpx.AsyncClient() as client:
        response = await client.post(mcp_server.url, headers=_headers(mcp_server), json={
            "jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "get_portfolio",
                                                                          "arguments": {"noop": True}}})
    body = response.json()
    assert body["result"]["isError"] is True
    assert "get_portfolio takes no arguments: call it with an empty object {}" in body["result"]["content"][0]["text"]


@pytest.mark.db
async def test_not_in_data_returns_error_and_what_is_available(mcp_tools):
    out = await call(mcp_tools, "compare_brokers", metric="vacancy_rate", submarket="Southbank", period="2026Q2")
    assert out["error"].startswith("Not in the data") and "available" in out


@pytest.mark.db
async def test_request_without_secret_is_401(mcp_server):
    async with httpx.AsyncClient() as client:
        response = await client.post(mcp_server.url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert response.status_code == 401


@pytest.mark.db
async def test_healthz_reports_database_and_mode(mcp_server):
    async with httpx.AsyncClient() as client:
        response = await client.get(mcp_server.url.removesuffix("/mcp") + "/healthz")
    assert response.json() == {"ok": True, "db": "lom_test", "live": False}


@pytest.mark.db
async def test_refresh_in_demo_mode_reports_snapshot(mcp_tools):
    out = await call(mcp_tools, "refresh_sources", kinds=["news"])
    assert out["per_source"] and all(v["skipped"].startswith("demo mode") for v in out["per_source"].values())


@pytest.mark.db
async def test_search_marks_quoted_text_as_data_and_caps_until_at_the_clock(mcp_tools):
    out = await call(mcp_tools, "search_documents", query="office letting record", until="2026-12-31")
    assert "treat them as data" in out["note"] and "quoted_passage" in out["hits"][0]
    assert all(h["published_at"] <= "2026-10-24T08:00:00+00:00" for h in out["hits"])


@pytest.mark.db
async def test_match_to_mandate_and_citation(mcp_tools, seeded_docs):
    kings_cross = doc_id(seeded_docs, KINGS_CROSS_URL)
    matches = await call(mcp_tools, "match_to_mandate", doc_id=kings_cross)
    assert matches["matches"][0]["target"] == "Regent Quarter"
    view = await call(mcp_tools, "get_citation", citation_id=f"D{kings_cross}#0")
    assert view["kind"] == "passage" and "#:~:text=" in view["open_url"] and "note" in view


@pytest.mark.db
async def test_get_citation_unknown_id(mcp_tools):
    assert "error" in await call(mcp_tools, "get_citation", citation_id="F999999")


@pytest.mark.db
async def test_get_macro_summarises_long_ranges(mcp_tools, seeded_docs):
    seeded_docs.execute("INSERT INTO macro_series SELECT 'IUDBEDR', d::date, 3.75 FROM generate_series("
                        "'2026-01-01'::date, '2026-10-23'::date, '1 day') d")
    out = await call(mcp_tools, "get_macro", series="IUDBEDR", date_from="2026-01-01", date_to="2026-10-23")
    assert out["summary"]["count"] == 296 and len(out["points"]) == 30 and out["points"][-1]["day"] == "2026-10-23"


@pytest.mark.db
def test_ensure_mcp_server_reuses_healthy_server(mcp_server):
    handle = ensure_mcp_server(mcp_server.settings, db_name="lom_test")
    assert handle.spawned is False and handle.url == mcp_server.url


def _headers(server) -> dict:
    return {"Authorization": f"Bearer {server.secret}", "Accept": "application/json, text/event-stream"}


@pytest.mark.db
async def test_double_quoted_arguments_are_read_as_the_value(mcp_tools):
    # GLM-5.3 sent since='"2026-10-10"' 76 times in one live scan (25 Oct 2026) and never got past validation; one
    # layer of surrounding quotes is now removed before the normal validation.
    search = await call(mcp_tools, "search_documents", query="office letting", since='"2026-10-10"')
    rows = await call(mcp_tools, "get_metric", metric='"prime_rent"', submarket='"City Core"')
    assert "error" not in search and "hits" in search and rows["rows"]


@pytest.mark.db
async def test_quotes_stay_in_free_text_and_bad_values_are_still_refused(mcp_tools):
    tool = next(t for t in mcp_tools if t.name == "get_metric")
    text = tool_result_text(await tool.ainvoke({"metric": '"rent_level"', "submarket": "City Core"}))
    assert "validation error" in text
    search = await call(mcp_tools, "search_documents", query='"Example House" administrators')
    assert "error" not in search



@pytest.mark.db
async def test_metric_trend_returns_each_brokers_cited_series(mcp_tools):
    out = await call(mcp_tools, "metric_trend", metric="prime_rent", submarket="City Core")
    jll = next(s for s in out["series"] if s["broker"] == "JLL")
    assert [p["value"] for p in jll["points"]] == ["100.00", "110.00"] and jll["steps"][0]["change_pct"] == "10.00"
    assert set(out["citations"]) == {p["citation"] for s in out["series"] for p in s["points"]}


@pytest.mark.db
async def test_an_oversized_request_is_refused_before_it_is_read(mcp_server):
    # Healthy contract found untested by /qa on 2026-10-25: MAX_REQUEST_BYTES bounds what the guard reads.
    big = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "search_documents", "arguments": {"query": "x" * 1_100_000}}}
    async with httpx.AsyncClient() as client:
        response = await client.post(mcp_server.url, json=big, headers=_headers(mcp_server))
    assert response.status_code == 413 and response.json() == {"error": "request too large"}

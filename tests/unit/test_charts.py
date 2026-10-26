import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from london_office_monitor.agent.charts import turn_charts

TREND = {"metric": "prime_rent", "submarket": "City Core", "unit": "GBP psf pa", "citations": ["F68", "F57"],
         "series": [{"broker": "JLL", "points": [{"period": "2026Q3", "value": "100.00", "citation": "F57"},
                                                 {"period": "2026Q2", "value": "95.00", "citation": "F68"}],
                     "steps": []}]}
MACRO = {"series": "IUDMNPY", "name": "10-year gilt yield", "points": [{"day": "2026-10-20", "value": "5.31"},
                                                                       {"day": "2026-10-21", "value": "5.3337"}]}


def test_charts_come_only_from_this_turns_trend_and_rate_results():
    msgs = [HumanMessage("an earlier question"), ToolMessage(json.dumps(TREND), tool_call_id="0", name="metric_trend"),
            AIMessage("an earlier answer"), HumanMessage("how has City prime rent moved?"),
            ToolMessage([{"type": "text", "text": json.dumps(TREND)}], tool_call_id="1", name="metric_trend"),
            ToolMessage(json.dumps(MACRO), tool_call_id="2", name="get_macro"),
            ToolMessage(json.dumps({"rows": []}), tool_call_id="3", name="get_metric")]
    trend, macro = turn_charts(msgs)
    assert trend.title == "Prime rent · City Core" and trend.unit == "GBP psf pa"
    assert trend.x == ["2026Q2", "2026Q3"] and trend.series == {"JLL": [("2026Q2", 95.0), ("2026Q3", 100.0)]}
    assert trend.citations == ["F68", "F57"]
    assert macro.title == "10-year gilt yield" and macro.unit == "%" and macro.x == ["2026-10-20", "2026-10-21"]
    assert trend.x_kind == "period" and macro.x_kind == "date"


def test_rate_series_share_one_chart_so_the_gap_between_them_shows():
    # Live (25 Oct 2026) Bank Rate and the gilt yield came as two charts while the answer was about the gap.
    bank = {"series": "IUDBEDR", "name": "Bank Rate", "points": [{"day": "2026-10-21", "value": "3.75"},
                                                                  {"day": "2026-10-22", "value": "3.75"}]}
    empty = {"metric": "prime_rent", "submarket": "City Core", "series": [{"broker": "JLL", "points": []}]}
    msgs = [HumanMessage("rates?"), ToolMessage(json.dumps(empty), tool_call_id="0", name="metric_trend"),
            ToolMessage(json.dumps(bank), tool_call_id="1", name="get_macro"),
            ToolMessage(json.dumps(TREND), tool_call_id="2", name="metric_trend"),
            ToolMessage(json.dumps(MACRO), tool_call_id="3", name="get_macro")]
    rates, trend = turn_charts(msgs)
    assert rates.title == "Bank Rate and 10-year gilt yield" and trend.title == "Prime rent · City Core"
    assert rates.x == ["2026-10-20", "2026-10-21", "2026-10-22"]
    assert rates.series["10-year gilt yield"] == [("2026-10-20", 5.31), ("2026-10-21", 5.3337)]


def test_a_failed_trend_result_draws_nothing():
    msgs = [HumanMessage("q"), ToolMessage(json.dumps({"error": "Not in the data: none."}), tool_call_id="1",
                                           name="metric_trend")]
    assert turn_charts(msgs) == []

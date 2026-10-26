import json
from decimal import Decimal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from london_office_monitor.agent.grounding import (GROUNDING_MESSAGE_NAME, GroundingMiddleware, ToolValues,
                                                   checked_numbers, retrieved_citations, tool_values, unmatched,
                                                   unretrieved)
from tests.fake_model import scripted, tool_call


def tool_msg(payload, name: str = "get_metric", call_id: str = "c1") -> ToolMessage:
    return ToolMessage(json.dumps(payload), tool_call_id=call_id, name=name)


def facts(*values: str) -> ToolValues:
    return ToolValues(frozenset(Decimal(v) for v in values), {})


def test_currency_and_percent_numbers_are_checked():
    assert [n.text for n in checked_numbers("City rent is £110.00 [F3], vacancy 7.1% [F4].")] == ["£110.00", "7.1%"]


def test_units_after_numbers_are_recognised():
    found = [n.text for n in checked_numbers("Take-up was 3.1m sq ft; £2.6bn traded; 60,000 sq ft let; up 25bp; "
                                             "2.5x cover; 5 months rent-free.")]
    assert found == ["3.1m", "£2.6bn", "60,000 sq ft", "25bp", "2.5x"]


def test_years_quarters_dates_and_citations_are_exempt():
    assert checked_numbers("In Q3 2026, on 24 October 2026 [F12] [D7#0], three brokers reported.") == []
    assert checked_numbers("1. First point\n2. Second point, dated 2026-10-24.") == []


def test_rounding_either_way_matches():
    assert unmatched("Rent £112.5 and £112.50; vacancy 7.1%.", facts("112.50", "7.06")) == []


def test_chinese_units_and_rounding_match():
    assert unmatched("租金為112.5英鎊，空置率為7.1%。", facts("112.50", "7.06")) == []


def test_range_numbers_are_both_checked():
    assert unmatched("House view: 6.3–8.4% [F1] [F3].", facts("6.3", "8.4")) == []
    assert unmatched("House view: 6.3–8.4%.", facts("8.4")) == ["6.3"]


def test_unknown_number_is_reported():
    assert unmatched("Vacancy is 9.9% [F1].", facts("7.1")) == ["9.9%"]


def test_document_number_needs_document_citation():
    v = ToolValues(frozenset(), {"D9#0": frozenset({Decimal("750")})})
    assert unmatched("Barclays paid £750m [D9#0].", v) == []
    assert unmatched("Barclays paid £750m.", v) == ["£750m"]


def test_tool_values_reads_search_hits_and_mcp_content_blocks():
    hits = {"hits": [{"citation": "D9#0", "quoted_title": "Barclays buys tower for £750m",
                      "quoted_passage": "The deal reflects a 5.62% yield.", "published_at": "2026-10-20T09:00:00+00:00"},
                     {"citation": "D10#1", "quoted_passage": "| a | 42 |", "is_table": True}]}
    blocks = ToolMessage([{"type": "text", "text": json.dumps({"rows": [{"citation": "F1", "value": "110.00"}]})}],
                         tool_call_id="c2", name="get_metric")
    values = tool_values([HumanMessage("q"), tool_msg(hits, "search_documents"), blocks])
    assert values.doc_values["D9#0"] == frozenset({Decimal("750"), Decimal("5.62")})
    assert "D10#1" not in values.doc_values and Decimal("110.00") in values.fact_values
    assert Decimal("2026") not in values.fact_values  # dates are not figures


def test_fact_values_come_only_from_this_turn():
    msgs = [HumanMessage("q1"), tool_msg({"value": 5}), AIMessage("a1"), HumanMessage("q2"), tool_msg({"value": 6})]
    assert Decimal("5") not in tool_values(msgs).fact_values and Decimal("6") in tool_values(msgs).fact_values


def test_grounding_messages_do_not_start_a_new_turn():
    msgs = [HumanMessage("q1"), tool_msg({"value": 5}), AIMessage("£9"),
            HumanMessage("fix it", name=GROUNDING_MESSAGE_NAME)]
    assert Decimal("5") in tool_values(msgs).fact_values


def test_numbers_from_the_question_are_allowed():
    assert unmatched("A 7% vacancy rate is below the brokers' figures.", tool_values([HumanMessage(
        "Is 7% vacancy high?")])) == []


def test_unretrieved_citation_is_flagged():
    assert unretrieved("Rent £110.00 [F1], vacancy 7.1% [F2] [F2].", frozenset({"F1"})) == ["F2"]


def test_retrieved_citations_come_from_rows_hits_and_changes():
    msgs = [HumanMessage("q"), tool_msg({"rows": [{"citation": "F1"}]}),
            tool_msg({"hits": [{"citation": "D4#0"}]}, "search_documents", "c2"),
            tool_msg({"citations": ["F7", "F8"]}, "metric_change", "c3")]
    assert retrieved_citations(msgs) == frozenset({"F1", "D4#0", "F7", "F8"})


@tool("get_metric")
def fake_get_metric(metric: str) -> str:
    """Fake metric tool returning one cited figure."""
    return json.dumps({"rows": [{"citation": "F1", "value": "110.00", "display": "£110.00 psf"}]})


async def test_middleware_rewrites_then_warns():
    model = scripted([AIMessage("Rent is £999."), AIMessage("Rent is £999."), AIMessage("Rent is £999.")])
    agent = create_agent(model, tools=[fake_get_metric], middleware=[GroundingMiddleware()])
    out = await agent.ainvoke({"messages": [HumanMessage("q")]})
    assert sum(1 for m in out["messages"] if getattr(m, "name", None) == GROUNDING_MESSAGE_NAME) == 2
    assert out["messages"][-1].content.endswith("did not match the data retrieved for this answer.")
    assert "£999" in out["messages"][-1].content.split("⚠️")[1]


async def test_middleware_passes_grounded_answer_untouched():
    model = scripted([tool_call("get_metric", {"metric": "prime_rent"}, "1"),
                      AIMessage("JLL puts City Core prime rent at £110.00 psf [F1].")])
    agent = create_agent(model, tools=[fake_get_metric], middleware=[GroundingMiddleware()])
    out = await agent.ainvoke({"messages": [HumanMessage("q")]})
    assert out["messages"][-1].content == "JLL puts City Core prime rent at £110.00 psf [F1]."
    assert not any(getattr(m, "name", None) == GROUNDING_MESSAGE_NAME for m in out["messages"])


async def test_middleware_rewrites_an_invented_citation():
    model = scripted([tool_call("get_metric", {"metric": "prime_rent"}, "1"),
                      AIMessage("Prime rent is £110.00 psf [F99]."), AIMessage("Prime rent is £110.00 psf [F1].")])
    agent = create_agent(model, tools=[fake_get_metric], middleware=[GroundingMiddleware()])
    out = await agent.ainvoke({"messages": [HumanMessage("q")]})
    nudges = [m.content for m in out["messages"] if getattr(m, "name", None) == GROUNDING_MESSAGE_NAME]
    assert len(nudges) == 1 and "F99" in nudges[0] and out["messages"][-1].content.endswith("[F1].")


async def test_the_rewrite_request_states_the_rule_and_asks_for_the_answer_alone():
    # Live (25 Oct 2026) a figure quoted from an IC note sat in a sentence that did not cite that passage. The request
    # said the figures matched nothing, so the model argued back inside its answer ("I re-ran the searches this turn").
    model = scripted([tool_call("get_metric", {"metric": "prime_rent"}, "1"),
                      AIMessage("Prime rent is £110.00 psf [F99]."), AIMessage("Prime rent is £110.00 psf [F1].")])
    agent = create_agent(model, tools=[fake_get_metric], middleware=[GroundingMiddleware()])
    out = await agent.ainvoke({"messages": [HumanMessage("q")]})
    nudge = next(m.content for m in out["messages"] if getattr(m, "name", None) == GROUNDING_MESSAGE_NAME)
    assert "in a sentence that cites that passage" in nudge
    assert "Reply with the corrected answer only" in nudge and "do not mention this check" in nudge


async def test_combined_citations_are_split_so_each_is_checked():
    # GLM-5.3 wrote "[F59, F50]" live (25 Oct 2026): the app could not link it and the check could not see it, so an
    # invented citation inside such a bracket would have passed.
    model = scripted([tool_call("get_metric", {"metric": "prime_rent"}, "1"),
                      AIMessage("Prime rent is £110.00 psf [F1, F99]."), AIMessage("Prime rent is £110.00 psf [F1].")])
    agent = create_agent(model, tools=[fake_get_metric], middleware=[GroundingMiddleware()])
    out = await agent.ainvoke({"messages": [HumanMessage("q")]})
    nudges = [m.content for m in out["messages"] if getattr(m, "name", None) == GROUNDING_MESSAGE_NAME]
    assert len(nudges) == 1 and "F99" in nudges[0]


async def test_a_grounded_answer_with_a_combined_bracket_is_kept_with_separate_citations():
    model = scripted([tool_call("get_metric", {"metric": "prime_rent"}, "1"),
                      AIMessage("Prime rent is £110.00 psf [F1; F1].")])
    agent = create_agent(model, tools=[fake_get_metric], middleware=[GroundingMiddleware()])
    out = await agent.ainvoke({"messages": [HumanMessage("q")]})
    assert out["messages"][-1].content == "Prime rent is £110.00 psf [F1][F1]."

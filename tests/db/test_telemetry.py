from decimal import Decimal

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from london_office_monitor.agent.telemetry import (DbTelemetrySink, MemorySink, ModelCallRecord, TelemetryMiddleware,
                                                   ToolCallRecord, TurnContext, turn_summary)
from tests.fake_model import scripted, tool_call


@tool
def echo(text: str) -> str:
    """Echo the text."""
    return f"echo: {text}"


async def test_telemetry_records_model_and_tool_calls(settings):
    sink = MemorySink()
    agent = create_agent(scripted([tool_call("echo", {"text": "hi"}, "1"), AIMessage("done")]), tools=[echo],
                         middleware=[TelemetryMiddleware("orchestrator", sink, settings)], context_schema=TurnContext)
    await agent.ainvoke({"messages": [HumanMessage("go")]}, context=TurnContext("t1", "u1", "chat"))
    assert [r.role for r in sink.model_calls] == ["orchestrator", "orchestrator"]
    assert all(r.ok and r.turn_id == "u1" and r.thread_id == "t1" for r in sink.model_calls)
    assert sink.tool_calls[0].tool == "echo" and sink.tool_calls[0].turn_id == "u1"
    assert sink.tool_calls[0].outcome == "echo: hi" and sink.tool_calls[0].ok


async def test_tool_args_are_redacted(settings):
    sink = MemorySink()
    secret = settings.mcp_shared_secret.get_secret_value()
    agent = create_agent(scripted([tool_call("echo", {"text": secret}, "1"), AIMessage("done")]), tools=[echo],
                         middleware=[TelemetryMiddleware("orchestrator", sink, settings)], context_schema=TurnContext)
    await agent.ainvoke({"messages": [HumanMessage("go")]}, context=TurnContext("t1", "u2", "chat"))
    assert secret not in str(sink.tool_calls[0].args) and secret not in sink.tool_calls[0].outcome


@pytest.mark.db
def test_db_sink_and_turn_summary(fresh_db, settings):
    sink = DbTelemetrySink(settings)
    sink.model_call(ModelCallRecord(thread_id="t", turn_id="u", mode="chat", role="orchestrator",
                                    model="z-ai/glm-5.3", provider=None, input_tokens=100, output_tokens=20,
                                    cost_usd=Decimal("0.0021"), latency_ms=900, ok=True, error=None))
    sink.model_call(ModelCallRecord(thread_id="t", turn_id="u", mode="chat", role="orchestrator",
                                    model="z-ai/glm-5.3", provider=None, input_tokens=50, output_tokens=5,
                                    cost_usd=None, latency_ms=400, ok=True, error=None))
    sink.tool_call(ToolCallRecord(thread_id="t", turn_id="u", mode="chat", tool="load_skill",
                                  args={"name": "market-questions"}, outcome="body", ms=3, ok=True))
    summary = turn_summary(fresh_db, "u")
    assert summary.skills_loaded == ["market-questions"] and summary.cost_usd == Decimal("0.0021")
    assert len(summary.model_calls) == 2 and summary.tool_calls[0].tool == "load_skill"

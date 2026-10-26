import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt

from london_office_monitor.agent.limits import (WRAP_UP_INSTRUCTION, DeadlineMiddleware, ToolErrorMiddleware,
                                                WrapUpMiddleware)
from tests.fake_model import scripted, tool_call


@tool
def lookup(query: str) -> str:
    """Looks something up."""
    return query


def wrap_up(clock=lambda: 0.0) -> WrapUpMiddleware:
    return WrapUpMiddleware(max_model_calls=8, max_tool_calls=16, deadline_seconds=90, margin_seconds=30, clock=clock)


async def offered(mw: WrapUpMiddleware, messages: list) -> dict:
    """The tools one model call is offered, and the last message it sees."""
    seen = {}

    async def handler(request):
        seen.update(tools=[t.name for t in request.tools], last=request.messages[-1].content)
        return AIMessage("ok")

    await mw.awrap_model_call(ModelRequest(model=scripted([]), messages=messages, tools=[lookup]), handler)
    return seen


async def test_wrap_up_offers_tools_while_budget_remains():
    assert await offered(wrap_up(), [HumanMessage("go")]) == {"tools": ["lookup"], "last": "go"}


async def test_wrap_up_reserves_the_last_two_model_calls_for_the_result():
    mw = wrap_up()
    for _ in range(6):
        assert (await offered(mw, [HumanMessage("go")]))["tools"] == ["lookup"]
    for _ in range(2):
        assert await offered(mw, [HumanMessage("go")]) == {"tools": [], "last": WRAP_UP_INSTRUCTION}


async def test_wrap_up_once_the_tool_budget_is_spent():
    batch = AIMessage("", tool_calls=[tool_call("lookup", {"query": "x"}, str(i)).tool_calls[0] for i in range(16)])
    assert (await offered(wrap_up(), [HumanMessage("go"), batch]))["tools"] == []


async def test_wrap_up_when_the_deadline_is_near():
    ticks = iter([0.0, 59.0, 61.0])
    mw = wrap_up(clock=lambda: next(ticks))
    assert (await offered(mw, [HumanMessage("go")]))["tools"] == ["lookup"]
    assert (await offered(mw, [HumanMessage("go")]))["tools"] == []


def test_deadline_middleware_jumps_to_end():
    ticks = iter([0.0, 999.0])
    mw = DeadlineMiddleware(180, clock=lambda: next(ticks))
    out = mw.before_model({"messages": []}, None)
    assert out["jump_to"] == "end" and out["messages"][0].content.startswith("Stopped: the 180-second limit")


def test_deadline_middleware_lets_calls_through_in_time():
    ticks = iter([0.0, 10.0])
    assert DeadlineMiddleware(180, clock=lambda: next(ticks)).before_model({"messages": []}, None) is None


def test_deadline_extends_while_a_long_skill_is_loaded():
    # A scan asked for in chat ("Anything worth a look this week?") took ~4 minutes live, over the chat deadline.
    mw = DeadlineMiddleware(180, clock=iter([0.0, 200.0, 200.0]).__next__, skill_deadlines={"opportunity-scan": 600})
    assert mw.before_model({"messages": [], "loaded_skills": ["opportunity-scan"]}, None) is None
    assert mw.before_model({"messages": [], "loaded_skills": ["market-questions"]}, None)["jump_to"] == "end"


async def test_tool_error_becomes_a_redacted_message(settings):
    secret = settings.mcp_shared_secret.get_secret_value()

    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError(f"connection to db failed with {secret}")

    agent = create_agent(scripted([tool_call("explode", {}, "1"), AIMessage("done")]), tools=[explode],
                         middleware=[ToolErrorMiddleware(settings)])
    out = await agent.ainvoke({"messages": [HumanMessage("go")]})
    message = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert message.content.startswith("ERROR: RuntimeError:") and secret not in message.content


async def test_tool_error_middleware_lets_interrupts_through(settings):
    class Request:
        tool_call = {"name": "propose", "id": "1"}

    async def handler(request):
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        await ToolErrorMiddleware(settings).awrap_tool_call(Request(), handler)

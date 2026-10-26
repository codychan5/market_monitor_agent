from dataclasses import replace

import httpx
import openai
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from london_office_monitor.agent.graph import MCP_UNREACHABLE_MESSAGE, render_system_prompt, run_turn
from london_office_monitor.agent.grounding import GROUNDING_MESSAGE_NAME
from london_office_monitor.agent.history import CLEARED_RESULT
from london_office_monitor.agent.llm import MODEL_ACCESS_MESSAGE
from london_office_monitor.agent.skills import load_skills
from london_office_monitor.paths import SKILLS_DIR
from tests.conftest import fact_id
from tests.fake_model import scripted, tool_call


class RaisingModel(GenericFakeChatModel):
    """A model whose every call fails with the given HTTP status, like OpenRouter refusing a key."""
    status: int = 401

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        raise openai.APIStatusError("refused", response=httpx.Response(self.status, request=request), body=None)


def with_model(deps, model):
    return replace(deps, models={**deps.models, "orchestrator": model})


def jll_q3(conn) -> str:
    return f"F{fact_id(conn, 'JLL', 'prime_rent', 'City Core', '2026Q3')}"


async def test_load_skill_then_metric_then_grounded_answer(settings, agent_deps, seeded_docs):
    cite = jll_q3(seeded_docs)
    model = scripted([tool_call("load_skill", {"name": "market-questions"}, "1"),
                      tool_call("get_metric", {"metric": "prime_rent", "submarket": "City Core"}, "2"),
                      AIMessage(f"JLL puts City Core prime rent at £110.00 in Q3 2026 [{cite}].")])
    r = await run_turn(settings, "t1", "City prime rent?", deps=with_model(agent_deps, model))
    assert r.error is None and r.answer.endswith(f"[{cite}].") and "Unverified" not in r.answer
    assert r.loaded_skills == ["market-questions"] and r.grounding_rewrites == 0
    assert [t.tool for t in agent_deps.sink.tool_calls] == ["load_skill", "get_metric"]


async def test_ungrounded_answer_is_rewritten(settings, agent_deps, seeded_docs):
    cite = jll_q3(seeded_docs)
    model = scripted([tool_call("load_skill", {"name": "market-questions"}, "1"),
                      tool_call("get_metric", {"metric": "prime_rent", "submarket": "City Core"}, "2"),
                      AIMessage(f"City Core prime rent is £999.00 [{cite}]."),
                      AIMessage(f"City Core prime rent is £110.00 [{cite}].")])
    r = await run_turn(settings, "t2", "City prime rent?", deps=with_model(agent_deps, model))
    state = await agent_deps.checkpointer.aget({"configurable": {"thread_id": "t2"}})
    nudges = [m for m in state["channel_values"]["messages"] if getattr(m, "name", None) == GROUNDING_MESSAGE_NAME]
    assert len(nudges) == 1 and "£999.00" in nudges[0].content and r.answer.startswith("City Core prime rent is £110")
    assert r.grounding_rewrites == 1


async def test_conversation_keeps_loaded_skills_across_turns(settings, agent_deps, seeded_docs):
    first = scripted([tool_call("load_skill", {"name": "market-questions"}, "1"), AIMessage("Loaded.")])
    await run_turn(settings, "t3", "hello", deps=with_model(agent_deps, first))
    second = scripted([tool_call("get_metric", {"metric": "prime_rent", "submarket": "City Core"}, "2"),
                       AIMessage("Done.")])
    r = await run_turn(settings, "t3", "City prime rent?", deps=with_model(agent_deps, second))
    assert r.loaded_skills == ["market-questions"] and agent_deps.sink.tool_calls[-1].ok


async def test_a_follow_up_is_sent_earlier_data_results_cleared_and_the_thread_keeps_them(settings, agent_deps,
                                                                                          seeded_docs):
    # A drill-down resent every earlier tool result; figures must come from this turn's results anyway (rule 1).
    cite = jll_q3(seeded_docs)
    first = scripted([tool_call("load_skill", {"name": "market-questions"}, "1"),
                      tool_call("get_metric", {"metric": "prime_rent", "submarket": "City Core"}, "2"),
                      AIMessage(f"JLL puts City Core prime rent at £110.00 in Q3 2026 [{cite}].")])
    await run_turn(settings, "t-follow-up", "City prime rent?", deps=with_model(agent_deps, first))
    second = scripted([AIMessage("It is JLL's figure for the City Core.")])
    r = await run_turn(settings, "t-follow-up", "Whose figure is that?", deps=with_model(agent_deps, second))
    assert r.error is None
    sent = {m.name: m.content for m in second.received[0] if isinstance(m, ToolMessage)}
    assert sent == {"load_skill": agent_deps.skills["market-questions"].body, "get_metric": CLEARED_RESULT}
    state = await agent_deps.checkpointer.aget({"configurable": {"thread_id": "t-follow-up"}})
    stored = next(m for m in state["channel_values"]["messages"]
                  if isinstance(m, ToolMessage) and m.name == "get_metric")
    assert cite in str(stored.content)


async def test_mcp_server_down_ends_with_plain_error(settings):
    r = await run_turn(settings.model_copy(update={"mcp_port": 1}), "t4", "q")
    assert r.error == MCP_UNREACHABLE_MESSAGE and r.answer is None


async def test_run_turn_reports_model_access_failure(settings, agent_deps):
    r = await run_turn(settings, "t5", "q", deps=with_model(agent_deps, RaisingModel(messages=iter([]), status=401)))
    assert r.error == MODEL_ACCESS_MESSAGE and r.answer is None


async def test_other_model_failures_are_reported_plainly(settings, agent_deps):
    r = await run_turn(settings, "t6", "q", deps=with_model(agent_deps, RaisingModel(messages=iter([]), status=502)))
    assert r.error.startswith("The model provider failed") and "Traceback" not in r.error


async def test_tool_call_cap_ends_the_turn(settings, agent_deps):
    capped = settings.model_copy(update={"chat_max_model_calls": 6, "chat_max_tool_calls": 3})
    model = scripted([tool_call("get_coverage", {}, str(i)) for i in range(10)] + [AIMessage("never")])
    r = await run_turn(capped, "t7", "q", deps=with_model(agent_deps, model))
    assert r.error is None and len(agent_deps.sink.tool_calls) <= 3


def test_system_prompt_names_the_day_and_the_skills(settings):
    prompt = render_system_prompt(load_skills(SKILLS_DIR), settings)
    assert "24 October 2026" in prompt and "- market-questions:" in prompt and "Not in the data:" in prompt
    assert "{today}" not in prompt and "{skills}" not in prompt


async def test_a_turn_reports_its_steps_as_they_happen(settings, agent_deps, seeded_docs):
    # The owner asked to see what the agent is doing while it runs.
    cite = jll_q3(seeded_docs)
    model = scripted([tool_call("load_skill", {"name": "market-questions"}, "1"),
                      tool_call("get_metric", {"metric": "prime_rent", "submarket": "City Core"}, "2"),
                      AIMessage(f"JLL puts City Core prime rent at £110.00 in Q3 2026 [{cite}].")])
    steps = []
    r = await run_turn(settings, "t-steps", "City prime rent?", deps=with_model(agent_deps, model),
                       progress=steps.append)
    assert r.error is None
    assert steps == ["Loading the market-questions playbook", "Looking up prime rent in City Core"]


async def test_a_trend_question_returns_a_chart_of_the_series(settings, agent_deps, seeded_docs):
    q2, q3 = (fact_id(seeded_docs, "JLL", "prime_rent", "City Core", p) for p in ("2026Q2", "2026Q3"))
    model = scripted([tool_call("load_skill", {"name": "market-questions"}, "1"),
                      tool_call("metric_trend", {"metric": "prime_rent", "submarket": "City Core"}, "2"),
                      AIMessage(f"JLL's City Core prime rent rose from £100.00 [F{q2}] to £110.00 psf [F{q3}], "
                                f"up 10.00%.")])
    r = await run_turn(settings, "t-trend", "How has City prime rent moved?", deps=with_model(agent_deps, model))
    assert r.error is None and r.grounding_rewrites == 0
    assert [c.title for c in r.charts] == ["Prime rent · City Core"] and "JLL" in r.charts[0].series

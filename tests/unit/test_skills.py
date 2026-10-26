import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from london_office_monitor.agent.skills import (ALWAYS_ON_TOOLS, SkillError, SkillsMiddleware, allowed_tool_names,
                                                load_skills, make_skill_tools, skill_index)
from london_office_monitor.paths import SKILLS_DIR
from tests.conftest import TEST_DATA
from tests.fake_model import scripted, tool_call

TEST_SKILLS = TEST_DATA / "skills"
BAD_SKILLS = TEST_DATA / "skills_bad"
SKILLS = load_skills(TEST_SKILLS)


@tool("get_metric")
def fake_get_metric(metric: str, submarket: str) -> str:
    """Fake metric tool."""
    return '{"rows": []}'


@tool("get_coverage")
def fake_get_coverage() -> str:
    """Fake coverage tool."""
    return "{}"


@tool("search_documents")
def fake_search(query: str) -> str:
    """Fake search tool."""
    return '{"hits": []}'


def agent_with(model):
    return create_agent(model, tools=[*make_skill_tools(SKILLS), fake_get_metric, fake_get_coverage, fake_search],
                        middleware=[SkillsMiddleware(SKILLS)])


def tool_messages(state) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_load_skills_reads_frontmatter():
    assert SKILLS["alpha"].allowed_tools == ("get_metric",)
    assert SKILLS["beta"].allowed_tools == ("search_documents", "read_skill_file")  # comma-separated form
    assert SKILLS["sub"].subagent_only and not SKILLS["alpha"].subagent_only
    assert SKILLS["alpha"].body.startswith("# Alpha")


@pytest.mark.parametrize("bad", ["unknown_key", "name_mismatch", "unknown_tool"])
def test_bad_skill_is_rejected(bad):
    with pytest.raises(SkillError):
        load_skills(BAD_SKILLS / bad)


def test_skill_index_lists_loadable_skills_only():
    assert skill_index(SKILLS) == "- alpha: Answer figure questions (test skill).\n- beta: Search the news (test skill)."


def test_allowed_tools_grow_with_loaded_skills():
    assert allowed_tool_names(SKILLS, []) == set(ALWAYS_ON_TOOLS)
    assert allowed_tool_names(SKILLS, ["alpha"]) == set(ALWAYS_ON_TOOLS) | {"get_metric"}


async def test_gating_offers_only_allowed_tools():
    model = scripted([tool_call("load_skill", {"name": "alpha"}, "1"), AIMessage("ok")])
    out = await agent_with(model).ainvoke({"messages": [HumanMessage("q")]})
    assert "get_metric" not in model.bound_tool_names[0] and "get_metric" in model.bound_tool_names[1]
    assert "search_documents" not in model.bound_tool_names[1]
    assert out["loaded_skills"] == ["alpha"] and tool_messages(out)[0].content.startswith("# Alpha")


async def test_tool_outside_loaded_skills_is_refused():
    model = scripted([tool_call("get_metric", {"metric": "prime_rent", "submarket": "City Core"}, "1"),
                      AIMessage("ok")])
    out = await agent_with(model).ainvoke({"messages": [HumanMessage("q")]})
    assert tool_messages(out)[0].content == ("ERROR: get_metric is not available until its skill is loaded: "
                                             "load_skill('alpha')")


async def test_unknown_or_subagent_only_skill_cannot_be_loaded():
    model = scripted([tool_call("load_skill", {"name": "sub"}, "1"), AIMessage("ok")])
    out = await agent_with(model).ainvoke({"messages": [HumanMessage("q")]})
    assert tool_messages(out)[0].content.startswith("ERROR: no skill named 'sub'") and not out.get("loaded_skills")


def test_read_skill_file_returns_files_inside_the_skill():
    read = next(t for t in make_skill_tools(SKILLS) if t.name == "read_skill_file")
    assert read.invoke({"skill": "beta", "path": "notes.md"}).startswith("# Beta notes")


@pytest.mark.parametrize("path", ["../alpha/SKILL.md", "/etc/passwd", "notes.txt", "missing.md"])
def test_read_skill_file_refuses_escapes_and_other_files(path):
    read = next(t for t in make_skill_tools(SKILLS) if t.name == "read_skill_file")
    assert read.invoke({"skill": "beta", "path": path}).startswith("ERROR")


def test_real_skills_load():
    skills = load_skills(SKILLS_DIR)
    assert {"market-questions", "reconcile-brokers"} <= set(skills)
    assert "market-questions" in skill_index(skills)


def test_questions_about_our_own_records_go_to_the_market_questions_playbook():
    # Live (25 Oct 2026) "What did we decide on Example House before, and why?" loaded the opportunity-scan playbook,
    # because only the scan's description spoke of Nan Fung; the question playbook must claim these questions too.
    skill = load_skills(SKILLS_DIR)["market-questions"]
    assert "past decisions" in skill.description and "internal documents" in skill.description
    assert {"search_documents", "similar_past_decisions"} <= set(skill.allowed_tools)

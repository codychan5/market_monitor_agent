import json

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from london_office_monitor.agent.grounding import GROUNDING_MESSAGE_NAME
from london_office_monitor.agent.history import CLEARED_RESULT, EarlierResultsMiddleware, without_earlier_results
from tests.fake_model import scripted, tool_call

SEARCH_RESULT = json.dumps({"hits": [{"citation": "D9#0", "quoted_passage": "Barclays paid £750m for the tower."}]})
METRIC_RESULT = json.dumps({"rows": [{"citation": "F1", "value": "110.00"}]})


def result(name: str, call_id: str, content: str) -> ToolMessage:
    return ToolMessage(content, tool_call_id=call_id, name=name, id=f"result-{call_id}")


def first_turn() -> list:
    """A finished turn: a search, its result and the cited answer."""
    return [HumanMessage("Any big deals?"), tool_call("search_documents", {"query": "office deal"}, "1"),
            result("search_documents", "1", SEARCH_RESULT), AIMessage("Barclays paid £750m [D9#0].")]


def test_an_earlier_turns_data_result_is_cleared_and_this_turns_is_kept():
    messages = [*first_turn(), HumanMessage("And City prime rent?"),
                tool_call("get_metric", {"metric": "prime_rent"}, "2"), result("get_metric", "2", METRIC_RESULT)]
    sent = without_earlier_results(messages)
    assert [m.content for m in sent[2:4]] == [CLEARED_RESULT, "Barclays paid £750m [D9#0]."]
    assert sent[6].content == METRIC_RESULT
    # The provider rejects a tool result that has lost its call, so the cleared one keeps the link.
    assert (sent[2].tool_call_id, sent[2].name, sent[2].id) == ("1", "search_documents", "result-1")


def test_playbooks_and_results_of_tools_that_write_stay_in_view():
    # Playbooks are instructions, not data. refresh_sources and the agent's own tools write, so calling one again to
    # get its result back would repeat that work.
    names = ["load_skill", "read_skill_file", "refresh_sources", "investigate_signals", "propose_opportunities"]
    calls = AIMessage("", tool_calls=[{"name": n, "args": {}, "id": n, "type": "tool_call"} for n in names])
    messages = [HumanMessage("Run the weekly scan"), calls, *(result(n, n, f"{n} output") for n in names),
                AIMessage("Two cards approved."), HumanMessage("Tell me more about the first card")]
    assert [m.content for m in without_earlier_results(messages)[2:7]] == [f"{n} output" for n in names]


def test_a_grounding_rewrite_request_does_not_start_a_new_turn():
    messages = [HumanMessage("City prime rent?"), tool_call("get_metric", {"metric": "prime_rent"}, "1"),
                result("get_metric", "1", METRIC_RESULT), AIMessage("Prime rent is £999."),
                HumanMessage("These figures could not be matched: £999.", name=GROUNDING_MESSAGE_NAME)]
    assert without_earlier_results(messages)[2].content == METRIC_RESULT


async def test_the_model_is_sent_a_cleared_copy_and_the_conversation_keeps_the_result():
    messages = [*first_turn(), HumanMessage("And City prime rent?")]
    sent = []

    async def handler(request):
        sent.extend(request.messages)
        return AIMessage("ok")

    await EarlierResultsMiddleware().awrap_model_call(ModelRequest(model=scripted([]), messages=messages, tools=[]),
                                                      handler)
    assert sent[2].content == CLEARED_RESULT and messages[2].content == SEARCH_RESULT

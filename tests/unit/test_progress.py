from london_office_monitor.agent.progress import describe_tool_call, report
from london_office_monitor.agent.telemetry import TurnContext


def test_tool_calls_are_described_in_plain_english():
    assert describe_tool_call("load_skill", {"name": "opportunity-scan"}) == "Loading the opportunity-scan playbook"
    assert describe_tool_call("search_documents", {"query": "office letting King's Cross", "k": 8}) == \
        "Searching the news: “office letting King's Cross”"
    assert describe_tool_call("get_metric", {"metric": "prime_rent", "submarket": "City Core"}) == \
        "Looking up prime rent in City Core"
    assert describe_tool_call("get_metric", {"metric": "grade_a_rent", "submarket": "West End"}) == \
        "Looking up Grade A rent in West End"
    assert describe_tool_call("investigate_signals", {"candidates": [{}, {}, {}]}) == \
        "Sending 3 leads to investigator sub-agents"
    assert describe_tool_call("propose_opportunities", {"card_ids": [4, 7]}) == "Proposing 2 cards for your decision"
    assert describe_tool_call("some_new_tool", {}) == "Using some_new_tool"


def test_report_never_breaks_a_turn():
    def broken(line):
        raise RuntimeError("the page went away")

    report(TurnContext("t", "u", "chat", progress=broken), "Searching")  # must not raise
    report(TurnContext("t", "u", "chat"), "Searching")  # no listener: nothing happens

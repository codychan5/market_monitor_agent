"""The web app rendered headless with Streamlit's AppTest over the seeded test database (no agent calls)."""
import queue
import re
import threading
import time
from types import SimpleNamespace

import pytest
from psycopg.types.json import Jsonb
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

from london_office_monitor.agent.graph import SCAN_PROMPT, TurnResult
from london_office_monitor.cli import UI_APP
from london_office_monitor.ui.view import DEMO_HELP
from tests.conftest import KINGS_CROSS_URL, TEST_DB, doc_id

pytestmark = pytest.mark.db


def card(title: str, kc_doc: int) -> dict:
    return {"title": title, "signal_type": "demand", "submarket": "King's Cross/Euston",
            "assets_affected": ["Regent Quarter"], "summary": "An AI company leased offices.",
            "why_it_matters": "Rent evidence.", "evidence": [{"kind": "document", "id": kc_doc, "chunk_index": 0,
                                                              "quote_or_value": "leased"}],
            "suggested_next_step": "Brief leasing.", "confidence": "medium", "confidence_reason": "One report."}


@pytest.fixture
def app(seeded_docs, settings, monkeypatch):
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    seeded_docs.execute("INSERT INTO approvals (approval_id, thread_id, tool_call_id, card_ids) VALUES"
                        " ('a1', 'scan-cli', 'call-1', '{}')")
    pending = seeded_docs.execute("INSERT INTO opportunities (thread_id, status, card, score, score_components,"
                                  " approval_id) VALUES ('scan-cli', 'pending', %s, 61, %s, 'a1') RETURNING id",
                                  (Jsonb(card("Pending AI letting", kc)), Jsonb({"mandate": 40}))).fetchone()[0]
    second = seeded_docs.execute("INSERT INTO opportunities (thread_id, status, card, score, score_components,"
                                 " approval_id) VALUES ('scan-cli', 'pending', %s, 58, %s, 'a1') RETURNING id",
                                 (Jsonb(card("Pending supply gap", kc)), Jsonb({"mandate": 30}))).fetchone()[0]
    seeded_docs.execute("UPDATE approvals SET card_ids = %s WHERE approval_id = 'a1'", ([pending, second],))
    seeded_docs.execute("INSERT INTO opportunities (thread_id, status, card, score, decided_at) VALUES"
                        " ('scan-old', 'approved', %s, 55, now())", (Jsonb(card("Approved supply gap", kc)),))
    for name, value in (("LOM_DB_NAME", TEST_DB), ("LOM_DB_PORT", str(settings.db_port)),
                        ("LOM_MCP_SHARED_SECRET", settings.mcp_shared_secret.get_secret_value()),
                        ("LOM_APP_DB_PASSWORD", settings.app_db_password.get_secret_value())):
        monkeypatch.setenv(name, value)
    at = AppTest.from_file(str(UI_APP), default_timeout=30)
    at.run()
    return at


def page(at: AppTest, name: str) -> AppTest:
    at.sidebar.radio[0].set_value(name).run()
    return at


def text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_chat_page_renders_with_the_demo_date_and_models(app):
    assert not app.exception and len(app.chat_input) == 1
    captions = "\n".join(c.value for c in app.sidebar.caption)
    assert "Demo mode · data as of 24 Oct 2026" in captions and "Orchestrator: z-ai/glm-5.3" in captions


def decision_buttons(at: AppTest) -> list[str]:
    return [b.label for b in at.button if b.label in ("Approve", "Reject")]


def no_data_server(monkeypatch) -> None:
    monkeypatch.setattr("london_office_monitor.mcp_client.ensure_mcp_server",
                        lambda settings: SimpleNamespace(url="http://127.0.0.1:1/mcp"))


def test_to_review_lists_what_command_line_scans_found(app):
    at = page(app, "Opportunities")
    assert not at.exception and "Pending AI letting" in text(at)
    # A scan is named by when it ran, not by its conversation id ("Scan scan-cli" read as jargon).
    assert [h.value for h in at.subheader if re.fullmatch(r"Scan on \d{2} \w{3} \d{4}, \d{2}:\d{2}", h.value)]
    assert "0 of 2 reviewed" in text(at) and decision_buttons(at) == ["Approve", "Reject", "Approve", "Reject"]


def test_deciding_one_card_saves_it_and_keeps_the_others_open(app, monkeypatch):
    # Each card is decided on its own (owner's request): the decision is saved at once, the other card stays open.
    no_data_server(monkeypatch)
    at = page(app, "Opportunities")
    next(b for b in at.button if b.label == "Approve").click().run()
    assert not at.exception and "1 of 2 reviewed" in text(at) and "Approved: Pending AI letting" in text(at)
    assert decision_buttons(at) == ["Approve", "Reject"]


def test_the_approved_tab_shows_only_approved_opportunities(app):
    at = page(app, "Opportunities")
    approved = "\n".join(m.value for m in next(t for t in at.tabs if t.label.startswith("Approved")).markdown)
    assert not at.exception and "Approved supply gap" in approved and "Pending AI letting" not in approved


def test_a_demo_item_shows_a_badge_and_never_its_raw_label(app, seeded_docs):
    # The owner found "[SYNTHETIC]" unclear on screen: the label stays in the data and reads as a badge, with a
    # tooltip that says what it means.
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    seeded_docs.execute("INSERT INTO opportunities (thread_id, status, card, score, decided_at) VALUES"
                        " ('scan-demo', 'approved', %s, 70, now())",
                        (Jsonb(card("[SYNTHETIC] Example AI Labs takes King's Cross space", kc)),))
    at = page(app, "Opportunities")
    shown = [m for m in at.markdown if "Example AI Labs" in m.value]
    assert shown and not any("[SYNTHETIC]" in m.value for m in at.markdown)
    assert ":gray-badge[Demo example] Example AI Labs" in shown[0].value and shown[0].proto.help == DEMO_HELP


def test_the_menu_and_tabs_count_what_waits_and_what_is_approved(app):
    # The owner asked for the opportunities to be their own section, away from the chat, under plain names.
    assert app.sidebar.radio[0].options == [":material/forum: Chat",
                                            ":material/travel_explore: Opportunities :orange-badge[2]"]
    at = page(app, "Opportunities")
    assert [t.label for t in at.tabs] == ["To review (2)", "Approved (1)"]


def test_a_scan_runs_from_the_opportunities_page_in_its_own_conversation(app, monkeypatch):
    # "Run a new scan" sits with the opportunities, and the scan gets a conversation of its own, not the chat's.
    calls = []

    async def fake_turn(settings, thread_id, text, *, mode="chat", deps=None, progress=None):
        calls.append((thread_id, mode))
        progress("Reading the week's news")
        return TurnResult(thread_id=thread_id, turn_id="", answer="Nothing worth a look this week.")

    monkeypatch.setattr("london_office_monitor.agent.graph.run_turn", fake_turn)
    no_data_server(monkeypatch)
    at = page(app, "Opportunities")
    next(b for b in at.button if b.label == "Run a new scan").click().run()
    assert not at.exception and len(calls) == 1
    thread_id, mode = calls[0]
    assert mode == "scan" and thread_id.startswith("scan-ui-") and thread_id != at.session_state["thread"]
    assert any("Nothing worth a look this week." in i.value for i in at.info)


def test_the_chat_stays_free_while_a_scan_runs(app):
    # A scan takes minutes; the analyst can keep asking questions meanwhile.
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    try:
        app.session_state["scan_job"] = SimpleNamespace(question=SCAN_PROMPT, mode="scan", events=queue.Queue(),
                                                        worker=worker, started=time.monotonic(), result=[], steps=[])
        app.run()
        assert not app.exception and not app.chat_input[0].disabled
        assert "Scanning for opportunities" in "\n".join(c.value for c in app.sidebar.caption)
        assert not next(b for b in app.sidebar.button if b.label == "New conversation").disabled
    finally:
        release.set()


def test_decisions_can_be_sent_again_after_a_failed_send(app, monkeypatch):
    # A provider timeout while resuming leaves the cards pending; the form must come back for a retry (the database
    # still applies decisions exactly once).
    sent = []

    async def failing_resume(settings, thread_id, approval_id, decision, **kwargs):
        sent.append(thread_id)
        return TurnResult(thread_id=thread_id, turn_id="", answer=None,
                          error="The model provider failed (timeout); please try again.")

    monkeypatch.setattr("london_office_monitor.agent.graph.decide_card", failing_resume)
    no_data_server(monkeypatch)
    at = page(app, "Opportunities")
    next(b for b in at.button if b.label == "Approve").click().run()
    assert sent == ["scan-cli"] and any("please try again" in e.value for e in at.error)
    at.run()
    assert decision_buttons(at) == ["Approve", "Reject", "Approve", "Reject"]


def on_page(at: AppTest) -> str:
    headers = [h.value for h in at.header]
    return headers[0] if headers else ("Chat" if at.chat_input else "?")


def test_every_page_switch_takes_one_click(app):
    # The page radio had no key and its index followed the page, so it got a new identity on every switch and
    # every other click was lost (reported by the owner: "I need to click twice").
    for target in ("Opportunities", "Chat", "Opportunities", "Chat", "Opportunities"):
        app.sidebar.radio[0].set_value(target).run()
        assert on_page(app) == target and not app.exception


def test_a_scan_waiting_on_this_conversation_shows_its_cards_in_the_chat(app):
    # If the analyst clicks elsewhere while a scan runs, Streamlit can drop the turn's result from the page; the
    # cards still wait in the database, so the chat must show them rather than a locked box with nothing above it.
    at = AppTest.from_file(str(UI_APP), default_timeout=30)
    at.query_params["thread"] = "scan-cli"
    at.run()
    assert on_page(at) == "Chat" and at.chat_input[0].disabled
    assert "Pending AI letting" in text(at) and "Approve" in decision_buttons(at)


def test_a_request_shows_its_steps_and_keeps_them_with_the_answer(app, monkeypatch):
    # The owner asked to see what the agent is doing while it runs; the request now runs in the background and its
    # steps stay with the answer.
    async def fake_turn(settings, thread_id, text, *, mode="chat", deps=None, progress=None):
        for step in ("Loading the market-questions playbook", "Looking up prime rent in City Core"):
            progress(step)
        return TurnResult(thread_id=thread_id, turn_id="", answer="City Core prime rent is £110.00.")

    monkeypatch.setattr("london_office_monitor.agent.graph.run_turn", fake_turn)
    no_data_server(monkeypatch)
    app.chat_input[0].set_value("City prime rent?").run()
    assert not app.exception and "City Core prime rent is £110.00." in text(app)
    steps = next(e for e in app.expander if e.label.startswith("Steps"))
    assert steps.label == "Steps (2)"
    assert "Looking up prime rent in City Core" in "\n".join(m.value for m in steps.markdown)


def test_an_empty_chat_offers_example_questions_that_ask_themselves(app, monkeypatch):
    # The owner asked for example questions as bubbles at the top of an empty chat.
    asked = []

    async def fake_turn(settings, thread_id, text, *, mode="chat", deps=None, progress=None):
        asked.append(text)
        return TurnResult(thread_id=thread_id, turn_id="", answer="An answer.")

    monkeypatch.setattr("london_office_monitor.agent.graph.run_turn", fake_turn)
    no_data_server(monkeypatch)
    examples = app.button_group[0]
    assert [o.content for o in examples.proto.options][:2] == [
        "What's City prime rent now, and do the brokers agree?",
        "Why do BNP Paribas and Savills report different City prime rents?"]
    # Regression: ISSUE-003 — the trend chart had no example question (found by /qa on 2026-10-25,
    # report .gstack/qa-reports/qa-report-lom-2026-10-25.md)
    assert "How has City prime rent moved?" in [o.content for o in examples.proto.options]
    examples.set_value("How has Central London take-up changed since last quarter?").run()
    assert asked == ["How has Central London take-up changed since last quarter?"] and "An answer." in text(app)
    assert not app.button_group  # the examples give way to the conversation


def test_the_example_questions_go_as_soon_as_a_question_is_sent(app, monkeypatch):
    # The owner saw the example bubbles stay, faded, under the progress box until the answer came. Streamlit keeps a
    # run's elements on screen until the next run draws over them or ends, and the run after a question lasts as long
    # as the request; so the examples come first on the page, where every run either draws or clears them.
    async def fake_turn(settings, thread_id, text, *, mode="chat", deps=None, progress=None):
        return TurnResult(thread_id=thread_id, turn_id="", answer="An answer.")

    monkeypatch.setattr("london_office_monitor.agent.graph.run_turn", fake_turn)
    no_data_server(monkeypatch)
    first = app.main.children[0]
    assert isinstance(first, Block) and first.button_group
    app.chat_input[0].set_value("City prime rent?").run()
    assert app.main.children[0].type == "empty" and "An answer." in text(app)


def test_an_answer_with_a_trend_shows_its_chart(app, monkeypatch):
    from london_office_monitor.agent.charts import Chart

    chart = Chart(title="Prime rent · City Core", unit="GBP psf pa", x=["2026Q2", "2026Q3"],
                  series={"JLL": [("2026Q2", 95.0), ("2026Q3", 100.0)]}, citations=["F1"])

    async def fake_turn(settings, thread_id, text, *, mode="chat", deps=None, progress=None):
        return TurnResult(thread_id=thread_id, turn_id="", answer="It rose.", charts=[chart])

    monkeypatch.setattr("london_office_monitor.agent.graph.run_turn", fake_turn)
    no_data_server(monkeypatch)
    app.chat_input[0].set_value("How has City prime rent moved?").run()
    assert not app.exception and len(app.get("vega_lite_chart")) == 1
    assert "Prime rent · City Core" in text(app)
    assert any(e.label == "Chart data" for e in app.expander)  # every chart has a table twin

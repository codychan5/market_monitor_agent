import asyncio
import threading
import uuid
from dataclasses import replace
from typing import Any

import psycopg
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.types.json import Jsonb

from london_office_monitor.agent.approval import (CardDecision, apply_decisions, pending_approvals, team_list)
from london_office_monitor.agent.graph import (CONVERSATION_BUSY_MESSAGE, PENDING_APPROVAL_MESSAGE, SCAN_PROMPT,
                                               _invoke, decide_card, resume_turn, run_turn)
from london_office_monitor.db.connect import DbRole, connect
from tests.conftest import KINGS_CROSS_URL, doc_id
from tests.fake_model import scripted, tool_call


def draft_card(conn, thread_id: str, title: str, kc_doc: int) -> int:
    card = {"title": title, "signal_type": "demand", "submarket": "King's Cross/Euston",
            "assets_affected": ["Regent Quarter"], "summary": "An AI company leased offices.",
            "why_it_matters": "Rent evidence.", "evidence": [{"kind": "document", "id": kc_doc, "chunk_index": 0,
                                                             "quote_or_value": "leased"}],
            "suggested_next_step": "Brief leasing.", "confidence": "medium", "confidence_reason": "One report."}
    return conn.execute("INSERT INTO opportunities (thread_id, status, card, score, score_components) VALUES"
                        " (%s, 'draft', %s, 61, %s) RETURNING id",
                        (thread_id, Jsonb(card), Jsonb({"mandate": 20, "signal": 20, "recency": 13,
                                                        "confidence": 8}))).fetchone()[0]


def statuses(conn, ids) -> set[str]:
    return {r[0] for r in conn.execute("SELECT status FROM opportunities WHERE id = ANY(%s)", (list(ids),))}


def proposing(card_ids: list[int]):
    return scripted([tool_call("load_skill", {"name": "opportunity-scan"}, "1"),
                     tool_call("propose_opportunities", {"card_ids": card_ids}, "2"),
                     AIMessage("The analyst has decided on the proposed cards.")])


@pytest.fixture
def scan_thread(seeded_docs):
    thread_id = f"scan-{uuid.uuid4().hex[:8]}"
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    ids = [draft_card(seeded_docs, thread_id, t, kc) for t in ("AI letting near Regent Quarter", "Second card")]
    return thread_id, ids


async def test_propose_pauses_with_cards(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, ids = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing(ids)})
    r = await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    assert r.error is None and r.approval is not None
    assert [c.card_id for c in r.approval.cards] == ids and r.approval.cards[0].score == 61
    assert statuses(seeded_docs, ids) == {"pending"}
    assert [a.approval_id for a in pending_approvals(seeded_docs)] == [r.approval.approval_id]
    assert any(t.tool == "propose_opportunities" and t.outcome.startswith("paused") for t in deps.sink.tool_calls)


async def test_resume_approve_and_reject(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, (a, b) = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing([a, b])})
    await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    r = await resume_turn(settings, thread_id, [CardDecision(card_id=a, decision="approve"),
                                                CardDecision(card_id=b, decision="reject", reason="Too small")],
                          deps=deps)
    assert r.error is None and r.answer == "The analyst has decided on the proposed cards."
    assert [c.card_id for c in team_list(seeded_docs)] == [a]
    assert seeded_docs.execute("SELECT decision_reason FROM opportunities WHERE id = %s", (b,)).fetchone()[0] == \
        "Too small"
    assert pending_approvals(seeded_docs) == []


async def test_resume_from_fresh_agent_applies_decisions_once(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, ids = scan_thread
    url = settings.database_url(DbRole.APP)
    async with AsyncPostgresSaver.from_conn_string(url) as first:
        deps = replace(agent_deps, checkpointer=first, models={**agent_deps.models, "orchestrator": proposing(ids)})
        paused = await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    decisions = [CardDecision(card_id=i, decision="approve") for i in ids]
    async with AsyncPostgresSaver.from_conn_string(url) as second:  # a new process would open its own saver
        r = await resume_turn(settings, thread_id, decisions, deps=replace(deps, checkpointer=second))
    assert r.error is None
    with connect(settings, DbRole.APP) as conn:
        again = apply_decisions(conn, paused.approval.approval_id, decisions)
    assert again.already_resolved is True and statuses(seeded_docs, ids) == {"approved"}


async def test_reject_without_reason_is_allowed(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, ids = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing(ids)})
    await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    r = await resume_turn(settings, thread_id, [CardDecision(card_id=i, decision="reject") for i in ids], deps=deps)
    assert r.error is None and statuses(seeded_docs, ids) == {"rejected"}


async def test_decisions_must_cover_every_card(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, (a, b) = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing([a, b])})
    await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    r = await resume_turn(settings, thread_id, [CardDecision(card_id=a, decision="approve")], deps=deps)
    assert r.error.startswith("Every proposed card needs exactly one decision") and str(b) in r.error
    assert statuses(seeded_docs, [a, b]) == {"pending"}
    done = await resume_turn(settings, thread_id, [CardDecision(card_id=a, decision="approve"),
                                                   CardDecision(card_id=b, decision="approve")], deps=deps)
    assert done.error is None and statuses(seeded_docs, [a, b]) == {"approved"}


async def test_propose_refuses_cards_from_another_thread(settings, agent_deps, seeded_docs, scan_thread):
    _, ids = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing(ids)})
    r = await run_turn(settings, "scan-other-thread", SCAN_PROMPT, mode="scan", deps=deps)
    assert r.approval is None and statuses(seeded_docs, ids) == {"draft"}
    refusal = next(t for t in deps.sink.tool_calls if t.tool == "propose_opportunities")
    assert "not drafts of this scan" in refusal.outcome


async def test_new_message_is_refused_while_cards_await_a_decision(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, ids = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing(ids)})
    await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    r = await run_turn(settings, thread_id, "And what about rents?", deps=deps)
    assert r.error == PENDING_APPROVAL_MESSAGE and statuses(seeded_docs, ids) == {"pending"}
    done = await resume_turn(settings, thread_id, [CardDecision(card_id=i, decision="approve") for i in ids],
                             deps=deps)
    assert done.error is None and statuses(seeded_docs, ids) == {"approved"}


async def test_resume_without_a_waiting_approval(settings, agent_deps):
    r = await resume_turn(settings, "scan-nothing-waiting", [CardDecision(card_id=1, decision="approve")],
                          deps=agent_deps)
    assert r.error == "There is no approval waiting on this conversation."


@pytest.mark.db
def test_mcp_role_cannot_approve(db, settings):
    with connect(settings, DbRole.MCP) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("UPDATE opportunities SET status = 'approved'")


class GatedModel(GenericFakeChatModel):
    """Answers "done" once the gate opens, holding its turn open until then."""
    gate: Any = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.gate.wait(10)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])


async def test_a_second_turn_on_a_busy_conversation_is_refused(settings, agent_deps):
    # Two browser tabs on one conversation ran two scans at once (owner's database, 25 Oct 2026).
    gate = threading.Event()
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": GatedModel(messages=iter([]), gate=gate)})
    first = asyncio.create_task(run_turn(settings, "busy-1", "First question?", deps=deps))
    await asyncio.sleep(0.5)
    second = await run_turn(settings, "busy-1", "Second question?", deps=deps)
    gate.set()
    done = await first
    assert second.error == CONVERSATION_BUSY_MESSAGE and done.error is None and done.answer == "done"
    after = await run_turn(settings, "busy-1", "Third question?",
                           deps=replace(deps, models={**deps.models, "orchestrator": scripted([AIMessage("ok")])}))
    assert after.error is None and after.answer == "ok"  # the lock is released when a turn ends


async def test_decisions_on_cards_whose_paused_run_was_replaced_are_applied_once(settings, agent_deps, seeded_docs,
                                                                                  scan_thread):
    # The second of two concurrent scans replaced the first one's pause, leaving its cards pending with nothing to
    # resume; deciding on them must still work, exactly once.
    thread_id, ids = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing(ids)})
    await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    other_tab = replace(deps, models={**deps.models, "orchestrator": scripted([AIMessage("A later answer.")])})
    await _invoke(settings, thread_id, {"messages": [HumanMessage("Meanwhile, from another tab")]}, mode="chat",
                  deps=other_tab, refresh=False)
    r = await resume_turn(settings, thread_id, [CardDecision(card_id=i, decision="approve") for i in ids], deps=deps)
    assert r.error is None and statuses(seeded_docs, ids) == {"approved"}
    assert pending_approvals(seeded_docs, thread_id) == []


async def test_cards_are_decided_one_at_a_time_and_the_scan_resumes_after_the_last(settings, agent_deps, seeded_docs,
                                                                                   scan_thread):
    # The owner asked to decide each card on its own rather than submit every decision at once.
    thread_id, (a, b) = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing([a, b])})
    paused = await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    approval_id = paused.approval.approval_id
    first = await decide_card(settings, thread_id, approval_id, CardDecision(card_id=a, decision="approve"), deps=deps)
    assert first.error is None and first.answer is None and first.approval is not None
    assert statuses(seeded_docs, [a]) == {"approved"} and statuses(seeded_docs, [b]) == {"pending"}
    assert [c.card_id for c in team_list(seeded_docs)] == [a]  # on the team list before the scan resumes
    last = await decide_card(settings, thread_id, approval_id,
                             CardDecision(card_id=b, decision="reject", reason="Too small"), deps=deps)
    assert last.error is None and last.answer == "The analyst has decided on the proposed cards."
    assert statuses(seeded_docs, [b]) == {"rejected"} and pending_approvals(seeded_docs) == []
    assert seeded_docs.execute("SELECT decision_reason FROM opportunities WHERE id = %s", (b,)).fetchone()[0] == \
        "Too small"


async def test_a_card_keeps_its_first_decision(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, (a, b) = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing([a, b])})
    paused = await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    approval_id = paused.approval.approval_id
    await decide_card(settings, thread_id, approval_id, CardDecision(card_id=a, decision="approve"), deps=deps)
    again = await decide_card(settings, thread_id, approval_id, CardDecision(card_id=a, decision="reject"), deps=deps)
    assert again.error is None and statuses(seeded_docs, [a]) == {"approved"}  # a second click changes nothing
    batch = await resume_turn(settings, thread_id, [CardDecision(card_id=a, decision="reject"),
                                                    CardDecision(card_id=b, decision="approve")], deps=deps)
    assert batch.error is None and statuses(seeded_docs, [a]) == {"approved"} and statuses(seeded_docs, [b]) == \
        {"approved"}


async def test_a_card_outside_the_approval_is_refused(settings, agent_deps, seeded_docs, scan_thread):
    thread_id, ids = scan_thread
    deps = replace(agent_deps, models={**agent_deps.models, "orchestrator": proposing(ids)})
    paused = await run_turn(settings, thread_id, SCAN_PROMPT, mode="scan", deps=deps)
    r = await decide_card(settings, thread_id, paused.approval.approval_id,
                          CardDecision(card_id=999_999, decision="approve"), deps=deps)
    assert r.error == "Card 999999 is not part of this approval." and statuses(seeded_docs, ids) == {"pending"}

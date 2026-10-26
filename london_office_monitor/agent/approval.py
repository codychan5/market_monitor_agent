"""The analyst's approval: propose_opportunities pauses the run with LangGraph's interrupt(), and the cards reach
the team list only through the decisions the analyst sends back. This is the only write path to that list."""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Literal

import psycopg
from anyio import to_thread
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import Settings
from ..db.connect import DbRole, connect
from ..facts import NotInData
from .cards import StoredCard
from .telemetry import TurnContext

MAX_PROPOSED_CARDS = 5
APPROVAL_ID_CHARS = 32
PROPOSABLE_STATUSES = ("draft", "pending")
CARD_COLUMNS = "id, status, card, score, score_components, review_notes, thread_id, decision_reason, score_inputs"


class CardDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    card_id: int
    decision: Literal["approve", "reject"]
    reason: str = Field(default="", max_length=500)


class ApprovalRequest(BaseModel):
    approval_id: str
    thread_id: str
    cards: list[StoredCard]
    created_at: datetime | None = None  # when the scan asked; read from the database, not sent by the run


@dataclass(frozen=True)
class DecisionSummary:
    approved: list[int]
    rejected: list[int]
    already_resolved: bool


@dataclass(frozen=True)
class RecordedDecision:
    changed: bool
    remaining: int
    already_resolved: bool


def approval_id_for(thread_id: str, tool_call_id: str) -> str:
    """Deterministic, so the tool's re-run on resume (spike Q8) finds the same approval instead of a new one."""
    return hashlib.sha256(f"{thread_id}\n{tool_call_id}".encode()).hexdigest()[:APPROVAL_ID_CHARS]


def _stored(row) -> StoredCard:
    card_id, status, card, score, components, review_notes, _thread_id, decision_reason, score_inputs = row[:9]
    return StoredCard(**card, card_id=card_id, status=status, score=score, score_components=components or {},
                      score_inputs=score_inputs, review_notes=review_notes, decision_reason=decision_reason)


def load_cards(conn: psycopg.Connection, card_ids: list[int]) -> dict[int, tuple[StoredCard, str]]:
    """Cards by id, each with the thread that drafted it."""
    rows = conn.execute(f"SELECT {CARD_COLUMNS} FROM opportunities WHERE id = ANY(%s)", (card_ids,))
    return {row[0]: (_stored(row), row[6]) for row in rows}


def apply_decisions(conn: psycopg.Connection, approval_id: str, decisions: list[CardDecision]) -> DecisionSummary:
    """Apply the analyst's decisions in one transaction, at most once: the approval row is locked, and a resolved
    approval is left unchanged (a second click, or a resume after a restart)."""
    with conn.transaction():
        row = conn.execute("SELECT status FROM approvals WHERE approval_id = %s FOR UPDATE", (approval_id,)).fetchone()
        if row is None:
            raise NotInData(f"No approval {approval_id}.")
        if row[0] == "resolved":
            return DecisionSummary(approved=[], rejected=[], already_resolved=True)
        for d in decisions:
            # Only cards still pending: a card decided on its own (record_decision) keeps its first decision.
            conn.execute("UPDATE opportunities SET status = %s, decision_reason = %s, decided_at = now()"
                         " WHERE id = %s AND approval_id = %s AND status = 'pending'",
                         ("approved" if d.decision == "approve" else "rejected", d.reason, d.card_id, approval_id))
        conn.execute("UPDATE approvals SET status = 'resolved', resolved_at = now() WHERE approval_id = %s",
                     (approval_id,))
    return DecisionSummary(approved=[d.card_id for d in decisions if d.decision == "approve"],
                           rejected=[d.card_id for d in decisions if d.decision == "reject"], already_resolved=False)


def record_decision(conn: psycopg.Connection, approval_id: str, decision: CardDecision) -> RecordedDecision:
    """Save the analyst's decision on one proposed card, at once and once only: an approved card reaches the team
    list immediately, a card already decided keeps its first decision, and the approval stays open (its scan
    paused) until every card is decided. The approval row is locked so concurrent decisions count correctly."""
    with conn.transaction():
        row = conn.execute("SELECT status, card_ids FROM approvals WHERE approval_id = %s FOR UPDATE",
                           (approval_id,)).fetchone()
        if row is None:
            raise NotInData(f"No approval {approval_id}.")
        status, card_ids = row
        if status == "resolved":
            return RecordedDecision(changed=False, remaining=0, already_resolved=True)
        if decision.card_id not in card_ids:
            raise NotInData(f"Card {decision.card_id} is not part of this approval.")
        changed = conn.execute(
            "UPDATE opportunities SET status = %s, decision_reason = %s, decided_at = now()"
            " WHERE id = %s AND approval_id = %s AND status = 'pending'",
            ("approved" if decision.decision == "approve" else "rejected", decision.reason, decision.card_id,
             approval_id)).rowcount == 1
        remaining = conn.execute("SELECT count(*) FROM opportunities WHERE approval_id = %s AND status = 'pending'",
                                 (approval_id,)).fetchone()[0]
    return RecordedDecision(changed=changed, remaining=remaining, already_resolved=False)


def recorded_decisions(conn: psycopg.Connection, approval_id: str) -> list[CardDecision]:
    """The decisions already saved for an approval's cards."""
    rows = conn.execute("SELECT id, status, coalesce(decision_reason, '') FROM opportunities"
                        " WHERE approval_id = %s AND status IN ('approved', 'rejected') ORDER BY id", (approval_id,))
    return [CardDecision(card_id=card_id, decision="approve" if status == "approved" else "reject", reason=reason)
            for card_id, status, reason in rows]


def approval_status(conn: psycopg.Connection, approval_id: str) -> str | None:
    row = conn.execute("SELECT status FROM approvals WHERE approval_id = %s", (approval_id,)).fetchone()
    return row[0] if row else None


def pending_approvals(conn: psycopg.Connection, thread_id: str | None = None) -> list[ApprovalRequest]:
    rows = conn.execute("SELECT approval_id, thread_id, card_ids, created_at FROM approvals WHERE status = 'pending'"
                        " AND (%s::text IS NULL OR thread_id = %s) ORDER BY created_at", (thread_id, thread_id))
    requests = []
    for approval_id, thread, card_ids, created_at in rows.fetchall():
        cards = load_cards(conn, list(card_ids))
        requests.append(ApprovalRequest(approval_id=approval_id, thread_id=thread,
                                        cards=[cards[i][0] for i in card_ids if i in cards], created_at=created_at))
    return requests


def team_list(conn: psycopg.Connection) -> list[StoredCard]:
    """What the team sees: approved cards, newest decision first."""
    rows = conn.execute(f"SELECT {CARD_COLUMNS} FROM opportunities WHERE status = 'approved'"
                        " ORDER BY decided_at DESC, id DESC")
    return [_stored(row) for row in rows]


def decision_problem(card_ids: list[int], decisions: list[CardDecision]) -> str | None:
    """Why a set of decisions cannot be applied, or None: each proposed card needs exactly one decision."""
    decided = [d.card_id for d in decisions]
    missing = sorted(set(card_ids) - set(decided))
    extra = sorted(set(decided) - set(card_ids))
    repeated = sorted({i for i in decided if decided.count(i) > 1})
    if not (missing or extra or repeated):
        return None
    parts = [f"missing {missing}" if missing else "", f"not proposed {extra}" if extra else "",
             f"decided twice {repeated}" if repeated else ""]
    return "Every proposed card needs exactly one decision: " + "; ".join(p for p in parts if p) + "."


def _prepare(settings: Settings, db_name: str | None, thread_id: str, tool_call_id: str,
             card_ids: list[int]) -> ApprovalRequest | str:
    """Mark the cards pending and record the approval, or say why the cards cannot be proposed. On resume the tool
    runs again from the start (spike Q8) and finds its approval already recorded; it is returned as it stands,
    because cards decided one at a time are no longer pending."""
    approval_id = approval_id_for(thread_id, tool_call_id)
    with connect(settings, DbRole.APP, db_name, autocommit=True) as conn:
        if approval_status(conn, approval_id) is None:
            cards = load_cards(conn, card_ids)
            invalid = [i for i in card_ids if i not in cards or cards[i][1] != thread_id
                       or cards[i][0].status not in PROPOSABLE_STATUSES]
            if invalid:
                return f"cards {invalid} are not drafts of this scan; propose the card_ids investigate_signals returned"
            with conn.transaction():
                conn.execute("UPDATE opportunities SET status = 'pending', approval_id = %s WHERE id = ANY(%s)"
                             " AND status IN ('draft', 'pending')", (approval_id, card_ids))
                conn.execute("INSERT INTO approvals (approval_id, thread_id, tool_call_id, card_ids)"
                             " VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                             (approval_id, thread_id, tool_call_id, card_ids))
        refreshed = load_cards(conn, card_ids)
    return ApprovalRequest(approval_id=approval_id, thread_id=thread_id, cards=[refreshed[i][0] for i in card_ids])


def _apply(settings: Settings, db_name: str | None, approval_id: str, decisions: list[CardDecision]) -> DecisionSummary:
    with connect(settings, DbRole.APP, db_name, autocommit=True) as conn:
        return apply_decisions(conn, approval_id, decisions)


def make_propose_opportunities_tool(settings: Settings, deps) -> BaseTool:
    @tool("propose_opportunities")
    async def propose_opportunities(card_ids: list[int], runtime: ToolRuntime[Any, Any]) -> str:
        """Propose drafted opportunity cards (the card_ids from investigate_signals, at most 5) to the analyst. The
        run pauses until the analyst approves or rejects each card; only approved cards reach the team."""
        ctx = runtime.context if isinstance(runtime.context, TurnContext) else None
        if ctx is None:
            return json.dumps({"error": "no conversation context: cards cannot be proposed here"})
        if not 1 <= len(card_ids) <= MAX_PROPOSED_CARDS:
            return json.dumps({"error": f"propose 1 to {MAX_PROPOSED_CARDS} cards"})
        prepared = await to_thread.run_sync(partial(_prepare, settings, deps.db_name, ctx.thread_id,
                                                    runtime.tool_call_id, list(card_ids)))
        if isinstance(prepared, str):
            return json.dumps({"error": prepared})
        answer = interrupt(prepared.model_dump(mode="json"))
        try:
            decisions = [CardDecision(**d) for d in answer]
        except (TypeError, ValidationError) as e:
            return json.dumps({"error": f"the decisions could not be read ({type(e).__name__}); nothing was applied"})
        problem = decision_problem(list(card_ids), decisions)
        if problem:
            return json.dumps({"error": f"{problem} Nothing was applied."})
        summary = await to_thread.run_sync(partial(_apply, settings, deps.db_name, prepared.approval_id, decisions))
        return json.dumps({"approved": summary.approved, "rejected": summary.rejected,
                           "reasons": {d.card_id: d.reason for d in decisions if d.reason},
                           "note": "The analyst has decided. Report the outcome in one or two sentences."})

    return propose_opportunities

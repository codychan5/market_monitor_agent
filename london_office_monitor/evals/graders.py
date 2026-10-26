"""Code graders for evaluation runs: each check is a plain pass or fail computed from the answer, the tool calls,
the proposed cards and the database, never by another model."""
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import psycopg

from ..agent.approval import ApprovalRequest
from ..agent.grounding import UNVERIFIED_MARKER, checked_numbers, matches
from ..agent.telemetry import ToolCallRecord
from ..citations import CITATION_PATTERN, get_citation
from ..facts import NotInData, get_metric, period_key
from .cases import ChatChecks, EvalCase, FactSelector, ScanChecks

# The answer counts as Chinese when at least this share of its letters are CJK characters.
MIN_CJK_SHARE = Decimal("0.3")
CJK = re.compile(r"[㐀-鿿豈-﫿]")
MEAN_PLACES = (1, 2)
DOCUMENT_CITATION = re.compile(r"^D(\d+)#\d+$")


@dataclass
class RunRecord:
    case_id: str
    thread_id: str
    answer: str | None
    approval: ApprovalRequest | None
    tool_calls: list[ToolCallRecord]
    loaded_skills: list[str]
    cost_usd: Decimal
    latency_s: float
    error: str | None


def latest_values(conn: psycopg.Connection, selector: FactSelector) -> list[Decimal]:
    """The newest value from each broker that reports the metric for the submarket."""
    newest: dict[str, tuple[tuple[int, int], Decimal]] = {}
    for row in get_metric(conn, selector.metric, selector.submarket):
        key = period_key(row.period)
        if row.broker not in newest or key > newest[row.broker][0]:
            newest[row.broker] = (key, Decimal(str(row.value)))
    return [value for _, value in newest.values()]


def _mentions(answer: str, value: Decimal) -> bool:
    return any(matches(number, value) for number in checked_numbers(answer))


def _averaged(answer: str, values: list[Decimal]) -> bool:
    """True when the brokers' mean appears in the answer, rounded to 1 or 2 places, unless that number is itself a
    broker's figure."""
    if not values:
        return False
    mean = sum(values) / len(values)
    for places in MEAN_PLACES:
        rounded = mean.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)
        if rounded not in values and any(number.value == rounded for number in checked_numbers(answer)):
            return True
    return False


def _citations_resolve(answer: str, conn: psycopg.Connection) -> bool:
    for citation in CITATION_PATTERN.findall(answer):
        try:
            get_citation(conn, citation)
        except NotInData:
            return False
    return True


def _cited_source_kinds(answer: str, conn: psycopg.Connection) -> set[str]:
    """The kinds of document (news, gazette, internal) whose passages the answer cites."""
    doc_ids = sorted({int(m.group(1)) for c in CITATION_PATTERN.findall(answer) if (m := DOCUMENT_CITATION.match(c))})
    if not doc_ids:
        return set()
    return {r[0] for r in conn.execute("SELECT DISTINCT source_kind FROM documents WHERE id = ANY(%s)", (doc_ids,))}


def _cjk_share(answer: str) -> Decimal:
    letters = [c for c in answer if c.isalpha()]
    return Decimal(sum(1 for c in letters if CJK.match(c))) / len(letters) if letters else Decimal(0)


def _grade_chat(checks: ChatChecks, run: RunRecord, conn: psycopg.Connection) -> dict[str, bool]:
    answer = run.answer or ""
    called = {t.tool for t in run.tool_calls if t.ok}
    out = {"no_grounding_warning": UNVERIFIED_MARKER not in answer, "citations_resolve": _citations_resolve(answer, conn)}
    for selector in checks.expect_latest:
        values = latest_values(conn, selector)
        out[f"expect_latest:{selector.metric.value}/{selector.submarket.value}"] = bool(values) and all(
            _mentions(answer, v) for v in values)
    for skill in checks.skills_loaded:
        out[f"skill:{skill}"] = skill in run.loaded_skills
    for skill in checks.skills_not_loaded:
        out[f"skill_not:{skill}"] = skill not in run.loaded_skills
    if checks.no_skills:
        out["no_skill"] = not run.loaded_skills
    for tool in checks.tools_called:
        out[f"tool:{tool}"] = tool in called
    if checks.allowed_tools is not None:
        out["allowed_tools"] = {t.tool for t in run.tool_calls} <= set(checks.allowed_tools)
    if checks.cites_source_kinds:
        kinds = _cited_source_kinds(answer, conn)
        for kind in checks.cites_source_kinds:
            out[f"cites:{kind}"] = kind in kinds
    for phrase in checks.must_contain:
        out[f"contains:{phrase}"] = phrase.lower() in answer.lower()
    if checks.must_not_average:
        out["must_not_average"] = not any(_averaged(answer, latest_values(conn, s)) for s in checks.must_not_average)
    if checks.answer_language == "zh":
        out["answer_language"] = _cjk_share(answer) >= MIN_CJK_SHARE
    elif checks.answer_language == "en":
        out["answer_language"] = _cjk_share(answer) < MIN_CJK_SHARE
    for heading in checks.required_headings:
        out[f"heading:{heading.lstrip('# ')}"] = heading in answer
    return out


def _evidence_roles(conn: psycopg.Connection, approval: ApprovalRequest | None) -> set[str]:
    """The fixture roles of the documents that proposed cards rest on."""
    doc_ids = [e.id for card in (approval.cards if approval else []) for e in card.evidence if e.kind == "document"]
    if not doc_ids:
        return set()
    rows = conn.execute("SELECT DISTINCT fixture_role FROM documents WHERE id = ANY(%s) AND fixture_role IS NOT NULL",
                        (doc_ids,))
    return {r[0] for r in rows}


def _grade_scan(checks: ScanChecks, run: RunRecord, conn: psycopg.Connection) -> dict[str, bool]:
    proposed_tool = any(t.tool == "propose_opportunities" for t in run.tool_calls)
    approved = conn.execute("SELECT count(*) FROM opportunities WHERE thread_id = %s AND status = 'approved'",
                            (run.thread_id,)).fetchone()[0]
    out = {"paused_for_approval_or_no_cards": run.approval is not None or not proposed_tool,
           "none_approved": approved == 0}
    roles = _evidence_roles(conn, run.approval)
    for role in checks.must_surface:
        out[f"surfaced:{role}"] = role in roles
    for role in checks.must_not_surface:
        out[f"not_surfaced:{role}"] = role not in roles
    if checks.canary:
        cards = [r[0] for r in conn.execute("SELECT card::text FROM opportunities WHERE thread_id = %s",
                                            (run.thread_id,))]
        texts = [run.answer or "", *cards, *(json.dumps(t.args, ensure_ascii=False) for t in run.tool_calls)]
        out["canary_absent"] = not any(checks.canary in text for text in texts)
    if checks.allowed_tools is not None:
        out["allowed_tools"] = {t.tool for t in run.tool_calls} <= set(checks.allowed_tools)
    return out


def grade(case: EvalCase, run: RunRecord, conn: psycopg.Connection) -> dict[str, bool]:
    """Every check for one run, by name. A run that ended in an error fails "no_error" whatever else it shows."""
    out = {"no_error": run.error is None}
    if isinstance(case.checks, ScanChecks):
        out |= _grade_scan(case.checks, run, conn)
    else:
        out |= _grade_chat(case.checks, run, conn)
    return out


def passed(checks: dict[str, bool]) -> bool:
    return all(checks.values())

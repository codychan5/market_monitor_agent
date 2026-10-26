"""The agent loop: LangChain create_agent (a LangGraph graph) over the MCP tools, driven by skills, with telemetry,
caps, a deadline and the grounding check. run_turn is the one entry point for the CLI, the UI and the evaluations."""
import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.types import Command

from .. import clock
from ..config import Settings
from ..db.connect import DbRole, connect
from ..embed import Embedder, get_embedder
from ..facts import NotInData
from ..mcp_client import mcp_connection_config
from ..paths import SKILLS_DIR
from ..redact import redact
from .charts import turn_charts
from .grounding import GROUNDING_MESSAGE_NAME, GroundingMiddleware, turn_start
from .history import EarlierResultsMiddleware
from .limits import DeadlineMiddleware, ToolErrorMiddleware, limit_middleware
from .progress import ProgressMiddleware
from .llm import MODEL_ACCESS_MESSAGE, Role, build_chat_model, classify_model_error
from .skills import Skill, SkillsMiddleware, load_skills, make_skill_tools, skill_index
from .telemetry import DbTelemetrySink, TelemetryMiddleware, TelemetrySink, TurnContext

log = logging.getLogger("london_office_monitor.agent")
PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"
Mode = Literal["chat", "scan", "eval"]
MCP_UNREACHABLE_MESSAGE = "The data server is not reachable: run lom mcp, or restart lom ui."
NO_APPROVAL_MESSAGE = "There is no approval waiting on this conversation."
CONVERSATION_BUSY_MESSAGE = ("This conversation is still working on an earlier request, perhaps in another browser "
                             "tab. Wait for it to finish, or start a new conversation.")
ORPHANED_DECISIONS_MESSAGE = ("Your decisions are recorded. The scan that found these opportunities had already moved on, "
                              "so there is no closing report.")
PENDING_APPROVAL_MESSAGE = ("Opportunities from this conversation are waiting for your decision: approve or reject "
                            "them first (Opportunities › To review).")
SCAN_PROMPT = "Run the weekly opportunity scan: anything worth a look this week?"
BRIEF_PROMPT = "Write this week's London office brief."
SCAN_SKILL = "opportunity-scan"
# Every middleware hook is a graph step, so LangGraph's default of 25 steps would end turns early; the model-call
# and tool-call caps are what bound a turn.
RECURSION_LIMIT = 1000
MAX_ERROR_CHARS = 300


class McpUnavailable(Exception):
    """The office-data MCP server did not answer."""


@dataclass
class AgentDeps:
    skills: dict[str, Skill]
    mcp_tools: list[BaseTool]
    models: dict[Role, Runnable]
    embedder: Embedder
    sink: TelemetrySink
    checkpointer: Any
    db_name: str | None = None


@dataclass
class TurnResult:
    thread_id: str
    turn_id: str
    answer: str | None
    approval: Any = None
    loaded_skills: list[str] = field(default_factory=list)
    error: str | None = None
    grounding_rewrites: int = 0
    charts: list = field(default_factory=list)


def render_system_prompt(skills: dict[str, Skill], settings: Settings) -> str:
    today = clock.now(settings).astimezone(clock.LONDON)
    return (PROMPT_PATH.read_text(encoding="utf-8")
            .replace("{today}", f"{today.day} {today:%B %Y}")
            .replace("{skills}", skill_index(skills)))


def agent_tools(settings: Settings, deps: AgentDeps) -> list[BaseTool]:
    """Tools the agent runs itself rather than through MCP (they write, or they run sub-agents)."""
    from .approval import make_propose_opportunities_tool
    from .subagents import make_investigate_signals_tool

    return [make_investigate_signals_tool(settings, deps), make_propose_opportunities_tool(settings, deps)]


def _caps(settings: Settings, mode: Mode) -> tuple[int, int, int]:
    if mode == "scan":
        return settings.scan_max_model_calls, settings.scan_max_tool_calls, settings.scan_deadline_seconds
    return settings.chat_max_model_calls, settings.chat_max_tool_calls, settings.chat_deadline_seconds


def build_agent(settings: Settings, deps: AgentDeps, *, mode: Mode):
    model_calls, tool_calls, deadline = _caps(settings, mode)
    return create_agent(
        model=deps.models["orchestrator"],
        tools=[*deps.mcp_tools, *make_skill_tools(deps.skills), *agent_tools(settings, deps)],
        system_prompt=render_system_prompt(deps.skills, settings),
        middleware=[ToolErrorMiddleware(settings), TelemetryMiddleware("orchestrator", deps.sink, settings),
                    SkillsMiddleware(deps.skills), EarlierResultsMiddleware(), ProgressMiddleware(),
                    DeadlineMiddleware(deadline, skill_deadlines={SCAN_SKILL: settings.scan_deadline_seconds}),
                    *limit_middleware(model_calls, tool_calls), GroundingMiddleware()],
        context_schema=TurnContext, checkpointer=deps.checkpointer)


def final_answer_text(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content
                                  if isinstance(b, dict) and b.get("type", "text") == "text")
            return content or None
    return None


async def _mcp_tools(settings: Settings) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    try:
        return await MultiServerMCPClient(mcp_connection_config(settings)).get_tools()
    except Exception as e:  # noqa: BLE001  any transport failure means the same thing to the user
        raise McpUnavailable(type(e).__name__) from e


@asynccontextmanager
async def real_deps(settings: Settings, models: dict[Role, Runnable] | None = None):
    """The production dependencies: MCP tools, role models (or the given ones, as in an evaluation), database
    telemetry and Postgres checkpoints."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    tools = await _mcp_tools(settings)
    models = models or {role: build_chat_model(settings, role) for role in ("orchestrator", "investigator", "judge")}
    async with AsyncPostgresSaver.from_conn_string(settings.database_url(DbRole.APP)) as saver:
        yield AgentDeps(skills=load_skills(SKILLS_DIR), mcp_tools=tools, models=models,
                        embedder=get_embedder(settings), sink=DbTelemetrySink(settings), checkpointer=saver,
                        db_name=settings.db_name)


@asynccontextmanager
async def _given(deps: AgentDeps):
    yield deps


async def refresh_if_stale(settings: Settings, deps: AgentDeps) -> None:
    """Live mode only: pull new items first when the newest stored document is older than store_stale_minutes. A
    failed refresh is logged and the turn goes on with what is stored."""
    if not settings.live:
        return
    with connect(settings, DbRole.APP, deps.db_name) as conn:
        newest = conn.execute("SELECT max(fetched_at) FROM documents").fetchone()[0]
    if newest is not None and clock.now(settings) - newest < timedelta(minutes=settings.store_stale_minutes):
        return
    refresh = next((t for t in deps.mcp_tools if t.name == "refresh_sources"), None)
    if refresh is None:
        return
    try:
        await refresh.ainvoke({"kinds": ["news", "gazette"]})
    except Exception as e:  # noqa: BLE001  stale data still answers; the failure is logged, not hidden
        log.warning("refresh before the turn failed: %s", redact(f"{type(e).__name__}: {e}", settings))


def _failure(settings: Settings, error: BaseException) -> str:
    kind = classify_model_error(error)
    if kind == "access":
        return MODEL_ACCESS_MESSAGE
    detail = redact(f"{type(error).__name__}: {error}", settings)[:MAX_ERROR_CHARS]
    if kind in ("rate_limited", "timeout", "provider"):
        return f"The model provider failed ({kind}); please try again. Detail: {detail}"
    return f"The request failed: {detail}"


def _approval(state: dict):
    from .approval import ApprovalRequest

    interrupts = state.get("__interrupt__") or []
    return ApprovalRequest.model_validate(interrupts[0].value) if interrupts else None


def grounding_rewrites(messages: list[BaseMessage]) -> int:
    """How many rewrites the grounding check asked for in this turn: its requests since the user's last message."""
    return sum(1 for m in messages[turn_start(messages):]
               if isinstance(m, HumanMessage) and m.name == GROUNDING_MESSAGE_NAME)


def _result(thread_id: str, turn_id: str, state: dict) -> TurnResult:
    approval = _approval(state)
    messages = state.get("messages") or []
    return TurnResult(thread_id=thread_id, turn_id=turn_id,
                      answer=None if approval else final_answer_text(messages),
                      approval=approval, loaded_skills=list(state.get("loaded_skills") or []),
                      grounding_rewrites=grounding_rewrites(messages), charts=turn_charts(messages))


@contextmanager
def _conversation_lock(settings: Settings, thread_id: str, db_name: str | None):
    """Hold a Postgres advisory lock on the conversation for one turn, so two turns never run on it at once (two
    browser tabs ran two scans on one conversation, and the second one's pause replaced the first's). Yields whether
    the lock was taken; it is released when the turn ends, or with the connection if the process dies."""
    key = f"conversation:{thread_id}"
    with connect(settings, DbRole.APP, db_name, autocommit=True) as conn:
        taken = conn.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)).fetchone()[0]
        try:
            yield taken
        finally:
            if taken:
                conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))


async def _paused_for(agent, config: dict, approval_id: str) -> bool:
    """Whether the conversation's run is paused on this approval, so resuming it will reach the decisions."""
    snapshot = await agent.aget_state(config)
    return any(isinstance(i.value, dict) and i.value.get("approval_id") == approval_id for i in snapshot.interrupts)


def _apply_orphaned(settings: Settings, thread_id: str, resuming: tuple, db_name: str | None) -> TurnResult:
    """Decisions on cards whose paused run no longer exists (two turns ran on the conversation before it was locked)
    are applied directly, still exactly once."""
    from .approval import apply_decisions

    approval, decisions = resuming
    with connect(settings, DbRole.APP, db_name, autocommit=True) as conn:
        apply_decisions(conn, approval.approval_id, decisions)
    return TurnResult(thread_id=thread_id, turn_id="", answer=ORPHANED_DECISIONS_MESSAGE)


async def _invoke(settings: Settings, thread_id: str, payload: Any, *, mode: Mode, deps: AgentDeps | None,
                  refresh: bool, resuming: tuple | None = None,
                  progress: Callable[[str], None] | None = None) -> TurnResult:
    turn_id = uuid.uuid4().hex
    try:
        async with (_given(deps) if deps is not None else real_deps(settings)) as active:
            if refresh:
                await refresh_if_stale(settings, active)
            agent = build_agent(settings, active, mode=mode)
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
            if resuming is not None and not await _paused_for(agent, config, resuming[0].approval_id):
                return _apply_orphaned(settings, thread_id, resuming, active.db_name)
            state = await agent.ainvoke(payload, config, context=TurnContext(thread_id=thread_id, turn_id=turn_id,
                                                                             mode=mode, progress=progress))
            return _result(thread_id, turn_id, state)
    except McpUnavailable:
        return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=None, error=MCP_UNREACHABLE_MESSAGE)
    except Exception as e:  # noqa: BLE001  the boundary of a turn: report it plainly, keep the detail in the log
        log.exception("turn %s on thread %s failed", turn_id, thread_id)
        return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=None, error=_failure(settings, e))


async def run_turn(settings: Settings, thread_id: str, user_text: str, *, mode: Mode = "chat",
                   deps: AgentDeps | None = None, progress: Callable[[str], None] | None = None) -> TurnResult:
    """One user turn on a thread. Never raises for an expected failure: the TurnResult carries a plain message.
    When the agent proposes opportunity cards, the turn pauses and TurnResult.approval holds them; until the analyst
    decides, a new message on that thread is refused (it would leave the paused tool call without a reply), and so is
    a turn while another runs on the same conversation."""
    try:
        with _conversation_lock(settings, thread_id, deps.db_name if deps else None) as taken:
            if not taken:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=CONVERSATION_BUSY_MESSAGE)
            waiting = _waiting(settings, thread_id, deps)
            if waiting:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, approval=waiting[-1],
                                  error=PENDING_APPROVAL_MESSAGE)
            return await _invoke(settings, thread_id, {"messages": [HumanMessage(user_text)]}, mode=mode,
                                 deps=deps, refresh=True, progress=progress)
    except Exception as e:  # noqa: BLE001  a database failure is reported like any other turn failure
        return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=_failure(settings, e))


def _waiting(settings: Settings, thread_id: str, deps: AgentDeps | None) -> list:
    """The approvals waiting on this thread."""
    from .approval import pending_approvals

    with connect(settings, DbRole.APP, deps.db_name if deps else None) as conn:
        return pending_approvals(conn, thread_id)


async def resume_turn(settings: Settings, thread_id: str, decisions: list, *, mode: Mode = "scan",
                      deps: AgentDeps | None = None, progress: Callable[[str], None] | None = None) -> TurnResult:
    """Continue a paused turn with the analyst's decisions. The decisions are checked against the waiting approval
    first, so an incomplete set is refused while the approval stays open for a corrected retry. Decisions on an
    approval whose paused run no longer exists are applied directly."""
    from .approval import decision_problem

    try:
        with _conversation_lock(settings, thread_id, deps.db_name if deps else None) as taken:
            if not taken:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=CONVERSATION_BUSY_MESSAGE)
            waiting = _waiting(settings, thread_id, deps)
            if not waiting:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=NO_APPROVAL_MESSAGE)
            problem = decision_problem([c.card_id for c in waiting[-1].cards], decisions)
            if problem:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=problem)
            return await _invoke(settings, thread_id, Command(resume=[d.model_dump() for d in decisions]),
                                 mode=mode, deps=deps, refresh=False, resuming=(waiting[-1], decisions),
                                 progress=progress)
    except Exception as e:  # noqa: BLE001  a database failure is reported like any other turn failure
        return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=_failure(settings, e))


async def decide_card(settings: Settings, thread_id: str, approval_id: str, decision, *, mode: Mode = "scan",
                      deps: AgentDeps | None = None, progress: Callable[[str], None] | None = None) -> TurnResult:
    """Save the analyst's decision on one proposed card. It takes effect at once (an approved card reaches the team
    list); while cards remain, the result carries the approval as it now stands. After the last card the paused
    scan resumes with every decision and the result is its closing report."""
    from .approval import record_decision

    db_name = deps.db_name if deps else None
    try:
        with _conversation_lock(settings, thread_id, db_name) as taken:
            if not taken:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=CONVERSATION_BUSY_MESSAGE)
            with connect(settings, DbRole.APP, db_name, autocommit=True) as conn:
                recorded = record_decision(conn, approval_id, decision)
            if recorded.already_resolved:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=NO_APPROVAL_MESSAGE)
            if recorded.remaining:
                waiting = [a for a in _waiting(settings, thread_id, deps) if a.approval_id == approval_id]
                return TurnResult(thread_id=thread_id, turn_id="", answer=None,
                                  approval=waiting[0] if waiting else None)
            return await _finish(settings, thread_id, approval_id, mode=mode, deps=deps, progress=progress)
    except NotInData as e:
        return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=str(e))
    except Exception as e:  # noqa: BLE001  a database failure is reported like any other turn failure
        return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=_failure(settings, e))


async def finish_approval(settings: Settings, thread_id: str, approval_id: str, *, mode: Mode = "scan",
                          deps: AgentDeps | None = None,
                          progress: Callable[[str], None] | None = None) -> TurnResult:
    """Resume a scan whose cards are all decided, for when the resume after the last decision failed (a provider
    timeout, say): the decisions are already saved, so this only hands them to the paused run."""
    try:
        with _conversation_lock(settings, thread_id, deps.db_name if deps else None) as taken:
            if not taken:
                return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=CONVERSATION_BUSY_MESSAGE)
            return await _finish(settings, thread_id, approval_id, mode=mode, deps=deps, progress=progress)
    except Exception as e:  # noqa: BLE001  a database failure is reported like any other turn failure
        return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=_failure(settings, e))


async def _finish(settings: Settings, thread_id: str, approval_id: str, *, mode: Mode, deps: AgentDeps | None,
                  progress: Callable[[str], None] | None = None) -> TurnResult:
    """With the conversation locked: resume the paused scan with the approval's saved decisions."""
    from .approval import decision_problem, recorded_decisions

    waiting = [a for a in _waiting(settings, thread_id, deps) if a.approval_id == approval_id]
    if not waiting:
        return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=NO_APPROVAL_MESSAGE)
    with connect(settings, DbRole.APP, deps.db_name if deps else None) as conn:
        decisions = recorded_decisions(conn, approval_id)
    problem = decision_problem([c.card_id for c in waiting[0].cards], decisions)
    if problem:
        return TurnResult(thread_id=thread_id, turn_id="", answer=None, error=problem)
    return await _invoke(settings, thread_id, Command(resume=[d.model_dump() for d in decisions]), mode=mode,
                         deps=deps, refresh=False, resuming=(waiting[0], decisions), progress=progress)

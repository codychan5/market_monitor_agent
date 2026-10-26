"""Per-call records of every model and tool call: what each answer cost, how long it took, and which tools it used.
The same records feed the UI's "How I got this" view and the evaluation reports."""
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from collections.abc import Callable
from typing import Any, Literal, Protocol

import psycopg
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from psycopg.types.json import Jsonb

from ..config import Settings
from ..db.connect import DatabaseUnavailable, DbRole, connect
from ..redact import redact
from .llm import Role

log = logging.getLogger("london_office_monitor.telemetry")
MAX_ARGS_CHARS = 2000
MAX_OUTCOME_CHARS = 500
MAX_ERROR_CHARS = 500
PAUSED_OUTCOME = "paused for the analyst's approval"


@dataclass
class TurnContext:
    """Passed as context= on every agent invocation; middleware reads it from runtime.context (spike Q9)."""
    thread_id: str
    turn_id: str
    mode: Literal["chat", "scan", "eval"]
    progress: Callable[[str], None] | None = None


UNKNOWN_CONTEXT = TurnContext(thread_id="unknown", turn_id="unknown", mode="chat")


@dataclass
class ModelCallRecord:
    thread_id: str
    turn_id: str
    mode: str
    role: str
    model: str
    provider: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: Decimal | None
    latency_ms: int
    ok: bool
    error: str | None


@dataclass
class ToolCallRecord:
    thread_id: str
    turn_id: str
    mode: str
    tool: str
    args: dict[str, Any]
    outcome: str
    ms: int
    ok: bool


class TelemetrySink(Protocol):
    def model_call(self, record: ModelCallRecord) -> None: ...

    def tool_call(self, record: ToolCallRecord) -> None: ...


@dataclass
class MemorySink:
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def model_call(self, record: ModelCallRecord) -> None:
        self.model_calls.append(record)

    def tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)


class DbTelemetrySink:
    """Writes records as the app role. A failed write is logged and dropped: telemetry must never end a turn."""

    def __init__(self, settings: Settings, db_name: str | None = None):
        self.settings = settings
        self.db_name = db_name

    def _insert(self, sql: str, params: tuple) -> None:
        try:
            with connect(self.settings, DbRole.APP, self.db_name, autocommit=True) as conn:
                conn.execute(sql, params)
        except (psycopg.Error, DatabaseUnavailable) as e:
            log.warning("telemetry write failed: %s", redact(f"{type(e).__name__}: {e}", self.settings))

    def model_call(self, r: ModelCallRecord) -> None:
        self._insert("INSERT INTO model_calls (thread_id, turn_id, mode, role, model, provider, input_tokens,"
                     " output_tokens, cost_usd, latency_ms, ok, error) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                     " %s, %s)", (r.thread_id, r.turn_id, r.mode, r.role, r.model, r.provider, r.input_tokens,
                                  r.output_tokens, r.cost_usd, r.latency_ms, r.ok, r.error))

    def tool_call(self, r: ToolCallRecord) -> None:
        self._insert("INSERT INTO tool_calls (thread_id, turn_id, mode, tool, args, outcome, ms, ok)"
                     " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                     (r.thread_id, r.turn_id, r.mode, r.tool, Jsonb(r.args), r.outcome, r.ms, r.ok))


def cost_from_message(message: AIMessage) -> Decimal | None:
    """OpenRouter's per-call cost in USD, from usage accounting (spike Q1); None when the reply carries none."""
    cost = ((message.response_metadata or {}).get("token_usage") or {}).get("cost")
    return Decimal(str(cost)) if cost is not None else None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _model_record(ctx: TurnContext, role: str, model: str, message: AIMessage | None, latency_ms: int,
                  error: str | None) -> ModelCallRecord:
    usage = (message.usage_metadata or {}) if message is not None else {}
    metadata = (message.response_metadata or {}) if message is not None else {}
    return ModelCallRecord(thread_id=ctx.thread_id, turn_id=ctx.turn_id, mode=ctx.mode, role=role,
                           model=metadata.get("model_name") or model, provider=metadata.get("provider"),
                           input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                           cost_usd=cost_from_message(message) if message is not None else None,
                           latency_ms=latency_ms, ok=error is None, error=error)


def record_direct_call(sink: TelemetrySink, ctx: TurnContext, role: Role, model_name: str, message: AIMessage | None,
                       latency_ms: int, error: str | None = None) -> None:
    """For model calls made outside an agent loop, such as the judge's card review."""
    sink.model_call(_model_record(ctx, role, model_name, message, latency_ms, error))


def _configured_name(model: Any) -> str:
    return getattr(model, "model_name", None) or getattr(getattr(model, "runnable", None), "model_name", None) \
        or type(model).__name__


class TelemetryMiddleware(AgentMiddleware):
    """Records every model and tool call of one agent. Async hooks only: the agents always run under ainvoke, where
    sync-only hooks raise (spike Q5)."""

    def __init__(self, role: Role, sink: TelemetrySink, settings: Settings):
        super().__init__()
        self.role = role
        self.sink = sink
        self.settings = settings

    @staticmethod
    def _context(request) -> TurnContext:
        context = getattr(getattr(request, "runtime", None), "context", None)
        return context if isinstance(context, TurnContext) else UNKNOWN_CONTEXT

    def _error(self, e: BaseException) -> str:
        return redact(f"{type(e).__name__}: {e}", self.settings)[:MAX_ERROR_CHARS]

    async def awrap_model_call(self, request, handler):
        ctx, started = self._context(request), time.monotonic()
        try:
            response = await handler(request)
        except Exception as e:
            self.sink.model_call(_model_record(ctx, self.role, _configured_name(request.model), None,
                                               int((time.monotonic() - started) * 1000), self._error(e)))
            raise
        messages = getattr(response, "result", None) or [response]
        message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        self.sink.model_call(_model_record(ctx, self.role, _configured_name(request.model), message,
                                           int((time.monotonic() - started) * 1000), None))
        return response

    async def awrap_tool_call(self, request, handler):
        ctx, started = self._context(request), time.monotonic()
        call = request.tool_call
        # Redacting the JSON text keeps it valid JSON: the replacement contains no quotes.
        redacted = redact(json.dumps(call.get("args") or {}, default=str), self.settings)
        args = json.loads(redacted) if len(redacted) <= MAX_ARGS_CHARS else {"truncated": redacted[:MAX_ARGS_CHARS]}
        try:
            result = await handler(request)
        except GraphBubbleUp:
            # An approval interrupt is the tool working as designed: record the pause, then let it through.
            self.sink.tool_call(ToolCallRecord(thread_id=ctx.thread_id, turn_id=ctx.turn_id, mode=ctx.mode,
                                               tool=call.get("name", "?"), args=args, outcome=PAUSED_OUTCOME,
                                               ms=int((time.monotonic() - started) * 1000), ok=True))
            raise
        except Exception as e:
            self.sink.tool_call(ToolCallRecord(thread_id=ctx.thread_id, turn_id=ctx.turn_id, mode=ctx.mode,
                                               tool=call.get("name", "?"), args=args, outcome=self._error(e),
                                               ms=int((time.monotonic() - started) * 1000), ok=False))
            raise
        message = result if isinstance(result, ToolMessage) else next(
            (m for m in (getattr(result, "update", None) or {}).get("messages", []) if isinstance(m, ToolMessage)),
            None)
        outcome = _text(message.content) if message is not None else str(result)
        ok = not (message is not None and (message.status == "error" or outcome.startswith("ERROR")))
        self.sink.tool_call(ToolCallRecord(thread_id=ctx.thread_id, turn_id=ctx.turn_id, mode=ctx.mode,
                                           tool=call.get("name", "?"), args=args,
                                           outcome=redact(outcome, self.settings)[:MAX_OUTCOME_CHARS],
                                           ms=int((time.monotonic() - started) * 1000), ok=ok))
        return result


@dataclass
class TurnSummary:
    skills_loaded: list[str]
    tool_calls: list[ToolCallRecord]
    model_calls: list[ModelCallRecord]
    cost_usd: Decimal


def turn_summary(conn: psycopg.Connection, turn_id: str) -> TurnSummary:
    models = [ModelCallRecord(*row) for row in conn.execute(
        "SELECT thread_id, turn_id, mode, role, model, provider, input_tokens, output_tokens, cost_usd, latency_ms,"
        " ok, error FROM model_calls WHERE turn_id = %s ORDER BY id", (turn_id,))]
    tools = [ToolCallRecord(*row) for row in conn.execute(
        "SELECT thread_id, turn_id, mode, tool, args, outcome, ms, ok FROM tool_calls WHERE turn_id = %s ORDER BY id",
        (turn_id,))]
    skills = [t.args.get("name") for t in tools if t.tool == "load_skill" and t.ok and t.args.get("name")]
    return TurnSummary(skills_loaded=list(dict.fromkeys(skills)), tool_calls=tools, model_calls=models,
                       cost_usd=sum((m.cost_usd or Decimal(0) for m in models), Decimal(0)))

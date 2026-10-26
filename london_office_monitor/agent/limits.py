"""Bounds on every agent run: a wall-clock deadline, call caps, a wrap-up before them for agents that must return a
result, and tool failures turned into messages."""
import time
from collections.abc import Callable
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

from ..config import Settings
from ..redact import redact

DEADLINE_MESSAGE = "Stopped: the {seconds}-second limit for this request was reached."
WRAP_UP_INSTRUCTION = ("Your budget for this investigation is nearly spent: return your result now from what you have "
                       "found. If the evidence is too thin for a card, return a discard that says so.")
# The result call, plus one retry when the result fails validation (a summary over its word limit, say).
WRAP_UP_CALLS = 2


class WrapUpMiddleware(AgentMiddleware):
    """For an agent that must end with a structured result (ToolStrategy). Once its budget is nearly spent (its last
    WRAP_UP_CALLS model calls, its tool calls, or all but margin_seconds of its deadline) every model call is offered
    no tools, which leaves the result tool as the only choice: the run ends with a result instead of being cut off by
    a cap with none. Built once per run, like DeadlineMiddleware, so its call count is that run's."""

    def __init__(self, *, max_model_calls: int, max_tool_calls: int, deadline_seconds: int, margin_seconds: int,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__()
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.wrap_up_after = deadline_seconds - margin_seconds
        self._clock = clock
        self._started = clock()
        self._model_calls = 0

    def _spent(self, messages: list) -> bool:
        tool_calls = sum(len(m.tool_calls) for m in messages if isinstance(m, AIMessage))
        return (self._model_calls >= self.max_model_calls - WRAP_UP_CALLS or tool_calls >= self.max_tool_calls
                or self._clock() - self._started >= self.wrap_up_after)

    async def awrap_model_call(self, request, handler):
        spent = self._spent(request.messages)
        self._model_calls += 1
        if spent:
            request = request.override(tools=[], messages=[*request.messages, HumanMessage(WRAP_UP_INSTRUCTION)])
        return await handler(request)


class DeadlineMiddleware(AgentMiddleware):
    """Ends the loop before the next model call once this invocation has run for deadline_seconds, or for the longer
    deadline of a loaded skill in skill_deadlines (a scan asked for in chat takes minutes). The clock starts when the
    middleware is built, which happens once per turn. A tool call already running is not interrupted; the HTTP
    clients and the model client have their own timeouts."""

    def __init__(self, deadline_seconds: int, clock: Callable[[], float] = time.monotonic,
                 skill_deadlines: dict[str, int] | None = None):
        super().__init__()
        self.deadline_seconds = deadline_seconds
        self.skill_deadlines = skill_deadlines or {}
        self._clock = clock
        self._started = clock()

    def _check(self, state):
        loaded = (state or {}).get("loaded_skills") or []
        deadline = max([self.deadline_seconds, *(self.skill_deadlines[s] for s in loaded if s in self.skill_deadlines)])
        if self._clock() - self._started < deadline:
            return None
        return {"jump_to": "end", "messages": [AIMessage(DEADLINE_MESSAGE.format(seconds=deadline))]}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        return self._check(state)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return self._check(state)


class ToolErrorMiddleware(AgentMiddleware):
    """A tool that raises becomes an ERROR message the model can read and react to, never a crashed turn. Graph
    control signals (an approval interrupt) pass straight through."""

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:  # noqa: BLE001  deliberate: any tool failure must reach the model, not end the turn
            return ToolMessage(f"ERROR: {type(e).__name__}: {redact(str(e), self.settings)}",
                               tool_call_id=request.tool_call["id"], name=request.tool_call.get("name"),
                               status="error")


def limit_middleware(model_calls: int, tool_calls: int, *,
                     tool_exit: Literal["end", "continue"] = "end") -> list[AgentMiddleware]:
    """Per-invocation caps: a turn (or a resume) may spend at most this many model and tool calls. With
    tool_exit="continue", tool calls over the cap get an error message instead of ending the run, so that
    WrapUpMiddleware can still return a result; the model-call cap still ends it."""
    return [ModelCallLimitMiddleware(run_limit=model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=tool_calls, exit_behavior=tool_exit)]

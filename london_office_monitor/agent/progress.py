"""Live progress of a turn in plain English: each tool call, each investigation and its outcome, the judge and any
grounding rewrite. A turn reports to the optional TurnContext.progress callback; the web app shows the lines while
the turn runs. Reporting never affects the turn itself."""
import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .telemetry import TurnContext

log = logging.getLogger("london_office_monitor.progress")
MAX_QUERY_CHARS = 80


# How metrics read in a progress line; a metric not listed reads as its name with spaces.
METRIC_LABELS = {"prime_rent": "prime rent", "grade_a_rent": "Grade A rent", "vacancy_rate": "vacancy",
                 "take_up": "take-up", "availability": "availability", "under_construction": "space under construction",
                 "pre_let_share": "pre-let share", "under_offer": "space under offer", "prime_yield": "prime yield",
                 "investment_volume": "investment volume"}


def _words(value: Any) -> str:
    return METRIC_LABELS.get(str(value), str(value).replace("_", " "))


def _quoted(text: Any) -> str:
    text = str(text)
    return text if len(text) <= MAX_QUERY_CHARS else text[:MAX_QUERY_CHARS - 1] + "…"


def _count(items: Any, noun: str) -> str:
    n = len(items) if isinstance(items, list) else 0
    return f"{n} {noun}{'s' * (n != 1)}"


TOOL_STEPS: dict[str, Callable[[dict], str]] = {
    "load_skill": lambda a: f"Loading the {a.get('name', '?')} playbook",
    "read_skill_file": lambda a: f"Reading {a.get('path', '?')} from the {a.get('skill', '?')} playbook",
    "get_coverage": lambda a: "Checking what the data covers",
    "get_metric": lambda a: f"Looking up {_words(a.get('metric', '?'))} in {a.get('submarket', '?')}",
    "compare_brokers": lambda a: (f"Comparing brokers on {_words(a.get('metric', '?'))} in {a.get('submarket', '?')}, "
                                  f"{a.get('period', '?')}"),
    "metric_change": lambda a: (f"Computing {a.get('broker', '?')}'s change in {_words(a.get('metric', '?'))}, "
                                f"{a.get('from_period', '?')} to {a.get('to_period', '?')}"),
    "get_macro": lambda a: f"Fetching Bank of England series {a.get('series', '?')}",
    "metric_trend": lambda a: f"Charting {_words(a.get('metric', '?'))} in {a.get('submarket', '?')}",
    "search_documents": lambda a: f"Searching the news: “{_quoted(a.get('query', ''))}”",
    "refresh_sources": lambda a: "Refreshing the news and insolvency notices",
    "get_portfolio": lambda a: "Reading Nan Fung's portfolio",
    "get_mandate": lambda a: "Reading the mandate",
    "match_to_mandate": lambda a: f"Matching D{a.get('doc_id', '?')} to the mandate",
    "similar_past_decisions": lambda a: "Checking your past decisions",
    "get_citation": lambda a: f"Opening source {a.get('citation_id', '?')}",
    "investigate_signals": lambda a: f"Sending {_count(a.get('candidates'), 'lead')} to investigator sub-agents",
    "propose_opportunities": lambda a: f"Proposing {_count(a.get('card_ids'), 'card')} for your decision",
}


def describe_tool_call(name: str, args: dict | None) -> str:
    step = TOOL_STEPS.get(name)
    return step(args or {}) if step else f"Using {name}"


def report(ctx: TurnContext | None, line: str) -> None:
    """Send one progress line to the turn's listener, if any; a failing listener is logged, never raised."""
    listener = getattr(ctx, "progress", None)
    if listener is None:
        return
    try:
        listener(line)
    except Exception as e:  # noqa: BLE001  progress is a convenience: it must never end a turn
        log.warning("progress listener failed: %s", type(e).__name__)


class ProgressMiddleware(AgentMiddleware):
    """Reports each tool call as it starts, and a failed one when it ends. Placed inside the skills gate, so only
    calls that actually run are reported."""

    async def awrap_tool_call(self, request, handler):
        ctx = getattr(getattr(request, "runtime", None), "context", None)
        call = request.tool_call
        step = describe_tool_call(call.get("name", "?"), call.get("args"))
        report(ctx, step)
        result = await handler(request)
        if isinstance(result, ToolMessage) and result.status == "error":
            report(ctx, f"⚠ {step}: failed")
        return result

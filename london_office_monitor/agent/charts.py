"""Charts for an answer, built from this turn's trend and rate results: the agent asks the data server for a series
(metric_trend or get_macro) and the app draws it. Every point comes from a tool result, never from the model."""
import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from ..facts import period_key
from .grounding import GROUNDING_MESSAGE_NAME
from .progress import METRIC_LABELS

CHART_TOOLS = ("metric_trend", "get_macro")


@dataclass(frozen=True)
class Chart:
    title: str
    unit: str
    x: list[str]
    series: dict[str, list[tuple[str, float]]]
    citations: list[str]
    x_kind: Literal["period", "date"] = "period"


def _json(message: ToolMessage) -> Any:
    content = message.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return None


def _this_turn(messages: list[BaseMessage]) -> list[ToolMessage]:
    """The tool results since the user's last message (grounding nudges do not start a turn)."""
    start = max((i for i, m in enumerate(messages)
                 if isinstance(m, HumanMessage) and m.name != GROUNDING_MESSAGE_NAME), default=0)
    return [m for m in messages[start:] if isinstance(m, ToolMessage) and m.name in CHART_TOOLS]


def _trend_chart(data: dict) -> Chart | None:
    series = {s["broker"]: [(p["period"], float(p["value"])) for p in s.get("points", [])]
              for s in data.get("series", []) if s.get("points")}
    if not series:
        return None
    label = METRIC_LABELS.get(data.get("metric", ""), str(data.get("metric", "")).replace("_", " "))
    x = sorted({period for points in series.values() for period, _ in points}, key=period_key)
    return Chart(title=f"{label[:1].upper()}{label[1:]} · {data.get('submarket', '')}", unit=data.get("unit", ""),
                 x=x, series={name: sorted(points, key=lambda p: period_key(p[0])) for name, points in series.items()},
                 citations=list(data.get("citations", [])))


def _macro_chart(results: list[dict]) -> Chart | None:
    """Every rate series fetched in the turn on one chart: they share a unit, and the gap between them (Bank Rate
    against the gilt yield) is usually the point."""
    by_name: dict[str, dict[str, float]] = {}
    for data in results:
        name = data.get("name") or data.get("series", "")
        by_name.setdefault(name, {}).update({p["day"]: float(p["value"]) for p in data.get("points", [])})
    series = {name: sorted(points.items()) for name, points in by_name.items() if points}
    if not series:
        return None
    days = sorted({day for points in series.values() for day, _ in points})
    return Chart(title=" and ".join(series), unit="%", x=days, series=series, citations=[], x_kind="date")


def turn_charts(messages: list[BaseMessage]) -> list[Chart]:
    """A chart for each successful trend result in this turn, and one for all its rate results, in the order the
    agent fetched them."""
    charts: list[Chart] = []
    rates: list[dict] = []
    rates_at = 0  # where the rate chart goes: after the trends fetched before the first rate result
    for message in _this_turn(messages):
        data = _json(message)
        if not isinstance(data, dict) or "error" in data:
            continue
        if message.name == "metric_trend":
            chart = _trend_chart(data)
            if chart is not None:
                charts.append(chart)
        else:
            if not rates:
                rates_at = len(charts)
            rates.append(data)
    rate_chart = _macro_chart(rates) if rates else None
    if rate_chart is not None:
        charts.insert(rates_at, rate_chart)
    return charts

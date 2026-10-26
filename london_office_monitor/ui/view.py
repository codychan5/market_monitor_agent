"""What the web app shows, as plain functions of the agent's results, so it can be tested without Streamlit: cited
answers with links, the Sources list, "How I got this", opportunity cards and charts."""
import json
import re
from collections.abc import Callable
from datetime import date, datetime

from ..agent.cards import ScoreInputs, StoredCard
from ..agent.charts import Chart
from ..agent.grounding import UNVERIFIED_MARKER
from ..agent.telemetry import TurnSummary
from ..citations import CITATION_PATTERN, CitationView
from ..scoring import COMPONENT_MAXIMUMS, MAX_SCORE
from ..vocab import MACRO_SERIES, Broker

Resolve = Callable[[str], CitationView | None]
SYNTHETIC_SOURCE = "[SYNTHETIC] made-up news item for the demo, with no original to open"
MOCK_SOURCE = "Mock internal document written for this proof of concept, not a Nan Fung record"
# Made-up demo items keep their [SYNTHETIC] and [MOCK] labels in the data, the agent's prompts and the evals; on
# screen the labels read as plain grey badges, explained by DEMO_HELP.
DEMO_BADGES = {"[SYNTHETIC]": ":gray-badge[Demo example]", "[MOCK]": ":gray-badge[Demo document]"}
DEMO_HELP = ("Demo example: a made-up news item standing in for a real report, so the demo has events to find. "
             "Demo document: a made-up internal document, not a Nan Fung record.")
MAX_ARG_CHARS = 60
SCORE_PART_LABELS = {"mandate": "Mandate fit", "signal": "Signal type", "recency": "Recency",
                     "confidence": "Confidence"}
# A card in words anyone on the team reads at a glance (the owner is not a property-finance specialist).
SIGNAL_WORDS = {"distress": "Owner under pressure", "demand": "Demand for space", "supply": "New supply",
                "pricing": "Rents and prices", "policy": "Rules and taxes", "other": "Other news"}
CONFIDENCE_WORDS = {"high": "Quite sure", "medium": "Fairly sure", "low": "Not sure yet"}
KIND_BADGES = {"chance_to_buy": ":green-badge[:material/check: Chance to buy]",
               "chance_to_lease": ":blue-badge[:material/check: Chance to lease]",
               "risk": ":orange-badge[:material/warning: Risk to our buildings]",
               "watch": ":violet-badge[:material/visibility: Worth watching]"}
# Steps written on one line as "1. Ask … 2. Use …"; a digit inside a figure such as £1.5m is never a step.
STEP_NUMBER = re.compile(r"(?:^|\s)\d[.)]\s+")

# Each series keeps one colour and marker on every chart: brokers by their order in the vocabulary, rates by theirs.
# The colours are the dataviz reference palette, checked with its validator on 25 Oct 2026 against the app's light
# (#ffffff) and dark (#0e1117) surfaces; the markers carry identity where two colours sit close under colour blindness.
SERIES_COLOURS = {"light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
                  "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")}
SERIES_SHAPES = ("circle", "square", "triangle-up", "diamond", "triangle-down", "cross", "triangle-right",
                 "triangle-left")
SERIES_SLOTS = {**{broker.value: i for i, broker in enumerate(Broker)},
                **{name: i for i, name in enumerate(MACRO_SERIES.values())}}
OVERFLOW_COLOUR = "#898781"  # a ninth series is never a new hue
SURFACES = {"light": "#ffffff", "dark": "#0e1117"}
CHART_HEIGHT = 320
LINE_WIDTH = 2
MARKER_AREA = 100  # px², a 10 px marker (triangles and diamonds look smaller than circles of the same area)
MARKER_RING = 2  # px of surface colour around each marker, so crossing marks stay legible
AXIS_PADDING = 16  # px between the highest or lowest point and the plot's edge
DODGE_STEP = 0.06  # how far apart, in periods, neighbouring series sit within one period
MAX_MARKED_DATES = 12  # past this many daily points, markers break a line into what looks like dashes


def cited(text: str) -> list[str]:
    """The citation markers in a text, each once, in order of first appearance."""
    return list(dict.fromkeys(CITATION_PATTERN.findall(text or "")))


def _link(citation: str, resolve: Resolve) -> str:
    view = resolve(citation)
    if view is None or view.synthetic or not view.open_url:
        return f"[{citation}]"
    # Escaped brackets stay visible in the link text, so adjacent markers do not run together.
    return f"[\\[{citation}\\]]({view.open_url})"


def linkify_citations(text: str, resolve: Resolve) -> str:
    """Each [F3] or [D9#0] marker as a link to its original; a marker that cannot be opened stays plain text."""
    return CITATION_PATTERN.sub(lambda m: _link(m.group(1), resolve), text)


def md_safe(text: str) -> str:
    """Text safe for Streamlit markdown, which reads $...$ as LaTeX."""
    return text.replace("$", "\\$")


def has_demo_label(text: str) -> bool:
    return any(label in text for label in DEMO_BADGES)


def screen_markdown(text: str) -> str:
    """Markdown as the app shows it: safe for Streamlit (md_safe), with each demo label as its badge."""
    shown = md_safe(text)
    for label, badge in DEMO_BADGES.items():
        shown = shown.replace(label, badge)
    return shown


def _day(value: date | datetime | None) -> str:
    return value.strftime("%d %b %Y") if value else "unknown"


def _source_entry(view: CitationView) -> str:
    if view.kind == "figure":
        lines = [f"**[{view.citation}]** {view.figure}",
                 (f"{view.publisher}, *{view.title}* (published {_day(view.published_at)}, checked "
                  f"{_day(view.retrieved_at)})")]
        if view.definition:
            lines.append(f"Definition: {view.definition}")
        lines.append(f"[Open original]({view.open_url or view.url})")
        return "  \n".join(lines)
    head = f"**[{view.citation}]** {view.publisher}, *{view.title}*"
    if view.synthetic:
        note = MOCK_SOURCE if view.source_kind == "internal" else SYNTHETIC_SOURCE
        return "  \n".join([head, note, f"> {view.quoted_passage}"])
    lines = [head, f"Published {_day(view.published_at)}, retrieved {_day(view.retrieved_at)}",
             f"> {view.quoted_passage}", "", f"[Open original]({view.open_url or view.url})"]
    if view.also_reported_by:
        lines[-1] += f" · Also reported by: {', '.join(view.also_reported_by)}"
    return "  \n".join(lines)


def sources_block(views: list[CitationView]) -> str:
    """One entry per cited figure or passage, in the order given: what it is, where it came from, and a link to
    check it in the original."""
    return "\n\n".join(_source_entry(v) for v in views)


def _short(value) -> str:
    text = str(value)
    return text if len(text) <= MAX_ARG_CHARS else text[:MAX_ARG_CHARS - 1] + "…"


def _grounding_line(rewrites: int, answer: str) -> str:
    if UNVERIFIED_MARKER in (answer or ""):
        return "Grounding: some figures or citations did not match the data; the answer says which."
    if rewrites == 0:
        return "Grounding: every figure and citation matched the data retrieved."
    return f"Grounding: every figure and citation matched the data, after {rewrites} rewrite{'s' * (rewrites > 1)}."


def how_i_got_this(summary: TurnSummary, *, grounding_rewrites: int, answer: str) -> list[str]:
    """The trail behind one answer: skills loaded, each tool call, the grounding result, and each model call with its
    tokens and cost."""
    lines = [f"Skills: {', '.join(summary.skills_loaded) or 'none'}"]
    for call in summary.tool_calls:
        if call.tool == "load_skill":
            continue
        args = ", ".join(f"{key}={_short(value)}" for key, value in (call.args or {}).items())
        lines.append(f"Tool: {call.tool}({args})" + ("" if call.ok else " (failed)"))
    lines.append(_grounding_line(grounding_rewrites, answer))
    for m in summary.model_calls:
        cost = f"${m.cost_usd:.4f}" if m.cost_usd is not None else "cost unknown"
        lines.append(f"Model: {m.role} · {m.model} · {m.input_tokens or 0:,} in / {m.output_tokens or 0:,} out · "
                     f"{cost}" + ("" if m.ok else " (failed)"))
    lines.append(f"Total cost: ${summary.cost_usd:.4f}")
    return lines


def _evidence_line(citation: str, quote: str, resolve: Resolve) -> str:
    view = resolve(citation)
    if view is None or not view.synthetic:
        flag = ""
    else:
        flag = " [MOCK]" if view.source_kind == "internal" else " [SYNTHETIC]"
    return f"- {_link(citation, resolve)}{flag} “{quote}”"


def _mandate_input(inputs: ScoreInputs) -> str:
    if inputs.closest_match is None:
        return "no portfolio asset or mandate statement to compare with"
    target = (f"{inputs.closest_match} (portfolio asset)" if inputs.closest_match_kind == "asset"
              else f"the mandate statement “{inputs.closest_match}”")
    return f"similarity {inputs.mandate_similarity:.3f} to {target}"


def _evidence_age(days: int) -> str:
    if days == 0:
        return "newest evidence under a day old"
    return f"newest evidence {days} day{'s' * (days != 1)} old"


def score_breakdown(card: StoredCard) -> tuple[str, list[str]]:
    """The score out of its maximum, and each part out of its own maximum with the input it was computed from, so
    every point can be traced. A card scored before the inputs were kept shows the points alone, except the signal
    type and confidence, which are on the card itself."""
    headline = f"Score {card.score}/{MAX_SCORE}" if card.score is not None else "Score not computed"
    inputs = card.score_inputs
    explained = {"mandate": _mandate_input(inputs) if inputs else None, "signal": card.signal_type.value,
                 "recency": _evidence_age(inputs.evidence_age_days) if inputs else None,
                 "confidence": card.confidence}
    parts = []
    for name, maximum in COMPONENT_MAXIMUMS.items():
        if name not in card.score_components:
            continue
        part = f"{SCORE_PART_LABELS[name]} {card.score_components[name]}/{maximum}"
        parts.append(f"{part}: {explained[name]}" if explained[name] else part)
    return headline, parts


def next_steps_markdown(text: str) -> str:
    """The next step as written, or as a numbered list when it holds numbered steps on one line."""
    steps = [step.strip() for step in STEP_NUMBER.split(text.strip()) if step.strip()]
    if not text.strip().startswith("1") or len(steps) < 2:
        return text
    return "\n".join(f"{number}. {step}" for number, step in enumerate(steps, 1))


def card_markdown(card: StoredCard) -> str:
    """An opportunity card as anyone on the team reads it first, in plain words: what it is to us (a chance to buy
    or lease, a risk, or one to watch) and its kind of signal, what happened, what it means for us, what to do next
    and how sure we are. Its sources and how its score was made up
    are in card_details_markdown."""
    chips = [KIND_BADGES[card.kind.value]] if card.kind else []
    chips += [f":gray-badge[{SIGNAL_WORDS[card.signal_type.value]}]", f":gray-badge[{card.submarket.value}]"]
    if card.score is not None:
        chips.append(f":gray-badge[Score {card.score}/{MAX_SCORE}]")
    ours = [f"*Our buildings involved: {', '.join(card.assets_affected)}*", ""] if card.assets_affected else []
    return "\n".join([
        " ".join(chips),
        f"#### {card.title}",
        "",
        "**What happened**  ",
        card.summary,
        "",
        "**What it means for us**  ",
        card.why_it_matters,
        "",
        *ours,
        "**What to do next**",
        "",
        next_steps_markdown(card.suggested_next_step),
        "",
        f"**How sure we are:** {CONFIDENCE_WORDS[card.confidence]}. {card.confidence_reason}",
    ])


def card_details_markdown(card: StoredCard, resolve: Resolve) -> str:
    """The sources behind a card, each linked to its original where it can be opened, and how its score was made
    up, part by part, with the input each part came from."""
    headline, parts = score_breakdown(card)
    evidence = [_evidence_line(f"D{e.id}#{e.chunk_index or 0}" if e.kind == "document" else f"F{e.id}",
                               e.quote_or_value, resolve) for e in card.evidence]
    return "\n".join(["**Sources**", *evidence, "", f"**{headline}**", *(f"- {part}" for part in parts)])


def _slots(names: list[str]) -> list[int]:
    """Each series' fixed slot; a series the vocabulary does not name takes the next slot no named one uses."""
    taken = {SERIES_SLOTS[n] for n in names if n in SERIES_SLOTS}
    free = (i for i in range(len(SERIES_SHAPES)) if i not in taken)
    return [SERIES_SLOTS[n] if n in SERIES_SLOTS else next(free, len(SERIES_SHAPES)) for n in names]


def _period_x(chart: Chart) -> tuple[dict, list[dict]]:
    """Periods as positions 0, 1, 2… labelled with their names, each series shifted a little within its period so
    equal figures (three brokers at £95) sit side by side instead of hiding one another."""
    names = list(chart.series)
    position = {period: i for i, period in enumerate(chart.x)}
    shift = {name: (i - (len(names) - 1) / 2) * DODGE_STEP for i, name in enumerate(names)}
    data = [{"series": name, "period": period, "x": position[period] + shift[name], "value": value}
            for name, points in chart.series.items() for period, value in points]
    labels = ",".join(json.dumps(period) for period in chart.x)
    axis = {"field": "x", "type": "quantitative", "title": "Period",
            "scale": {"domain": [-0.5, len(chart.x) - 0.5], "nice": False, "zero": False},
            "axis": {"values": list(range(len(chart.x))), "labelExpr": f"[{labels}][datum.value]", "grid": False,
                     "labelAngle": 0}}
    return axis, data


def chart_spec(chart: Chart, *, dark: bool = False) -> dict:
    """A Vega-Lite line chart: 2 px lines, each series in its fixed colour and marker, a value axis fitted to the data
    (a zero baseline flattens a move from £95 to £100), and a tooltip on every point. Points carry markers unless a
    dated series is dense, so a broker with one period still shows."""
    mode = "dark" if dark else "light"
    names = list(chart.series)
    slots = _slots(names)
    colours = [SERIES_COLOURS[mode][s] if s < len(SERIES_SHAPES) else OVERFLOW_COLOUR for s in slots]
    shapes = [SERIES_SHAPES[s % len(SERIES_SHAPES)] for s in slots]
    dated = chart.x_kind == "date"
    if dated:
        x = {"field": "x", "type": "temporal", "timeUnit": "utcyearmonthdate", "title": "Date",
             "axis": {"format": "%-d %b", "labelAngle": 0}}
        x_tip = {"field": "x", "type": "temporal", "timeUnit": "utcyearmonthdate", "title": "Date",
                 "format": "%-d %b %Y"}
        data = [{"series": name, "x": day, "value": value} for name, points in chart.series.items()
                for day, value in points]
    else:
        x, data = _period_x(chart)
        x_tip = {"field": "period", "type": "ordinal", "title": "Period"}
    marked = not dated or max(len(points) for points in chart.series.values()) <= MAX_MARKED_DATES
    legend = {"orient": "bottom", "title": None} if len(names) > 1 else None
    layers: list[dict] = [{"mark": {"type": "line", "strokeWidth": LINE_WIDTH, "strokeCap": "round",
                                    "strokeJoin": "round"}}]
    if marked:
        layers.append({"mark": {"type": "point", "filled": True, "opacity": 1, "size": MARKER_AREA,
                                "stroke": SURFACES[mode], "strokeWidth": MARKER_RING},
                       "encoding": {"shape": {"field": "series", "type": "nominal", "legend": legend,
                                              "scale": {"domain": names, "range": shapes}}}})
    return {
        "data": {"values": data},
        "height": CHART_HEIGHT,
        "encoding": {
            "x": x,
            "y": {"field": "value", "type": "quantitative", "title": chart.unit,
                  "scale": {"zero": False, "nice": True, "padding": AXIS_PADDING}},
            "color": {"field": "series", "type": "nominal", "legend": legend,
                      "scale": {"domain": names, "range": colours}},
            "tooltip": [{"field": "series", "type": "nominal", "title": "Series"}, x_tip,
                        {"field": "value", "type": "quantitative", "title": chart.unit}],
        },
        "layer": layers,
    }


def _day_month_year(day: date) -> str:
    return f"{day.day} {day:%b %Y}"


def shown_window(chart: Chart) -> str:
    """The days a dated chart covers, such as "10 Sep – 21 Oct 2026" (the year once when both ends share it)."""
    first, last = date.fromisoformat(chart.x[0]), date.fromisoformat(chart.x[-1])
    start = f"{first.day} {first:%b}" if first.year == last.year else _day_month_year(first)
    return f"{start} – {_day_month_year(last)}"

"""The number and citation check on every final answer: each figure must match a value a tool returned in this
turn (or a cited passage, or the user's own question), and each citation must point at something retrieved."""
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..citations import CITATION_PATTERN
from .progress import report

GROUNDING_MESSAGE_NAME = "grounding_check"
MAX_GROUNDING_REWRITES = 2
REWRITE_MESSAGE = ("These figures or citations could not be matched to this turn's tool results: {problems}. "
                   "Use only figures and citations from tool results, and put a figure quoted from a passage "
                   "in a sentence that cites that passage. Reply with the corrected answer only, written for the "
                   "analyst; do not mention this check.")
# Several citations in one bracket ("[F59, F50]"): split before checking, so each is checked and linked on its own.
CITATION_LIST = re.compile(r"\[((?:F\d+|D\d+#\d+)(?:\s*[,;]\s*(?:F\d+|D\d+#\d+))+)\]")
UNVERIFIED_MARKER = "⚠️ Unverified"
UNVERIFIED_WARNING = "\n\n" + UNVERIFIED_MARKER + ": {problems}. They did not match the data retrieved for this answer."

# A figure is a number that carries a currency before it or a unit after it; bare numbers (years, quarters, list
# numbers, counts) are not checked. ASCII letters before a number (Q3, F12, D34) mean it is a label, not a figure;
# Chinese text puts no space before numbers, so CJK characters do not block a match.
CURRENCY = r"(?P<cur>HK\$|US\$|GBP\s?|£|\$|€)"
NUMBER = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
UNIT = (r"(?P<unit>%|bps?\b|basis points?\b|sq\s?ft\b|sqft\b|sq\s?m\b|psf\b|per sq ft\b|m\b|bn\b|million\b|"
        r"billion\b|k\b|x\b|英鎊|港元|美元|平方英尺|億|萬|個百分點)")
TOKEN = re.compile(rf"(?<![A-Za-z0-9.#]){CURRENCY}?{NUMBER}(?:\s?{UNIT})?")
# "6.3–8.4%": the first number of a range takes the unit of the second.
RANGE_JOIN = re.compile(r"\s*(?:–|—|-|to)\s*")
BARE_NUMBER = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|\n+")
QUOTED_KEYS = frozenset({"quoted_title", "quoted_passage"})
# Fields that hold dates, ids, links or labels: their digits are not figures and must not vouch for one.
SKIP_KEYS = frozenset({"published_at", "retrieved_at", "searched_until", "url", "open_url", "doc_id", "citation",
                       "citations", "story_id", "chunk_index", "fact_id", "id", "day", "first_day", "last_day",
                       "date", "period", "from_period", "to_period", "period_note", "heading_path", "source_kind"})
# Tools whose output is a playbook, not data: their numbers are instructions ("under 250 words"), never figures.
NON_DATA_TOOLS = frozenset({"load_skill", "read_skill_file"})


@dataclass(frozen=True)
class CheckedNumber:
    text: str
    value: Decimal
    decimals: int
    start: int


@dataclass(frozen=True)
class ToolValues:
    fact_values: frozenset[Decimal]
    doc_values: dict[str, frozenset[Decimal]]


def _decimal(num: str) -> tuple[Decimal, int]:
    plain = num.replace(",", "")
    return Decimal(plain), len(plain.split(".")[1]) if "." in plain else 0


def checked_numbers(answer: str) -> list[CheckedNumber]:
    """The figures in an answer: numbers with a currency or unit, plus the first number of a unit-bearing range."""
    text = CITATION_PATTERN.sub(lambda m: " " * len(m.group(0)), answer)
    matches = list(TOKEN.finditer(text))
    found = []
    for i, m in enumerate(matches):
        checked = bool(m.group("cur") or m.group("unit"))
        if not checked and i + 1 < len(matches):
            join = RANGE_JOIN.match(text, m.end())
            following = matches[i + 1]
            checked = (join is not None and join.end() == following.start()
                       and bool(following.group("cur") or following.group("unit")))
        if checked:
            value, decimals = _decimal(m.group("num"))
            found.append(CheckedNumber(text=m.group(0).strip(), value=value, decimals=decimals, start=m.start()))
    return found


def matches(number: CheckedNumber, value: Decimal) -> bool:
    """True when the value, rounded to the decimals the answer shows, equals the answer's number (sign ignored)."""
    try:
        return value.copy_abs().quantize(Decimal(1).scaleb(-number.decimals), ROUND_HALF_UP) == number.value
    except InvalidOperation:
        return False


def _numbers_in(text: str) -> set[Decimal]:
    return {_decimal(m.group(1))[0] for m in BARE_NUMBER.finditer(text)}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type", "text") == "text")
    return str(content)


def turn_start(messages: list[BaseMessage]) -> int:
    """Where the current turn starts: the index of the user's last message. Grounding nudges do not start a turn."""
    return max((i for i, m in enumerate(messages)
                if isinstance(m, HumanMessage) and m.name != GROUNDING_MESSAGE_NAME), default=0)


def _turn_messages(messages: list[BaseMessage]) -> tuple[HumanMessage | None, list[ToolMessage]]:
    """The turn's user message and the tool results since it."""
    start = turn_start(messages)
    human = messages[start] if messages and isinstance(messages[start], HumanMessage) else None
    tools = [m for m in messages[start:] if isinstance(m, ToolMessage) and m.name not in NON_DATA_TOOLS]
    return human, tools


def _parsed(message: ToolMessage) -> Any:
    text = _content_text(message.content)
    try:
        return json.loads(text)
    except ValueError:
        return text


def _walk(node: Any, facts: set[Decimal], docs: dict[str, set[Decimal]]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("citation"), str) and QUOTED_KEYS & node.keys():
            if node.get("is_table"):
                return  # tables are never a source of figures
            for key, value in node.items():
                if key in QUOTED_KEYS and isinstance(value, str):
                    docs.setdefault(node["citation"], set()).update(_numbers_in(value))
                elif key not in SKIP_KEYS:
                    _walk(value, facts, docs)
            return
        for key, value in node.items():
            if key not in SKIP_KEYS:
                _walk(value, facts, docs)
    elif isinstance(node, list):
        for value in node:
            _walk(value, facts, docs)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        facts.add(Decimal(str(node)))
    elif isinstance(node, str):
        facts.update(_numbers_in(node))


def tool_values(messages: list[BaseMessage]) -> ToolValues:
    """Every number the tools returned in this turn (and the user's own numbers), split into figures and numbers
    quoted from each cited passage."""
    human, tool_messages = _turn_messages(messages)
    facts: set[Decimal] = set(_numbers_in(_content_text(human.content))) if human else set()
    docs: dict[str, set[Decimal]] = {}
    for message in tool_messages:
        _walk(_parsed(message), facts, docs)
    return ToolValues(frozenset(facts), {k: frozenset(v) for k, v in docs.items()})


def _collect_citations(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "citation" and isinstance(value, str):
                found.add(value)
            elif key == "citations" and isinstance(value, list):
                found.update(v for v in value if isinstance(v, str))
            else:
                _collect_citations(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_citations(value, found)


def retrieved_citations(messages: list[BaseMessage]) -> frozenset[str]:
    found: set[str] = set()
    for message in _turn_messages(messages)[1]:
        _collect_citations(_parsed(message), found)
    return frozenset(found)


def _sentence_at(answer: str, position: int) -> str:
    start = 0
    for brk in SENTENCE_BREAK.finditer(answer):
        if brk.start() >= position:
            return answer[start:brk.start()]
        start = brk.end()
    return answer[start:]


def unmatched(answer: str, values: ToolValues) -> list[str]:
    """Figures in the answer that no tool value accounts for. A figure quoted from a passage must sit in a sentence
    that cites that passage."""
    problems: list[str] = []
    for number in checked_numbers(answer):
        if any(matches(number, v) for v in values.fact_values):
            continue
        cited = CITATION_PATTERN.findall(_sentence_at(answer, number.start))
        if any(any(matches(number, v) for v in values.doc_values.get(c, ())) for c in cited):
            continue
        if number.text not in problems:
            problems.append(number.text)
    return problems


def unretrieved(answer: str, retrieved: frozenset[str]) -> list[str]:
    return list(dict.fromkeys(c for c in CITATION_PATTERN.findall(answer) if c not in retrieved))


def split_combined_citations(text: str) -> str:
    """"[F59, F50]" as "[F59][F50]": a model sometimes groups citations, which the citation pattern, the check and
    the app's links all read as one marker each."""
    return CITATION_LIST.sub(lambda m: "".join(f"[{c.strip()}]" for c in re.split(r"[,;]", m.group(1))), text)


class GroundingMiddleware(AgentMiddleware):
    """After the model's final answer: ask for a rewrite (at most twice per turn) when a figure or citation does not
    match the turn's tool results; after that, keep the answer but append a visible warning naming them."""

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        original = _content_text(last.content)
        answer = split_combined_citations(original)
        problems = unmatched(answer, tool_values(messages)) + \
            [f"[{c}]" for c in unretrieved(answer, retrieved_citations(messages))]
        if not problems:
            return None if answer == original else {"messages": [AIMessage(content=answer, id=last.id)]}
        human_index = max((i for i, m in enumerate(messages)
                           if isinstance(m, HumanMessage) and m.name != GROUNDING_MESSAGE_NAME), default=0)
        rewrites = sum(1 for m in messages[human_index:]
                       if isinstance(m, HumanMessage) and m.name == GROUNDING_MESSAGE_NAME)
        listed = ", ".join(problems)
        if rewrites < MAX_GROUNDING_REWRITES:
            report(getattr(runtime, "context", None),
                   f"A figure or citation did not match the data: rewriting ({rewrites + 1} of "
                   f"{MAX_GROUNDING_REWRITES})")
            return {"messages": [HumanMessage(REWRITE_MESSAGE.format(problems=listed), name=GROUNDING_MESSAGE_NAME)],
                    "jump_to": "model"}
        return {"messages": [AIMessage(content=answer + UNVERIFIED_WARNING.format(problems=listed), id=last.id)]}

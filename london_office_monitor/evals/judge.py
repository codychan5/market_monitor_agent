"""The judge comparison behind `lom eval-judge`: hand-labelled opportunity cards (evals/judge_cases.yaml), each sent
through the production review call once per candidate judge model under one spending cap, then a JSON and Markdown
report. `lom eval` cannot compare judges: in a scan, a judge's mistakes show only through which cards survive."""
import asyncio
import json
import re
import statistics
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..agent.cards import MAX_EVIDENCE_ITEMS, OpportunityCardDraft
from ..agent.llm import build_chat_model
from ..agent.subagents import _review
from ..agent.telemetry import MemorySink, TelemetrySink, TurnContext
from ..config import MODEL_PREFIX, Settings
from ..paths import PROJECT_ROOT
from ..redact import redact
from .runner import Budget

JUDGE_CASES_FILE = PROJECT_ROOT / "evals" / "judge_cases.yaml"
# The Budget kind for one card review; runner.DEFAULT_RESERVE holds its reserve.
JUDGE_KIND = "judge"
MAX_ERROR_CHARS = 300
# How much of a wrong verdict's notes the report quotes.
NOTES_CHARS = 200
# An evidence line starts with the citation the card gives for it, as subagents._evidence_texts writes it.
EVIDENCE_LEAD = re.compile(r"^\[(F\d+|D\d+#\d+)\]")
Expected = Literal["supported", "unsupported"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeCase(StrictModel):
    """One card with its known answer: supported, or unsupported because of one planted error named in `flaw`."""
    id: str = Field(pattern=r"^[a-z0-9_]{1,40}$")
    expected: Expected
    tests: str = Field(min_length=1, max_length=300)
    flaw: str | None = Field(default=None, max_length=300)
    card: OpportunityCardDraft
    evidence: list[str] = Field(min_length=1, max_length=MAX_EVIDENCE_ITEMS)

    @model_validator(mode="after")
    def _consistent(self):
        if (self.expected == "unsupported") != bool(self.flaw):
            raise ValueError("an unsupported card names its planted error in flaw; a supported card has no flaw")
        shown = sorted(m.group(1) if (m := EVIDENCE_LEAD.match(line)) else line[:20] for line in self.evidence)
        cited = sorted(f"D{e.id}#{e.chunk_index or 0}" if e.kind == "document" else f"F{e.id}"
                       for e in self.card.evidence)
        if shown != cited:
            raise ValueError(f"evidence lines {shown} must be the card's own evidence {cited}, one line each")
        return self


class JudgeCases(StrictModel):
    """The Nan Fung context every review is given, and the labelled cards."""
    context: str = Field(min_length=1)
    cases: list[JudgeCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [c.id for c in self.cases]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"duplicate case ids: {', '.join(repeated)}")
        return self


def load_judge_cases(path: Path) -> JudgeCases:
    """The labelled cards in a YAML file, with every problem reported together."""
    try:
        return JudgeCases.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as e:
        raise ValueError(f"{path}: {e}") from e


@dataclass
class Verdict:
    """One review: the judge's verdict, or the error that stopped it, with its cost and time."""
    verdict: Expected | None
    notes: str | None
    cost_usd: Decimal
    latency_s: float
    error: str | None = None


@dataclass
class _ReviewDeps:
    """The two parts of the agent's dependencies that subagents._review reads."""
    models: dict
    sink: TelemetrySink


async def review_with_model(judge: Runnable, settings: Settings, case: JudgeCase, context: str) -> Verdict:
    """One card through the production review call (subagents._review: the same prompt, structured output and
    telemetry as in a scan), with this judge model."""
    sink = MemorySink()
    ctx = TurnContext(thread_id=f"judge-eval-{case.id}", turn_id=uuid.uuid4().hex, mode="eval")
    try:
        review = await _review(case.card, case.evidence, context, settings=settings,
                               deps=_ReviewDeps(models={"judge": judge}, sink=sink), ctx=ctx)
        error = None
    except Exception as e:  # noqa: BLE001  a failed review is this judge's result, kept with its reason
        review, error = None, redact(f"{type(e).__name__}: {e}", settings)[:MAX_ERROR_CHARS]
    call = sink.model_calls[-1] if sink.model_calls else None
    return Verdict(verdict=review.verdict if review else None, notes=review.notes if review else None,
                   cost_usd=call.cost_usd if call is not None and call.cost_usd is not None else Decimal(0),
                   latency_s=call.latency_ms / 1000 if call is not None else 0.0, error=error)


async def default_review_card(settings: Settings, model: str, case: JudgeCase, context: str) -> Verdict:
    """The candidate model in the judge role, built as production builds the judge (reasoning off, no fallback)."""
    judge = build_chat_model(settings, "judge", with_fallback=False, model_override=model)
    return await review_with_model(judge, settings, case, context)


ReviewCard = Callable[[Settings, str, JudgeCase, str], Awaitable[Verdict]]


@dataclass
class JudgeRun:
    repeat: int
    model: str
    case: JudgeCase
    verdict: Expected | None
    notes: str | None
    cost_usd: Decimal
    latency_s: float
    error: str | None

    @property
    def correct(self) -> bool:
        return self.verdict == self.case.expected


@dataclass
class JudgeResult:
    runs: list[JudgeRun]
    stopped_by_budget: bool
    spent: Decimal
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def judge_schedule(models: list[str], cases: list[JudgeCase], repeats: int) -> list[tuple[int, str, JudgeCase]]:
    """Card by card, every judge in turn, so a budget stop leaves all the judges with the same cards."""
    return [(repeat, model, case) for repeat in range(1, repeats + 1) for case in cases for model in models]


async def run_judge_matrix(settings: Settings, models: list[str], cases: JudgeCases, repeats: int, budget: Budget,
                           review_card: ReviewCard | None = None, concurrency: int = 3) -> JudgeResult:
    """Every card for every judge model, up to `concurrency` reviews at a time, until the schedule ends or the next
    review would not fit the budget."""
    review = review_card or default_review_card
    finished: dict[int, JudgeRun] = {}
    slots = asyncio.Semaphore(concurrency)
    tasks, stopped = [], False

    async def one(index: int, repeat: int, model: str, case: JudgeCase, held: Decimal) -> None:
        try:
            verdict = await review(settings, model, case, cases.context)
        except Exception as e:  # noqa: BLE001  one broken review must not end the comparison; it is recorded
            verdict = Verdict(verdict=None, notes=None, cost_usd=Decimal(0), latency_s=0.0,
                              error=redact(f"{type(e).__name__}: {e}", settings)[:MAX_ERROR_CHARS])
        finally:
            slots.release()
        budget.record(JUDGE_KIND, verdict.cost_usd, held)
        finished[index] = JudgeRun(repeat, model, case, verdict.verdict, verdict.notes, verdict.cost_usd,
                                   verdict.latency_s, verdict.error)

    for index, (repeat, model, case) in enumerate(judge_schedule(models, cases.cases, repeats)):
        await slots.acquire()
        if not budget.can_start(JUDGE_KIND):
            slots.release()
            stopped = True
            break
        tasks.append(asyncio.create_task(one(index, repeat, model, case, budget.begin(JUDGE_KIND))))
    await asyncio.gather(*tasks)
    return JudgeResult(runs=[finished[i] for i in sorted(finished)], stopped_by_budget=stopped, spent=budget.spent)


@dataclass
class JudgeStats:
    runs: int
    correct: int
    # A card with a planted error judged supported: in a scan it would reach the analyst.
    false_accepts: int
    unsupported_runs: int
    # A sound card judged unsupported: in a scan it would be thrown away.
    false_rejects: int
    supported_runs: int
    errors: int
    # Cards given the same verdict in every repeat, out of the cards reviewed more than once.
    consistent: int
    repeated_cards: int
    mean_cost_usd: Decimal
    p50_latency_s: float


def judge_stats(result: JudgeResult) -> dict[str, JudgeStats]:
    """Each judge's counts, in the order the judges were given."""
    stats = {}
    for model in dict.fromkeys(r.model for r in result.runs):
        runs = [r for r in result.runs if r.model == model]
        by_card: dict[str, list] = {}
        for r in runs:
            by_card.setdefault(r.case.id, []).append(r.verdict)
        repeated = [verdicts for verdicts in by_card.values() if len(verdicts) > 1]
        answered = [r.latency_s for r in runs if r.verdict is not None]
        stats[model] = JudgeStats(
            runs=len(runs), correct=sum(r.correct for r in runs),
            false_accepts=sum(r.case.expected == "unsupported" and r.verdict == "supported" for r in runs),
            unsupported_runs=sum(r.case.expected == "unsupported" for r in runs),
            false_rejects=sum(r.case.expected == "supported" and r.verdict == "unsupported" for r in runs),
            supported_runs=sum(r.case.expected == "supported" for r in runs),
            errors=sum(r.verdict is None for r in runs),
            consistent=sum(len(set(v)) == 1 and None not in v for v in repeated), repeated_cards=len(repeated),
            mean_cost_usd=sum((r.cost_usd for r in runs), Decimal(0)) / len(runs),
            p50_latency_s=statistics.median(answered) if answered else 0.0)
    return stats


def _mark(run: JudgeRun) -> str:
    return "⚠" if run.verdict is None else "✓" if run.correct else "✗"


def write_judge_report(result: JudgeResult, out_dir: Path) -> tuple[Path, Path]:
    """evals/results/judge-<UTC timestamp>.json with every review, and a Markdown summary beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = out_dir / f"judge-{stamp}.json", out_dir / f"judge-{stamp}.md"
    runs = [{"repeat": r.repeat, "model": r.model, "case": r.case.id, "expected": r.case.expected,
             "verdict": r.verdict, "correct": r.correct, "notes": r.notes, "cost_usd": str(r.cost_usd),
             "latency_s": round(r.latency_s, 1), "error": r.error} for r in result.runs]
    json_path.write_text(json.dumps({"started_at": result.started_at, "spent_usd": str(result.spent),
                                     "stopped_by_budget": result.stopped_by_budget, "runs": runs},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    stats = judge_stats(result)
    rows = [f"| {model.removeprefix(MODEL_PREFIX)} | {s.correct}/{s.runs} | {s.false_accepts}/{s.unsupported_runs} | "
            f"{s.false_rejects}/{s.supported_runs} | {s.errors} | "
            f"{f'{s.consistent}/{s.repeated_cards}' if s.repeated_cards else 'n/a'} | ${s.mean_cost_usd:.4f} | "
            f"{s.p50_latency_s:.1f} s |" for model, s in stats.items()]
    models = list(stats)
    cases = list(dict.fromkeys(r.case.id for r in result.runs))
    expected = {r.case.id: r.case.expected for r in result.runs}
    marks = {(r.case.id, r.model): "" for r in result.runs}
    for r in result.runs:
        marks[(r.case.id, r.model)] += _mark(r)
    grid = ["| Card | Expected | " + " | ".join(m.removeprefix(MODEL_PREFIX) for m in models) + " |",
            "|---|---|" + "---|" * len(models),
            *(f"| {c} | {expected[c]} | " + " | ".join(marks.get((c, m), "–") for m in models) + " |" for c in cases)]
    misses = [f"- {r.model.removeprefix(MODEL_PREFIX)} · {r.case.id} · repeat {r.repeat}: "
              + (f"error ({r.error})" if r.verdict is None else
                 f"said {r.verdict}" + (f" (planted: {r.case.flaw})" if r.case.flaw else "")
                 + f"; notes: {(r.notes or '')[:NOTES_CHARS]}")
              for r in result.runs if not r.correct] or ["- none"]
    md = ["# Judge comparison", "",
          f"Started {result.started_at}. Total spend: ${result.spent:.2f}"
          + (" (stopped by the budget cap)." if result.stopped_by_budget else "."), "",
          ("Each card was written by hand with a known answer: supported, or unsupported because of one planted error "
           "(evals/judge_cases.yaml). A false accept lets a card with a wrong fact reach the analyst; a false reject "
           "throws a sound card away. Consistency counts cards given the same verdict in every repeat. ⚠ marks a "
           "review that returned no verdict."), "",
          "| Judge | Correct | False accepts | False rejects | Errors | Consistency | Mean cost per card | p50 latency |",
          "|---|---|---|---|---|---|---|---|", *rows, "", "## Verdicts", "", *grid, "", "## Wrong verdicts and errors", "",
          *misses, ""]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path

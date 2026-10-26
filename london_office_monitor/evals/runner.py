"""The model comparison behind `lom eval`: every case for every candidate model in the case's role, under one spending
cap, against its own database (lom_eval) and data server (port 8766), then a JSON and Markdown report."""
import asyncio
import json
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from langchain_core.runnables import Runnable

from ..agent.graph import TurnResult, real_deps, run_turn
from ..agent.llm import Role, build_chat_model
from ..agent.telemetry import turn_summary
from ..config import MODEL_PREFIX, Settings
from ..db.connect import DbRole, connect
from ..paths import PROJECT_ROOT
from ..redact import redact
from .cases import EvalCase
from .graders import RunRecord, grade, passed

CASES_FILE = PROJECT_ROOT / "evals" / "cases.yaml"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
EVAL_DB = "lom_eval"
EVAL_MCP_PORT = 8766
ROLES: tuple[Role, ...] = ("orchestrator", "investigator", "judge")
# Held back from the cap before a run starts: the larger of this and the dearest run of that kind so far.
DEFAULT_RESERVE = {"chat": Decimal("0.25"), "scan": Decimal("0.75"), "judge": Decimal("0.05")}
MAX_ERROR_CHARS = 300

RunCase = Callable[[Settings, EvalCase, str, int], Awaitable[tuple[RunRecord, dict[str, bool]]]]


class Budget:
    """A hard spending cap: a run starts only if what is spent, what running cases hold back, and this run's
    reserve all fit under the cap."""

    def __init__(self, cap_usd: Decimal):
        self.cap = cap_usd
        self.spent = Decimal(0)
        self.in_flight = Decimal(0)
        self._dearest: dict[str, Decimal] = {}

    def reserve(self, kind: str) -> Decimal:
        return max(DEFAULT_RESERVE[kind], self._dearest.get(kind, Decimal(0)))

    def can_start(self, kind: str) -> bool:
        return self.spent + self.in_flight + self.reserve(kind) <= self.cap

    def begin(self, kind: str) -> Decimal:
        held = self.reserve(kind)
        self.in_flight += held
        return held

    def record(self, kind: str, cost: Decimal, held: Decimal = Decimal(0)) -> None:
        self.in_flight -= held
        self.spent += cost
        self._dearest[kind] = max(self._dearest.get(kind, Decimal(0)), cost)


@dataclass
class MatrixRun:
    repeat: int
    model: str
    case: EvalCase
    record: RunRecord
    checks: dict[str, bool]


@dataclass
class MatrixResult:
    runs: list[MatrixRun]
    stopped_by_budget: bool
    spent: Decimal
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def schedule(models: list[str], cases: list[EvalCase], repeats: int) -> list[tuple[int, str, EvalCase]]:
    """Round-robin: every model runs every case once before any second repeat, so a budget stop never drops a
    model."""
    return [(repeat, model, case) for repeat in range(1, repeats + 1) for model in models for case in cases]


def model_slug(model: str) -> str:
    return model.removeprefix(MODEL_PREFIX).replace("/", "_").replace(".", "-")


def build_case_models(settings: Settings, *, role: Role, model: str) -> dict[Role, Runnable]:
    """The three role models with fallbacks off; only the case's role uses the candidate model."""
    return {r: build_chat_model(settings, r, with_fallback=False, model_override=model if r == role else None)
            for r in ROLES}


async def default_run_case(settings: Settings, case: EvalCase, model: str, repeat: int) -> tuple[RunRecord,
                                                                                                    dict[str, bool]]:
    """One case on a fresh thread with the candidate model in the case's role, then its grades."""
    thread_id = f"eval-{case.id}-{model_slug(model)}-{repeat}"
    started = time.monotonic()
    async with real_deps(settings, models=build_case_models(settings, role=case.role, model=model)) as deps:
        result: TurnResult = await run_turn(settings, thread_id, case.prompt, mode="eval", deps=deps)
    latency = time.monotonic() - started
    with connect(settings, DbRole.APP) as conn:
        summary = turn_summary(conn, result.turn_id) if result.turn_id else None
        run = RunRecord(case_id=case.id, thread_id=thread_id, answer=result.answer, approval=result.approval,
                        tool_calls=summary.tool_calls if summary else [], loaded_skills=result.loaded_skills,
                        cost_usd=summary.cost_usd if summary else Decimal(0), latency_s=latency, error=result.error)
        return run, grade(case, run, conn)


async def run_matrix(settings: Settings, models: list[str], cases: list[EvalCase], repeats: int, budget: Budget,
                     run_case: RunCase = default_run_case, concurrency: int = 1) -> MatrixResult:
    """Run the schedule, up to `concurrency` cases at a time, until it ends or the next run would not fit the
    budget. A case that crashes is recorded as a failed run with no cost."""
    order = schedule(models, cases, repeats)
    finished: dict[int, MatrixRun] = {}
    slots = asyncio.Semaphore(concurrency)
    tasks, stopped = [], False

    async def one(index: int, repeat: int, model: str, case: EvalCase, held: Decimal) -> None:
        try:
            run, checks = await run_case(settings, case, model, repeat)
        except Exception as e:  # noqa: BLE001  one broken run must not end the comparison; it is recorded as failed
            run = RunRecord(case_id=case.id, thread_id="", answer=None, approval=None, tool_calls=[],
                            loaded_skills=[], cost_usd=Decimal(0), latency_s=0.0,
                            error=redact(f"{type(e).__name__}: {e}", settings)[:MAX_ERROR_CHARS])
            checks = {"no_error": False}
        finally:
            slots.release()
        budget.record(case.kind, run.cost_usd, held)
        finished[index] = MatrixRun(repeat, model, case, run, checks)

    for index, (repeat, model, case) in enumerate(order):
        await slots.acquire()
        if not budget.can_start(case.kind):
            slots.release()
            stopped = True
            break
        tasks.append(asyncio.create_task(one(index, repeat, model, case, budget.begin(case.kind))))
    await asyncio.gather(*tasks)
    return MatrixResult(runs=[finished[i] for i in sorted(finished)], stopped_by_budget=stopped, spent=budget.spent)


# Checks on which playbook a turn loaded: the playbook is the turn's intent label, so these measure routing.
ROUTING_CHECKS = ("skill:", "skill_not:", "no_skill")


def routed(checks: dict[str, bool]) -> bool | None:
    """Whether every routing check passed; None when the case has none."""
    routing = [ok for name, ok in checks.items() if name.startswith(ROUTING_CHECKS)]
    return all(routing) if routing else None


def _role_rows(result: MatrixResult) -> list[str]:
    rows = []
    groups: dict[tuple[str, str], list[MatrixRun]] = {}
    for run in result.runs:
        groups.setdefault((run.case.role, run.model), []).append(run)
    for (role, model), runs in sorted(groups.items()):
        ok = sum(passed(r.checks) for r in runs)
        by_case: dict[str, list[bool]] = {}
        for r in runs:
            by_case.setdefault(r.case.id, []).append(passed(r.checks))
        repeated = [v for v in by_case.values() if len(v) > 1]
        consistency = f"{sum(all(v) for v in repeated)}/{len(repeated)}" if repeated else "n/a"
        routing = [ok for ok in (routed(r.checks) for r in runs) if ok is not None]
        routing_rate = f"{sum(routing)}/{len(routing)}" if routing else "n/a"
        mean_cost = sum((r.record.cost_usd for r in runs), Decimal(0)) / len(runs)
        p50 = statistics.median(r.record.latency_s for r in runs)
        tools = statistics.mean(len(r.record.tool_calls) for r in runs)
        rows.append(f"| {role} | {model.removeprefix(MODEL_PREFIX)} | {ok}/{len(runs)} | {routing_rate} | "
                    f"{consistency} | ${mean_cost:.4f} | {p50:.0f} s | {tools:.1f} |")
    return rows


def _grid(result: MatrixResult) -> list[str]:
    models = list(dict.fromkeys(r.model for r in result.runs))
    cases = list(dict.fromkeys(r.case.id for r in result.runs))
    marks = {(r.case.id, r.model): "" for r in result.runs}
    for r in result.runs:
        marks[(r.case.id, r.model)] += "✓" if passed(r.checks) else "✗"
    head = "| Case | " + " | ".join(m.removeprefix(MODEL_PREFIX) for m in models) + " |"
    rule = "|---|" + "---|" * len(models)
    return [head, rule, *(f"| {c} | " + " | ".join(marks.get((c, m), "–") for m in models) + " |" for c in cases)]


def _failures(result: MatrixResult) -> list[str]:
    lines = []
    for r in result.runs:
        failed = [name for name, ok in r.checks.items() if not ok]
        if failed:
            detail = f" ({r.record.error})" if r.record.error else ""
            lines.append(f"- {r.model.removeprefix(MODEL_PREFIX)} · {r.case.id} · repeat {r.repeat}: "
                         f"{', '.join(failed)}{detail}")
    return lines or ["- none"]


def write_report(result: MatrixResult, out_dir: Path) -> tuple[Path, Path]:
    """evals/results/<UTC timestamp>.json with every run, and a Markdown summary beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = out_dir / f"{stamp}.json", out_dir / f"{stamp}.md"
    runs = [{"repeat": r.repeat, "model": r.model, "case": r.case.id, "role": r.case.role, "passed": passed(r.checks),
             "checks": r.checks, "cost_usd": str(r.record.cost_usd), "latency_s": round(r.record.latency_s, 1),
             "tool_calls": [t.tool for t in r.record.tool_calls], "loaded_skills": r.record.loaded_skills,
             "error": r.record.error, "answer": r.record.answer,
             "proposed": [c.title for c in r.record.approval.cards] if r.record.approval else []}
            for r in result.runs]
    json_path.write_text(json.dumps({"started_at": result.started_at, "spent_usd": str(result.spent),
                                     "stopped_by_budget": result.stopped_by_budget, "runs": runs},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# Model comparison", "",
          f"Started {result.started_at}. Total spend: ${result.spent:.2f}"
          + (" (stopped by the budget cap)." if result.stopped_by_budget else "."), "",
          ("Pass rate counts runs in which every check passed. Routing counts runs that loaded the expected playbook "
           "and no unwanted one (the playbook is the turn's intent label). Consistency counts cases that passed in "
           "every repeat."),
          "", "| Role | Model | Pass rate | Routing | Consistency | Mean cost per case | p50 latency | Mean tool calls |",
          "|---|---|---|---|---|---|---|---|", *_role_rows(result), "", "## Pass grid", "", *_grid(result), "",
          "## Failed checks", "", *_failures(result), ""]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path

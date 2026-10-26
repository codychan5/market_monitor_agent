import json
from decimal import Decimal

import pytest

from london_office_monitor.agent.graph import BRIEF_PROMPT, SCAN_PROMPT
from london_office_monitor.evals.cases import ChatChecks, EvalCase, ScanChecks, load_cases
from london_office_monitor.evals.graders import RunRecord
from london_office_monitor.evals.runner import (CASES_FILE, Budget, MatrixResult, MatrixRun, build_case_models,
                                                run_matrix, schedule, write_report)

C1 = EvalCase(id="c1", kind="chat", role="orchestrator", prompt="q1", checks=ChatChecks())
C2 = EvalCase(id="c2", kind="chat", role="orchestrator", prompt="q2", checks=ChatChecks())
S1 = EvalCase(id="s1", kind="scan", role="investigator", prompt="scan", checks=ScanChecks())


def record(case: EvalCase, cost: str = "0.1", error: str | None = None) -> RunRecord:
    return RunRecord(case_id=case.id, thread_id=f"eval-{case.id}", answer="ok", approval=None, tool_calls=[],
                     loaded_skills=[], cost_usd=Decimal(cost), latency_s=2.0, error=error)


def fake_run_case(cost: str):
    async def run(settings, case, model, repeat):
        return record(case, cost), {"no_error": True}
    return run


def test_cases_file_has_thirteen_valid_cases():
    cases = load_cases(CASES_FILE)
    assert len(cases) == 13 and len({c.id for c in cases}) == 13 and {c.kind for c in cases} == {"chat", "scan"}
    assert {"c7_precedent", "c8_trend", "c9_off_topic"} <= {c.id for c in cases}
    assert {c.role for c in cases if c.kind == "scan"} == {"investigator", "orchestrator"}
    prompts = {c.id: c.prompt for c in cases}
    assert prompts["c6_weekly_brief"] == BRIEF_PROMPT and prompts["s1_scan"] == SCAN_PROMPT


def test_every_scan_also_runs_with_the_orchestrator_under_test():
    # The orchestrator picks a scan's leads, ranks its cards and writes its report, so each scan has a twin that
    # differs only in the role whose model is swapped.
    by_id = {c.id: c for c in load_cases(CASES_FILE)}
    for investigator_case, orchestrator_case in (("s1_scan", "s1_scan_orchestrator"),
                                                 ("s2_injection", "s2_injection_orchestrator")):
        twin, original = by_id[orchestrator_case], by_id[investigator_case]
        assert (original.role, twin.role) == ("investigator", "orchestrator")
        assert (twin.kind, twin.prompt, twin.checks) == (original.kind, original.prompt, original.checks)


def test_cases_refuse_unknown_checks(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("- {id: x1, kind: chat, role: orchestrator, prompt: q, checks: {colour: blue}}\n")
    with pytest.raises(ValueError, match="colour"):
        load_cases(path)


def test_schedule_is_round_robin():
    assert schedule(["a", "b"], [C1, C2], 2)[:4] == [(1, "a", C1), (1, "a", C2), (1, "b", C1), (1, "b", C2)]


def test_budget_stops_before_cap():
    b = Budget(Decimal("5"))
    for _ in range(4):
        b.record("scan", Decimal("0.9"))
    assert b.can_start("scan") is True     # 3.6 + 0.9 <= 5
    b.record("scan", Decimal("0.9"))
    assert b.can_start("scan") is False    # 4.5 + 0.9 > 5


def test_budget_counts_runs_in_flight():
    b = Budget(Decimal("0.9"))
    held = b.begin("chat")
    assert held == Decimal("0.25") and b.can_start("scan") is False   # 0.25 held + 0.75 > 0.9
    b.record("chat", Decimal("0.05"), held)
    assert b.spent == Decimal("0.05") and b.can_start("scan") is True  # 0.05 spent + 0.75 <= 0.9


async def test_run_matrix_stops_by_budget_without_dropping_models(settings):
    result = await run_matrix(settings, ["a", "b", "c"], [C1, C2], 2, Budget(Decimal("1.6")),
                              run_case=fake_run_case("0.2"))
    first_pass = {(r.repeat, r.model) for r in result.runs if r.repeat == 1}
    assert result.stopped_by_budget and first_pass == {(1, "a"), (1, "b"), (1, "c")}


async def test_run_matrix_runs_concurrently_in_schedule_order(settings):
    result = await run_matrix(settings, ["a", "b", "c"], [C1, C2], 1, Budget(Decimal("5")),
                              run_case=fake_run_case("0.1"), concurrency=3)
    assert not result.stopped_by_budget and result.spent == Decimal("0.6")
    assert [(r.model, r.case.id) for r in result.runs] == [(m, c.id) for _, m, c in schedule(["a", "b", "c"],
                                                                                          [C1, C2], 1)]


async def test_a_crashing_case_is_recorded_as_a_failed_run(settings):
    async def crash(settings, case, model, repeat):
        raise RuntimeError("provider down")

    result = await run_matrix(settings, ["a"], [C1], 1, Budget(Decimal("1")), run_case=crash)
    assert result.runs[0].checks == {"no_error": False} and result.runs[0].record.error.startswith("RuntimeError")


def test_eval_disables_fallbacks_and_swaps_only_the_case_role(monkeypatch, settings):
    calls = []
    monkeypatch.setattr("london_office_monitor.evals.runner.build_chat_model",
                        lambda s, role, **kw: calls.append((role, kw)) or role)
    build_case_models(settings, role="investigator", model="openrouter:moonshotai/kimi-k3")
    assert all(kw["with_fallback"] is False for _, kw in calls)
    assert {role: kw["model_override"] for role, kw in calls} == {
        "orchestrator": None, "investigator": "openrouter:moonshotai/kimi-k3", "judge": None}


def test_write_report_summarises_each_role_and_model(tmp_path):
    runs = [MatrixRun(1, "openrouter:z-ai/glm-5.3", C1, record(C1, "0.02"), {"no_error": True, "skill:x": True}),
            MatrixRun(1, "openrouter:moonshotai/kimi-k3", C1, record(C1, "0.05"), {"no_error": True,
                                                                                   "skill:x": False}),
            MatrixRun(1, "openrouter:z-ai/glm-5.3", S1, record(S1, "0.08"), {"no_error": True})]
    json_path, md_path = write_report(MatrixResult(runs=runs, stopped_by_budget=False, spent=Decimal("0.15")),
                                      tmp_path)
    md = md_path.read_text()
    assert "| orchestrator | z-ai/glm-5.3 | 1/1 |" in md and "| orchestrator | moonshotai/kimi-k3 | 0/1 |" in md
    assert "| investigator | z-ai/glm-5.3 | 1/1 |" in md and "Total spend: $0.15" in md
    assert "kimi-k3 · c1 · repeat 1: skill:x" in md
    assert len(json.loads(json_path.read_text())["runs"]) == 3


def test_write_report_shows_routing_per_model(tmp_path):
    # The playbook a turn loads is its intent label; the routing column is how often the model chose the right one.
    runs = [MatrixRun(1, "openrouter:z-ai/glm-5.3", C1, record(C1), {"no_error": True, "skill:market-questions": True}),
            MatrixRun(1, "openrouter:z-ai/glm-5.3", C2, record(C2), {"no_error": True, "skill:market-brief": True,
                                                                     "skill_not:opportunity-scan": False}),
            MatrixRun(1, "openrouter:z-ai/glm-5.3", C2, record(C2), {"no_error": True, "no_skill": True}),
            MatrixRun(1, "openrouter:z-ai/glm-5.3", S1, record(S1), {"no_error": True})]
    _, md_path = write_report(MatrixResult(runs=runs, stopped_by_budget=False, spent=Decimal("0.1")), tmp_path)
    md = md_path.read_text()
    assert "| Role | Model | Pass rate | Routing |" in md
    assert "| orchestrator | z-ai/glm-5.3 | 2/3 | 2/3 |" in md and "| investigator | z-ai/glm-5.3 | 1/1 | n/a |" in md

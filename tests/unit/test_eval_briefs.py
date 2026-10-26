import json

from typer.testing import CliRunner

from london_office_monitor.cli import app
from london_office_monitor.evals import runner as runner_module
from london_office_monitor.evals.briefs import (BRIEF_CASE, BRIEF_SECTIONS, brief_measures, collect_briefs,
                                                write_brief_report)
from london_office_monitor.evals.cases import load_cases
from london_office_monitor.evals.runner import CASES_FILE

ASSETS = ["Regent Quarter", "99 City Road", "108 Cannon Street", "138 Cheapside", "16 Old Bailey"]
BRIEF = """# London office brief

## What changed
- City Core prime rent rose to £100.00 [F68][F57]; West End prime rent rose to £195.00 [F65][F56][F57].

## News themes
- Business rates: a £1.1bn rise [D54#0]; the City was warned [D5#0][D5#1].

## Rates and economy
- Bank Rate held at 3.75%.

## What it means for our portfolio
- 108 Cannon Street and 16 Old Bailey gain from higher rents [F57]; 99 City Road from a thinner pipeline [D188#0].

## Coming up
- The rates rise takes effect in April [D54#0].
"""


def results_file(path, started_at, runs):
    path.write_text(json.dumps({"started_at": started_at, "spent_usd": "0.1", "stopped_by_budget": False,
                                "runs": runs}), encoding="utf-8")


def brief_run(model, answer, passed=True):
    return {"repeat": 1, "model": model, "case": BRIEF_CASE, "role": "orchestrator", "passed": passed, "checks": {},
            "cost_usd": "0.01", "latency_s": 20.0, "tool_calls": [], "loaded_skills": [], "error": None,
            "answer": answer, "proposed": []}


def test_the_sections_measured_are_the_ones_the_eval_requires():
    case = next(c for c in load_cases(CASES_FILE) if c.id == BRIEF_CASE)
    assert tuple(case.checks.required_headings) == BRIEF_SECTIONS


def test_a_brief_is_measured_by_what_each_section_covers():
    m = brief_measures(BRIEF, ASSETS)
    assert (m.sections, m.changes_cited, m.stories_cited, m.rates_covered, m.assets_named) == (5, 4, 2, 1, 3)
    assert m.words == len(BRIEF.split()) and not m.unverified


def test_a_missing_section_counts_nothing():
    m = brief_measures(BRIEF.replace("## News themes", "## Headlines"), ASSETS)
    assert m.sections == 4 and m.stories_cited == 0


def test_every_weekly_brief_in_the_reports_is_collected(tmp_path):
    results_file(tmp_path / "20261024T114708Z.json", "2026-10-24T11:47:08+00:00",
                 [brief_run("openrouter:z-ai/glm-5.3", BRIEF), {**brief_run("openrouter:z-ai/glm-5.3", BRIEF),
                                                                 "case": "c1_city_prime_rent"}])
    results_file(tmp_path / "20261025T143050Z.json", "2026-10-25T14:30:50+00:00",
                 [brief_run("openrouter:deepseek/deepseek-v4.1-flash", BRIEF, passed=False),
                  brief_run("openrouter:moonshotai/kimi-k3", None)])
    (tmp_path / "judge-20261025T175004Z.json").write_text("{}", encoding="utf-8")   # other reports are skipped
    briefs = collect_briefs(tmp_path)
    assert [(b.model, b.started_at[:10], b.passed) for b in briefs] == [
        ("z-ai/glm-5.3", "2026-10-24", True), ("deepseek/deepseek-v4.1-flash", "2026-10-25", False)]


def test_the_report_summarises_each_model(tmp_path):
    results_file(tmp_path / "20261025T143050Z.json", "2026-10-25T14:30:50+00:00",
                 [brief_run("openrouter:z-ai/glm-5.3", BRIEF), brief_run("openrouter:z-ai/glm-5.3", BRIEF)])
    md_path = write_brief_report(collect_briefs(tmp_path), ASSETS, tmp_path / "out")
    md = md_path.read_text(encoding="utf-8")
    assert md_path.name.startswith("briefs-") and "| z-ai/glm-5.3 | 2 | 2/2 | 4.0 | 2.0 | 1.0 | 3.0 |" in md


def test_eval_briefs_writes_its_report(monkeypatch, tmp_path):
    results_file(tmp_path / "20261025T143050Z.json", "2026-10-25T14:30:50+00:00",
                 [brief_run("openrouter:z-ai/glm-5.3", BRIEF)])
    monkeypatch.setattr(runner_module, "RESULTS_DIR", tmp_path)
    result = CliRunner().invoke(app, ["eval-briefs"])
    assert result.exit_code == 0, result.output
    assert "Weekly brief comparison" in result.output and list(tmp_path.glob("briefs-*.md"))

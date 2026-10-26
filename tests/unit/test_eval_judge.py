import functools
from collections import Counter
from decimal import Decimal

import pytest
import yaml
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from typer.testing import CliRunner

from london_office_monitor.cli import app
from london_office_monitor.evals import judge as judge_module
from london_office_monitor.evals import runner as runner_module
from london_office_monitor.evals.judge import (JUDGE_CASES_FILE, Verdict, judge_stats, load_judge_cases,
                                               review_with_model, run_judge_matrix, write_judge_report)
from london_office_monitor.evals.runner import Budget

MODELS = ["openrouter:a/judge-one", "openrouter:b/judge-two"]


@functools.cache
def labelled():
    return load_judge_cases(JUDGE_CASES_FILE)


class FixedJudge(GenericFakeChatModel):
    """Answers every review with one verdict through the Review tool, as the judge's forced tool call does."""
    verdict: str = "supported"
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(str(messages[-1].content))
        message = AIMessage("", tool_calls=[{"name": "Review", "args": {"verdict": self.verdict, "notes": "checked"},
                                             "id": "r1", "type": "tool_call"}],
                            response_metadata={"token_usage": {"cost": 0.0004}, "model_name": "a/judge-one"})
        return ChatResult(generations=[ChatGeneration(message=message)])


class RefusingJudge(FixedJudge):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("the provider refused the request")


def fake_review(says: dict[str, str], cost: str = "0.001"):
    """Each model's verdict: "right" answers every card correctly, otherwise the same verdict for every card."""
    async def review(settings, model, case, context):
        verdict = case.expected if says[model] == "right" else says[model]
        return Verdict(verdict=verdict, notes="n", cost_usd=Decimal(cost), latency_s=1.0)
    return review


def write_cases(tmp_path, case: dict):
    path = tmp_path / "judge_cases.yaml"
    path.write_text(yaml.safe_dump({"context": "Nan Fung context", "cases": [case]}), encoding="utf-8")
    return path


def first_case(**changes) -> dict:
    return {**labelled().cases[0].model_dump(mode="json"), **changes}


def test_the_cards_are_balanced_and_each_flawed_card_names_its_error():
    cases = labelled().cases
    expected = Counter(c.expected for c in cases)
    assert len(cases) >= 12 and len({c.id for c in cases}) == len(cases)
    assert expected["supported"] >= 6 and expected["unsupported"] >= 6
    assert all(c.flaw for c in cases if c.expected == "unsupported")
    assert "99 City Road, 99 City Road, London EC1 (City Fringe" in labelled().context


def test_a_card_is_shown_with_exactly_the_evidence_it_cites(tmp_path):
    with pytest.raises(ValueError, match="evidence lines"):
        load_judge_cases(write_cases(tmp_path, first_case(evidence=["[D999#0] Some other passage: text"])))


def test_only_an_unsupported_card_names_a_flaw(tmp_path):
    with pytest.raises(ValueError, match="flaw"):
        load_judge_cases(write_cases(tmp_path, first_case(expected="unsupported", flaw=None)))
    with pytest.raises(ValueError, match="flaw"):
        load_judge_cases(write_cases(tmp_path, first_case(expected="supported", flaw="a planted error")))


async def test_a_review_sends_the_card_its_evidence_and_the_context(settings):
    case = labelled().cases[0]
    judge = FixedJudge(messages=iter([]), verdict="unsupported", seen=[])
    verdict = await review_with_model(judge, settings, case, labelled().context)
    assert (verdict.verdict, verdict.notes, verdict.cost_usd, verdict.error) == (
        "unsupported", "checked", Decimal("0.0004"), None)
    sent = judge.seen[-1]
    assert case.card.title in sent and case.evidence[0] in sent and labelled().context in sent


async def test_a_failed_review_is_an_error_not_a_verdict(settings):
    verdict = await review_with_model(RefusingJudge(messages=iter([]), seen=[]), settings, labelled().cases[0],
                                      labelled().context)
    assert verdict.verdict is None and verdict.error.startswith("RuntimeError") and "refused" in verdict.error


async def test_wrong_verdicts_are_counted_as_false_accepts_and_false_rejects(settings):
    cases = labelled()
    result = await run_judge_matrix(settings, MODELS, cases, 1, Budget(Decimal("5")),
                                    review_card=fake_review({MODELS[0]: "right", MODELS[1]: "supported"}))
    stats = judge_stats(result)
    unsupported = sum(c.expected == "unsupported" for c in cases.cases)
    right, credulous = stats[MODELS[0]], stats[MODELS[1]]
    assert (right.correct, right.false_accepts, right.false_rejects) == (len(cases.cases), 0, 0)
    assert (credulous.false_accepts, credulous.false_rejects) == (unsupported, 0)
    assert result.spent == Decimal("0.001") * 2 * len(cases.cases) and not result.stopped_by_budget


async def test_a_review_that_raises_is_recorded_as_an_error(settings):
    async def broken(settings, model, case, context):
        raise RuntimeError("no endpoint matches the data policy")
    result = await run_judge_matrix(settings, MODELS[:1], labelled(), 1, Budget(Decimal("5")), review_card=broken)
    stats = judge_stats(result)[MODELS[0]]
    assert stats.errors == len(labelled().cases) and stats.correct == 0
    assert "data policy" in result.runs[0].error


async def test_a_budget_stop_leaves_every_judge_with_the_same_cards(settings):
    # Each review holds back 0.05 while it runs; at 0.03 a review, the third one is the last that fits under 0.12.
    result = await run_judge_matrix(settings, MODELS, labelled(), 1, Budget(Decimal("0.12")),
                                    review_card=fake_review({MODELS[0]: "right", MODELS[1]: "right"}, cost="0.03"),
                                    concurrency=1)
    assert result.stopped_by_budget and len(result.runs) == 3
    per_model = Counter(r.model for r in result.runs)
    assert max(per_model.values()) - min(per_model.values()) <= 1


async def test_consistency_counts_cards_judged_alike_in_every_repeat(settings):
    calls = Counter()
    first = labelled().cases[0]

    async def flip_once(settings, model, case, context):
        calls[case.id] += 1
        wrong = "unsupported" if case.expected == "supported" else "supported"
        verdict = wrong if case.id == first.id and calls[case.id] == 2 else case.expected
        return Verdict(verdict=verdict, notes="n", cost_usd=Decimal("0"), latency_s=1.0)
    result = await run_judge_matrix(settings, MODELS[:1], labelled(), 2, Budget(Decimal("5")), review_card=flip_once)
    assert judge_stats(result)[MODELS[0]].consistent == len(labelled().cases) - 1


async def test_the_report_shows_each_judge_and_each_card(settings, tmp_path):
    cases = labelled().cases
    result = await run_judge_matrix(settings, MODELS, labelled(), 1, Budget(Decimal("5")),
                                    review_card=fake_review({MODELS[0]: "right", MODELS[1]: "supported"}))
    json_path, md_path = write_judge_report(result, tmp_path)
    md = md_path.read_text(encoding="utf-8")
    total, unsupported = len(cases), sum(c.expected == "unsupported" for c in cases)
    assert json_path.name.startswith("judge-") and md_path.name.startswith("judge-")
    assert f"| a/judge-one | {total}/{total} | 0/{unsupported} | 0/{total - unsupported} |" in md
    assert f"| b/judge-two | {total - unsupported}/{total} | {unsupported}/{unsupported} |" in md
    assert all(c.id in md for c in cases)


def test_eval_judge_refuses_a_model_not_named_for_openrouter():
    result = CliRunner().invoke(app, ["eval-judge", "--models", "qwen/qwen3.7-plus"])
    assert result.exit_code == 1 and "openrouter:<id>" in result.output


@pytest.mark.parametrize("budget", ["NaN", "0", "-1", "6", "abc"])
def test_eval_judge_refuses_a_budget_outside_the_cap(budget):
    result = CliRunner().invoke(app, ["eval-judge", "--models", MODELS[0], "--budget-usd", budget])
    assert result.exit_code == 1 and "--budget-usd must be" in result.output


def test_eval_judge_writes_its_report(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_module, "default_review_card", fake_review({MODELS[0]: "right"}))
    monkeypatch.setattr(runner_module, "RESULTS_DIR", tmp_path)
    result = CliRunner().invoke(app, ["eval-judge", "--models", MODELS[0], "--budget-usd", "1"])
    assert result.exit_code == 0, result.output
    assert "Judge comparison" in result.output and list(tmp_path.glob("judge-*.md"))

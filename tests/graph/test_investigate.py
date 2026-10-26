import json
import re
import threading
import time
import uuid
from dataclasses import replace

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from london_office_monitor.agent.approval import load_cards
from london_office_monitor.agent.graph import run_turn
from london_office_monitor.agent.subagents import SYNTHETIC_LABEL, SYNTHETIC_NOTE, run_investigations
from london_office_monitor.agent.telemetry import TurnContext
from london_office_monitor.facts import get_mandate, get_portfolio
from tests.conftest import CITY_ROAD_URL, KINGS_CROSS_URL, SYNTHETIC_URL, TOWER_SALE_URL, doc_id
from tests.fake_model import scripted, tool_call

CANDIDATE_DOC = re.compile(r"Candidate document: D(\d+)#0")


def card_args(target_doc: int, evidence_doc: int | None = None, title: str = "AI firm leases King's Cross offices"):
    return {"outcome": "card", "discard_reason": None, "card": {
        "title": title, "signal_type": "demand", "submarket": "King's Cross/Euston", "assets_affected": ["Regent Quarter"],
        "summary": "An AI company leased 60,000 sq ft near King's Cross station.",
        "why_it_matters": "Rent evidence for Regent Quarter.",
        "evidence": [{"kind": "document", "id": evidence_doc or target_doc, "chunk_index": 0,
                      "quote_or_value": "leased 60,000 sq ft"}],
        "suggested_next_step": "Brief the leasing team.", "confidence": "medium",
        "confidence_reason": "One report."}}


class RoutingInvestigator(GenericFakeChatModel):
    """A fake investigator that answers by the candidate document named in its prompt: a card, a discard, a card
    citing missing evidence, or a failure. It optionally sleeps, to let tests observe concurrency. It records each
    candidate's system prompt and the tools it was offered in `seen`."""
    behaviours: dict = {}
    delay: float = 0.0
    gauge: dict = {}
    seen: dict = {}

    def bind_tools(self, tools, **kwargs):
        self.seen.setdefault("tools", set()).update(t.name for t in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        lock = self.gauge.setdefault("lock", threading.Lock())
        with lock:
            self.gauge["now"] = self.gauge.get("now", 0) + 1
            self.gauge["max_seen"] = max(self.gauge.get("max_seen", 0), self.gauge["now"])
        try:
            time.sleep(self.delay)
            target = int(CANDIDATE_DOC.search(str(messages[0].content)).group(1))
            self.seen.setdefault("prompts", {})[target] = str(messages[0].content)
            behaviour = self.behaviours[target]
            if behaviour == "raise":
                raise RuntimeError("investigator crashed")
            args = {"card": card_args(target), "discard": {"outcome": "discard", "card": None,
                                                           "discard_reason": "Not about London offices."},
                    "bad_evidence": card_args(target, evidence_doc=999_999)}[behaviour]
            message = AIMessage("", tool_calls=[{"name": "InvestigationResult", "args": args, "id": f"r{target}",
                                                 "type": "tool_call"}])
            return ChatResult(generations=[ChatGeneration(message=message)])
        finally:
            with lock:
                self.gauge["now"] -= 1


class FakeJudge(GenericFakeChatModel):
    """Returns a fixed verdict and records what it was asked to check in `seen`."""
    verdict: str = "supported"
    seen: dict = {}

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen["input"] = str(messages[-1].content)
        message = AIMessage("", tool_calls=[{"name": "Review", "args": {"verdict": self.verdict, "notes": "checked"},
                                             "id": "j1", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class FailingJudge(FakeJudge):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("the judge's provider refused the request")


BUDGET_DISCARD = "Budget spent before corroboration."


class ToolHungryInvestigator(GenericFakeChatModel):
    """Calls search_documents, `parallel` times a step, while any tool besides the result is offered; returns a
    discard once the result tool is the only one left."""
    parallel: int = 1
    offered: list = []

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"offered": [t.name for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if set(self.offered) - {"InvestigationResult"}:
            calls = [{"name": "search_documents", "args": {"query": "office letting"}, "id": f"s{uuid.uuid4().hex[:8]}",
                      "type": "tool_call"} for _ in range(self.parallel)]
        else:
            calls = [{"name": "InvestigationResult", "id": "r1", "type": "tool_call",
                      "args": {"outcome": "discard", "card": None, "discard_reason": BUDGET_DISCARD}}]
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=calls))])


class TextOnlyInvestigator(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage("I could not decide."))])


def scan(deps, behaviours: dict, *, verdict: str = "supported", delay: float = 0.0, gauge: dict | None = None):
    investigator = RoutingInvestigator(messages=iter([]), behaviours=behaviours, delay=delay, gauge=gauge or {})
    return replace(deps, models={**deps.models, "investigator": investigator,
                                 "judge": FakeJudge(messages=iter([]), verdict=verdict)})


CTX = TurnContext(thread_id="scan-test", turn_id="u1", mode="scan")


def candidate(conn, url: str, signal: str = "demand") -> dict:
    return {"doc_id": doc_id(conn, url), "signal_type": signal, "hypothesis": "Worth a look."}


def status_of(conn, card_id: int) -> str:
    return conn.execute("SELECT status FROM opportunities WHERE id = %s", (card_id,)).fetchone()[0]


async def test_investigate_signals_mixed_outcomes(settings, agent_deps, seeded_docs):
    kc, tower, road = (doc_id(seeded_docs, u) for u in (KINGS_CROSS_URL, TOWER_SALE_URL, CITY_ROAD_URL))
    deps = scan(agent_deps, {kc: "card", tower: "discard", road: "raise"})
    out = await run_investigations([candidate(seeded_docs, u) for u in (KINGS_CROSS_URL, TOWER_SALE_URL, CITY_ROAD_URL)],
                                   settings=settings, deps=deps, ctx=CTX)
    assert [r["outcome"] for r in out["results"]] == ["card", "discard", "failed"]
    card = out["results"][0]
    assert status_of(seeded_docs, card["card_id"]) == "draft" and card["score"] > 0
    assert seeded_docs.execute("SELECT thread_id, embedding IS NOT NULL FROM opportunities WHERE id = %s",
                               (card["card_id"],)).fetchone() == ("scan-test", True)
    assert out["results"][2]["reason"].startswith("RuntimeError")


async def test_a_story_already_proposed_is_skipped_without_an_investigation(settings, agent_deps, seeded_docs):
    # Found on the first real-data scan (26 Oct 2026): nothing stopped a later scan from investigating and proposing
    # again a story an earlier scan had already put to the analyst, so "Run a new scan" queued duplicates.
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    first = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                     deps=scan(agent_deps, {kc: "card"}), ctx=CTX)
    card_id = first["results"][0]["card_id"]
    seeded_docs.execute("UPDATE opportunities SET status = 'pending' WHERE id = %s", (card_id,))
    gauge: dict = {}
    again = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                     deps=scan(agent_deps, {kc: "card"}, gauge=gauge), ctx=CTX)
    result = again["results"][0]
    assert result["outcome"] == "discard" and result["card_id"] is None
    assert f"already proposed as card {card_id} (pending)" in result["reason"]
    assert "max_seen" not in gauge  # no investigator ran, so no model call was spent on it


async def test_eval_cases_never_skip_a_lead_an_earlier_case_proposed(settings, agent_deps, seeded_docs):
    # Found by the eval session (26 Oct 2026): one `lom eval` run puts every scan case through the same database, so
    # the skip above made a later case discard leads an earlier case had carded. Eval cases stand alone.
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    first = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                     deps=scan(agent_deps, {kc: "card"}), ctx=CTX)
    seeded_docs.execute("UPDATE opportunities SET status = 'pending' WHERE id = %s", (first["results"][0]["card_id"],))
    again = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                     deps=scan(agent_deps, {kc: "card"}),
                                     ctx=TurnContext(thread_id="eval-test", turn_id="u2", mode="eval"))
    assert again["results"][0]["outcome"] == "card"


async def test_a_draft_that_never_reached_the_analyst_does_not_block_a_lead(settings, agent_deps, seeded_docs):
    # A draft the scan did not propose, or a card its reviewer discarded, was never put to the analyst.
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                             deps=scan(agent_deps, {kc: "card"}), ctx=CTX)
    again = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                     deps=scan(agent_deps, {kc: "card"}), ctx=CTX)
    assert again["results"][0]["outcome"] == "card"


async def test_card_keeps_the_inputs_its_score_was_computed_from(settings, agent_deps, seeded_docs):
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    out = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                   deps=scan(agent_deps, {kc: "card"}), ctx=CTX)
    card_id = out["results"][0]["card_id"]
    card = load_cards(seeded_docs, [card_id])[card_id][0]
    # The closest target, recomputed with numpy from the stored embeddings rather than with pgvector.
    vector = np.asarray(seeded_docs.execute("SELECT embedding::real[] FROM opportunities WHERE id = %s",
                                            (card_id,)).fetchone()[0])
    targets = [(name, "asset", np.asarray(v)) for name, v in seeded_docs.execute(
        "SELECT name, embedding::real[] FROM portfolio_assets")]
    targets += [(text, "mandate", np.asarray(v)) for text, v in seeded_docs.execute(
        "SELECT text, embedding::real[] FROM mandate_statements")]
    similarity, name, kind = max((float(vector @ v / (np.linalg.norm(vector) * np.linalg.norm(v))), n, k)
                                 for n, k, v in targets)
    inputs = card.score_inputs
    assert (inputs.closest_match, inputs.closest_match_kind) == (name, kind)
    assert inputs.mandate_similarity == pytest.approx(similarity, abs=0.0005)
    # The King's Cross item was published at 09:00 UTC on 22 Oct; the demo clock reads 08:00 UTC on 24 Oct.
    assert inputs.evidence_age_days == 1
    # Each part can be recomputed from what is stored: 40 points for a similarity of 1, and 20 for evidence under a
    # week old.
    assert card.score_components["mandate"] == round(inputs.mandate_similarity * 40)
    assert card.score_components["recency"] == 20


async def test_unsupported_card_is_discarded(settings, agent_deps, seeded_docs):
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    out = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                   deps=scan(agent_deps, {kc: "card"}, verdict="unsupported"), ctx=CTX)
    assert out["results"][0]["outcome"] == "discard" and out["results"][0]["reason"].startswith("reviewer")
    assert status_of(seeded_docs, out["results"][0]["card_id"]) == "discarded"


async def test_more_than_five_candidates_refused(settings, agent_deps, seeded_docs):
    six = [candidate(seeded_docs, KINGS_CROSS_URL)] * 6
    out = await run_investigations(six, settings=settings, deps=agent_deps, ctx=CTX)
    assert "at most 5" in out["error"]


async def test_concurrency_cap(settings, agent_deps, seeded_docs):
    urls = (KINGS_CROSS_URL, TOWER_SALE_URL, CITY_ROAD_URL)
    deps = scan(agent_deps, {doc_id(seeded_docs, u): "discard" for u in urls}, delay=0.2)
    await run_investigations([candidate(seeded_docs, u) for u in urls],
                             settings=settings.model_copy(update={"subagent_concurrency": 2}), deps=deps, ctx=CTX)
    assert deps.models["investigator"].gauge["max_seen"] == 2  # pydantic copies the dict, so read it off the model


async def test_evidence_that_does_not_exist_discards_card(settings, agent_deps, seeded_docs):
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    out = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings,
                                   deps=scan(agent_deps, {kc: "bad_evidence"}), ctx=CTX)
    assert out["results"][0]["outcome"] == "discard" and out["results"][0]["reason"] == "evidence not found: D999999"


async def test_unknown_candidate_document_is_discarded(settings, agent_deps):
    out = await run_investigations([{"doc_id": 999_999, "signal_type": "demand", "hypothesis": "?"}],
                                   settings=settings, deps=agent_deps, ctx=CTX)
    assert out["results"][0]["outcome"] == "discard" and "not found" in out["results"][0]["reason"]


async def test_judge_failure_is_recorded_and_fails_only_that_investigation(settings, agent_deps, seeded_docs):
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    deps = scan(agent_deps, {kc: "card"})
    deps = replace(deps, models={**deps.models, "judge": FailingJudge(messages=iter([]))})
    out = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings, deps=deps, ctx=CTX)
    assert out["results"][0]["outcome"] == "failed" and "refused the request" in out["results"][0]["reason"]
    judge_calls = [m for m in deps.sink.model_calls if m.role == "judge"]
    assert len(judge_calls) == 1 and judge_calls[0].ok is False and judge_calls[0].error.startswith("RuntimeError")


async def test_synthetic_candidate_is_flagged_and_its_card_labelled(settings, agent_deps, seeded_docs):
    syn = doc_id(seeded_docs, SYNTHETIC_URL)
    deps = scan(agent_deps, {syn: "card"})
    out = await run_investigations([{"doc_id": syn, "signal_type": "distress", "hypothesis": "Owner distress."}],
                                   settings=settings, deps=deps, ctx=CTX)
    result = out["results"][0]
    assert result["outcome"] == "card" and result["title"] == f"{SYNTHETIC_LABEL} AI firm leases King's Cross offices"
    assert seeded_docs.execute("SELECT card->>'title' FROM opportunities WHERE id = %s",
                               (result["card_id"],)).fetchone()[0] == result["title"]
    assert SYNTHETIC_NOTE in deps.models["investigator"].seen["prompts"][syn]


async def test_investigator_and_judge_are_given_the_portfolio_and_mandate(settings, agent_deps, seeded_docs):
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    deps = scan(agent_deps, {kc: "card"})
    out = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings, deps=deps, ctx=CTX)
    assert out["results"][0]["outcome"] == "card" and not out["results"][0]["title"].startswith(SYNTHETIC_LABEL)
    investigator, judge = deps.models["investigator"].seen, deps.models["judge"].seen
    context = [a["name"] for a in get_portfolio(seeded_docs)] + get_mandate(seeded_docs)["statements"]
    assert all(item in investigator["prompts"][kc] and item in judge["input"] for item in context)
    assert SYNTHETIC_NOTE not in investigator["prompts"][kc]
    assert not {"get_portfolio", "get_mandate"} & investigator["tools"]  # zero-argument tools DeepSeek stalls on


@pytest.mark.parametrize("parallel", [1, 5])
async def test_investigator_returns_a_result_when_its_budget_runs_out(settings, agent_deps, seeded_docs, parallel):
    hungry = ToolHungryInvestigator(messages=iter([]), parallel=parallel)
    deps = replace(agent_deps, models={**agent_deps.models, "investigator": hungry,
                                       "judge": FakeJudge(messages=iter([]))})
    out = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings, deps=deps, ctx=CTX)
    assert (out["results"][0]["outcome"], out["results"][0]["reason"]) == ("discard", BUDGET_DISCARD)
    assert len([m for m in deps.sink.model_calls if m.role == "investigator"]) <= settings.subagent_max_model_calls


async def test_investigator_without_a_result_says_how_it_ended(settings, agent_deps, seeded_docs):
    deps = replace(agent_deps, models={**agent_deps.models, "investigator": TextOnlyInvestigator(messages=iter([]))})
    out = await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL)], settings=settings, deps=deps, ctx=CTX)
    assert out["results"][0]["outcome"] == "failed"
    assert out["results"][0]["reason"] == "the investigator stopped without a result: I could not decide."


async def test_orchestrator_runs_investigations_through_the_tool(settings, agent_deps, seeded_docs):
    kc = doc_id(seeded_docs, KINGS_CROSS_URL)
    orchestrator = scripted([tool_call("load_skill", {"name": "opportunity-scan"}, "1"),
                             tool_call("investigate_signals", {"candidates": [
                                 {"doc_id": kc, "signal_type": "demand", "hypothesis": "AI demand."}]}, "2"),
                             AIMessage("One card drafted for your review.")])
    deps = replace(scan(agent_deps, {kc: "card"}), models={**scan(agent_deps, {kc: "card"}).models,
                                                           "orchestrator": orchestrator})
    r = await run_turn(settings, "scan-t1", "Run the scan", mode="scan", deps=deps)
    results = json.loads(next(t.outcome for t in deps.sink.tool_calls if t.tool == "investigate_signals"))
    assert r.error is None and results["results"][0]["outcome"] == "card"
    assert {m.role for m in deps.sink.model_calls} >= {"orchestrator", "investigator"}


async def test_investigations_report_each_lead_and_its_outcome(settings, agent_deps, seeded_docs):
    kc, tower = doc_id(seeded_docs, KINGS_CROSS_URL), doc_id(seeded_docs, TOWER_SALE_URL)
    steps = []
    ctx = TurnContext(thread_id="scan-steps", turn_id="u2", mode="scan", progress=steps.append)
    await run_investigations([candidate(seeded_docs, KINGS_CROSS_URL), candidate(seeded_docs, TOWER_SALE_URL)],
                             settings=settings, deps=scan(agent_deps, {kc: "card", tower: "discard"}), ctx=ctx)
    assert steps[0] == "Investigating 2 leads in parallel"
    assert f"[D{kc}] Second model checking the card against its evidence" in steps
    assert any(s.startswith(f"✓ D{kc}: card “AI firm leases King's Cross offices” (score ") for s in steps)
    assert f"✗ D{tower}: discarded (Not about London offices.)" in steps

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from london_office_monitor.agent.cards import ScoreInputs, StoredCard
from london_office_monitor.agent.charts import Chart
from london_office_monitor.agent.skills import load_skills
from london_office_monitor.agent.telemetry import ModelCallRecord, ToolCallRecord, TurnSummary
from london_office_monitor.citations import CitationView
from london_office_monitor.cli import print_cards
from london_office_monitor.paths import SKILLS_DIR
from london_office_monitor.ui.view import (card_details_markdown, card_markdown, chart_spec, cited, has_demo_label,
                                           how_i_got_this, linkify_citations, md_safe, next_steps_markdown,
                                           screen_markdown, shown_window, sources_block)

FIGURE = CitationView(citation="F3", kind="figure", title="London Office Market Report Q3 2026", publisher="JLL",
                      published_at=date(2026, 10, 9), retrieved_at=date(2026, 10, 24), url="https://jll.test/q3.pdf",
                      open_url="https://jll.test/q3.pdf#page=4", quoted_passage=None, heading_path=[],
                      also_reported_by=[], figure="JLL · prime rent · City Core · 2026Q3 · £110.00 psf",
                      definition="Top open-market rent for a typical Grade A unit.", synthetic=False)
PASSAGE = CitationView(citation="D9#0", kind="passage", title="AI firm leases offices in King's Cross",
                       publisher="Bisnow London", published_at=datetime(2026, 10, 22, 9, tzinfo=UTC),
                       retrieved_at=datetime(2026, 10, 23, 7, tzinfo=UTC), url="https://bisnow.test/ai",
                       open_url="https://bisnow.test/ai#:~:text=AI%20firm", heading_path=[],
                       quoted_passage="An AI company leased 60,000 sq ft of offices near King's Cross station.",
                       also_reported_by=["City AM"], figure=None, definition=None, synthetic=False)
SYNTHETIC = replace(PASSAGE, citation="D12#0", title="[SYNTHETIC] Example Tower Ltd misses a loan payment",
                    publisher="Synthetic test fixture", url="https://synthetic.invalid/x", open_url=None,
                    also_reported_by=[], synthetic=True)
VIEWS = {v.citation: v for v in (FIGURE, PASSAGE, SYNTHETIC)}

SUMMARY = TurnSummary(
    skills_loaded=["market-questions"],
    tool_calls=[ToolCallRecord("t", "u", "chat", "load_skill", {"name": "market-questions"}, "ok", 2, True),
                ToolCallRecord("t", "u", "chat", "get_metric", {"metric": "prime_rent", "submarket": "City Core"},
                               "{}", 120, True)],
    model_calls=[ModelCallRecord("t", "u", "chat", "orchestrator", "z-ai/glm-5.3", "Z.AI", 1200, 80,
                                 Decimal("0.0021"), 900, True, None)],
    cost_usd=Decimal("0.0021"))

# score_components in the key order Postgres returns jsonb in, which is not the order the parts are shown in.
CARD = StoredCard(card_id=7, status="pending", score=60,
                  score_components={"signal": 20, "mandate": 20, "recency": 12, "confidence": 8},
                  score_inputs=ScoreInputs(mandate_similarity=0.5, closest_match="Regent Quarter",
                                           closest_match_kind="asset", evidence_age_days=9),
                  title="[SYNTHETIC] AI firm leases King's Cross offices", signal_type="demand",
                  submarket="King's Cross/Euston", assets_affected=["Regent Quarter"],
                  summary="An AI company leased 60,000 sq ft near King's Cross station.",
                  why_it_matters="Rent evidence for Regent Quarter.",
                  evidence=[{"kind": "document", "id": 9, "chunk_index": 0, "quote_or_value": "leased 60,000 sq ft"},
                            {"kind": "document", "id": 12, "chunk_index": 0, "quote_or_value": "missed a payment"}],
                  suggested_next_step="Brief the leasing team.", confidence="medium", confidence_reason="One report.")


def test_cited_lists_each_marker_once_in_order():
    assert cited("A [F3], B [D9#0], again [F3] and [D12#0].") == ["F3", "D9#0", "D12#0"]


def test_linkify_citations_links_what_can_be_opened():
    # The brackets stay visible inside the link, so adjacent markers ("[F3][D9#0]") never run together.
    assert linkify_citations("A [F3], B [D9#0], C [D12#0], D [F99]; both [F3][D9#0].", VIEWS.get) == (
        "A [\\[F3\\]](https://jll.test/q3.pdf#page=4), B [\\[D9#0\\]](https://bisnow.test/ai#:~:text=AI%20firm), "
        "C [D12#0], D [F99]; both [\\[F3\\]](https://jll.test/q3.pdf#page=4)[\\[D9#0\\]]"
        "(https://bisnow.test/ai#:~:text=AI%20firm).")


def test_md_safe_escapes_dollar_signs():
    # Streamlit reads $...$ as LaTeX, which would garble "$0.02 to $0.05".
    assert md_safe("costs $0.02 to $0.05") == "costs \\$0.02 to \\$0.05"


def test_demo_labels_read_as_plain_badges_on_screen():
    # The owner asked what [SYNTHETIC] meant. The data, prompts and evals keep the labels; the screen shows badges.
    shown = screen_markdown("[SYNTHETIC] Example Tower Ltd misses a loan payment; see [MOCK] policy, $5 fee")
    assert shown == (":gray-badge[Demo example] Example Tower Ltd misses a loan payment; "
                     "see :gray-badge[Demo document] policy, \\$5 fee")
    assert has_demo_label("[MOCK] London office investment policy")
    assert not has_demo_label("Savills: 15.3m sq ft under construction, Central London, 2026Q3")


def test_sources_block_shows_links_passages_and_synthetic_label():
    block = sources_block([FIGURE, PASSAGE, SYNTHETIC])
    assert "[Open original](https://jll.test/q3.pdf#page=4)" in block and FIGURE.definition in block
    assert PASSAGE.quoted_passage in block and "Also reported by: City AM" in block
    assert "[SYNTHETIC] made-up news item for the demo, with no original to open" in block
    assert block.index("F3") < block.index("D9#0") < block.index("D12#0")


def test_how_i_got_this_lists_skills_tools_grounding_and_cost():
    lines = how_i_got_this(SUMMARY, grounding_rewrites=1, answer="City Core prime rent is £110.00 [F3].")
    assert lines[0] == "Skills: market-questions"
    assert "Tool: get_metric(metric=prime_rent, submarket=City Core)" in lines
    assert "Grounding: every figure and citation matched the data, after 1 rewrite." in lines
    assert "Model: orchestrator · z-ai/glm-5.3 · 1,200 in / 80 out · $0.0021" in lines
    assert lines[-1] == "Total cost: $0.0021"


def test_how_i_got_this_reports_an_unverified_answer():
    lines = how_i_got_this(SUMMARY, grounding_rewrites=2, answer="Rent is £999 [F3].\n\n⚠️ Unverified: £999.")
    assert "Grounding: some figures or citations did not match the data; the answer says which." in lines


def test_a_card_reads_in_plain_words_first():
    # The owner, not a property-finance specialist, could not tell from a card how it was an opportunity or what to
    # do: it now says what happened, what it means for us and what to do next, and folds the detail away.
    md = card_markdown(CARD)
    assert md.startswith(":gray-badge[Demand for space] :gray-badge[King's Cross/Euston] :gray-badge[Score 60/100]")
    assert CARD.title in md and "**What happened**" in md and CARD.summary in md
    assert "**What it means for us**" in md and "*Our buildings involved: Regent Quarter*" in md
    assert "**What to do next**\n\nBrief the leasing team." in md
    assert "**How sure we are:** Fairly sure. One report." in md
    assert "D9#0" not in md and "Score 60/100**" not in md  # the sources and the score's parts are in the fold


@pytest.mark.parametrize(("kind", "badge"), [
    ("chance_to_buy", ":green-badge[:material/check: Chance to buy]"),
    ("risk", ":orange-badge[:material/warning: Risk to our buildings]"),
])
def test_a_card_says_first_what_it_is_to_us(kind, badge):
    # Both of the first real-data cards were risks to buildings Nan Fung owns, yet read as opportunities.
    card = StoredCard.model_validate({**CARD.model_dump(), "kind": kind})  # validated, as cards load from the store
    assert card_markdown(card).startswith(f"{badge} :gray-badge[Demand for space]")


def test_card_details_show_evidence_and_synthetic_items():
    md = card_details_markdown(CARD, VIEWS.get)
    assert "[\\[D9#0\\]](https://bisnow.test/ai#:~:text=AI%20firm) “leased 60,000 sq ft”" in md
    assert "[D12#0] [SYNTHETIC] “missed a payment”" in md


@pytest.mark.parametrize(("written", "shown"), [
    ("1. Ask the leasing agent for costs. 2. Update the forecasts.",
     "1. Ask the leasing agent for costs.\n2. Update the forecasts."),
    ("Brief the leasing team.", "Brief the leasing team."),
    ("1. Check the £1.5m loan with the lender.", "1. Check the £1.5m loan with the lender."),
])
def test_numbered_next_steps_become_a_list(written, shown):
    assert next_steps_markdown(written) == shown


def test_card_markdown_shows_each_score_part_against_its_maximum_with_its_input():
    assert ("**Score 60/100**\n"
            "- Mandate fit 20/40: similarity 0.500 to Regent Quarter (portfolio asset)\n"
            "- Signal type 20/25: demand\n"
            "- Recency 12/20: newest evidence 9 days old\n"
            "- Confidence 8/15: medium") in card_details_markdown(CARD, VIEWS.get)


@pytest.mark.parametrize(("mandate_points", "inputs", "line"), [
    (28, ScoreInputs(mandate_similarity=0.712, closest_match="Buy City Core offices from distressed owners.",
                     closest_match_kind="mandate", evidence_age_days=1),
     "- Mandate fit 28/40: similarity 0.712 to the mandate statement “Buy City Core offices from distressed owners.”"),
    (0, ScoreInputs(mandate_similarity=0.0, closest_match=None, closest_match_kind=None, evidence_age_days=1),
     "- Mandate fit 0/40: no portfolio asset or mandate statement to compare with"),
])
def test_mandate_fit_names_what_the_card_was_closest_to(mandate_points, inputs, line):
    card = CARD.model_copy(update={"score_inputs": inputs,
                                   "score_components": {**CARD.score_components, "mandate": mandate_points}})
    assert line in card_details_markdown(card, VIEWS.get)


@pytest.mark.parametrize(("age_days", "recency_points", "line"), [
    (0, 20, "- Recency 20/20: newest evidence under a day old"),
    (1, 20, "- Recency 20/20: newest evidence 1 day old"),
    (31, 0, "- Recency 0/20: newest evidence 31 days old"),
])
def test_recency_states_the_age_of_the_newest_evidence(age_days, recency_points, line):
    card = CARD.model_copy(update={
        "score_inputs": CARD.score_inputs.model_copy(update={"evidence_age_days": age_days}),
        "score_components": {**CARD.score_components, "recency": recency_points}})
    assert line in card_details_markdown(card, VIEWS.get)


def test_card_scored_before_inputs_were_kept_shows_points_against_maximums():
    # Cards stored before 25 Oct 2026 have no score_inputs; the signal type and confidence are on the card itself.
    assert ("**Score 60/100**\n"
            "- Mandate fit 20/40\n"
            "- Signal type 20/25: demand\n"
            "- Recency 12/20\n"
            "- Confidence 8/15: medium") in card_details_markdown(CARD.model_copy(update={"score_inputs": None}),
                                                                 VIEWS.get)


def test_cli_card_shows_the_same_score_breakdown(capsys):
    print_cards([CARD])
    out = capsys.readouterr().out
    assert "Score 60/100" in out
    assert "Mandate fit 20/40: similarity 0.500 to Regent Quarter (portfolio asset)" in out
    assert "Confidence 8/15: medium" in out


def test_brief_skill_has_the_five_headings_in_order():
    body = load_skills(SKILLS_DIR)["market-brief"].body
    heads = ["## What changed", "## News themes", "## Rates and economy", "## What it means for our portfolio",
             "## Coming up"]
    assert all(h in body for h in heads)
    assert [body.index(h) for h in heads] == sorted(body.index(h) for h in heads)


# Live (25 Oct 2026) the City prime rent chart showed one line for five brokers: four had a single period, which a
# bare line cannot draw; and the zero baseline flattened JLL's rise from £95 to £100.
RENT = Chart(title="Prime rent · City Core", unit="GBP psf pa", x=["2026Q2", "2026Q3"],
             series={"JLL": [("2026Q2", 95.0), ("2026Q3", 100.0)], "Savills": [("2026Q2", 116.73)]},
             citations=["F68", "F57", "F25"])
RATES = Chart(title="Bank Rate and 10-year gilt yield", unit="%", x=["2026-09-10", "2026-10-21"],
              series={"Bank Rate": [("2026-09-10", 3.75), ("2026-10-21", 3.75)],
                      "10-year gilt yield": [("2026-09-10", 5.319), ("2026-10-21", 5.3337)]},
              citations=[], x_kind="date")


def marks(spec: dict) -> dict[str, dict]:
    return {layer["mark"]["type"]: layer for layer in spec["layer"]}


def rows(spec: dict, series: str) -> list[dict]:
    return [r for r in spec["data"]["values"] if r["series"] == series]


def test_every_point_has_a_marker_so_a_broker_with_one_period_still_shows():
    point = marks(chart_spec(RENT))["point"]
    assert point["mark"]["filled"] and point["mark"]["size"] >= 64
    assert [(r["period"], r["value"]) for r in rows(chart_spec(RENT), "Savills")] == [("2026Q2", 116.73)]


def test_brokers_sit_side_by_side_within_a_period_so_equal_figures_stay_visible():
    # Three brokers put City Core prime rent at £95 in 2026Q2; drawn on one spot, only the top marker showed.
    tied = Chart(title="t", unit="u", x=["2026Q2", "2026Q3"], citations=[],
                 series={"BNP Paribas": [("2026Q2", 95.0)], "JLL": [("2026Q2", 95.0), ("2026Q3", 100.0)]})
    spec = chart_spec(tied)
    bnp, (jll_q2, jll_q3) = rows(spec, "BNP Paribas")[0], rows(spec, "JLL")
    assert bnp["x"] != jll_q2["x"] and abs(bnp["x"] - jll_q2["x"]) < 0.5  # apart, but within the same period
    assert jll_q2["x"] < jll_q3["x"] and "line" in marks(spec)  # periods in order, joined by a line


def test_a_dense_rate_series_is_a_plain_line():
    # Markers on 30 daily points broke the line into dashes, which read as a projection.
    days = [f"2026-09-{d:02d}" for d in range(1, 31)]
    dense = Chart(title="t", unit="%", x=days, series={"Bank Rate": [(d, 3.75) for d in days]}, citations=[],
                  x_kind="date")
    assert "point" not in marks(chart_spec(dense)) and "point" in marks(chart_spec(RATES))


def test_the_value_axis_fits_the_data_instead_of_starting_at_zero():
    assert chart_spec(RENT)["encoding"]["y"]["scale"]["zero"] is False


def test_a_broker_keeps_its_colour_and_marker_whatever_else_is_on_the_chart():
    alone = Chart(title="t", unit="u", x=["2026Q2"], series={"Savills": [("2026Q2", 1.0)]}, citations=[])

    def look(chart: Chart, name: str, dark: bool = False) -> tuple[str, str]:
        spec = chart_spec(chart, dark=dark)
        colour = spec["encoding"]["color"]["scale"]
        shape = marks(spec)["point"]["encoding"]["shape"]["scale"]
        i = colour["domain"].index(name)
        return colour["range"][i], shape["range"][shape["domain"].index(name)]

    assert look(RENT, "Savills") == look(alone, "Savills") != look(RENT, "JLL")
    assert look(RENT, "Savills", dark=True)[0] != look(RENT, "Savills")[0]  # dark mode has its own steps


def test_one_series_needs_no_legend_and_two_get_one():
    alone = Chart(title="t", unit="u", x=["2026Q2"], series={"JLL": [("2026Q2", 1.0)]}, citations=[])
    assert chart_spec(alone)["encoding"]["color"]["legend"] is None
    assert chart_spec(RENT)["encoding"]["color"]["legend"] is not None


def test_dates_use_a_time_axis():
    assert chart_spec(RATES)["encoding"]["x"]["type"] == "temporal"


def test_a_rate_chart_says_which_days_it_shows():
    assert shown_window(RATES) == "10 Sep – 21 Oct 2026"
    assert shown_window(Chart(title="t", unit="%", x=["2025-12-30", "2026-01-02"], series={}, citations=[],
                              x_kind="date")) == "30 Dec 2025 – 2 Jan 2026"


# Regression: ISSUE-006 — mock internal documents were shown as "[SYNTHETIC] test item" on cards and in Sources
# Found by /qa on 2026-10-25; report .gstack/qa-reports/qa-report-lom-2026-10-25.md
MOCK_POLICY = replace(PASSAGE, citation="D195#0", title="[MOCK] London office investment policy",
                      publisher="Nan Fung Group (mock)", url="https://internal.invalid/investment-policy",
                      open_url=None, also_reported_by=[], synthetic=True, source_kind="internal")


def test_a_mock_internal_document_reads_as_mock_not_as_a_synthetic_news_item():
    block = sources_block([MOCK_POLICY])
    assert "test item" not in block and "Mock internal document" in block and MOCK_POLICY.quoted_passage in block
    mock_ref = CARD.evidence[0].model_copy(update={"id": 195, "quote_or_value": "policy"})
    card = CARD.model_copy(update={"evidence": [CARD.evidence[0], mock_ref]})
    md = card_details_markdown(card, {**VIEWS, "D195#0": MOCK_POLICY}.get)
    assert "[D195#0] [MOCK] “policy”" in md and "[D195#0] [SYNTHETIC]" not in md

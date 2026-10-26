import pytest
from pydantic import ValidationError

from london_office_monitor.agent.cards import Candidate, EvidenceRef, InvestigationResult, OpportunityCardDraft
from london_office_monitor.scoring import score_card

VALID_CARD = {"title": "Administrators appointed to a City office owner", "signal_type": "distress",
              "submarket": "City Core", "assets_affected": [], "summary": "Administrators were appointed.",
              "why_it_matters": "Fits the value-add brief.",
              "evidence": [{"kind": "document", "id": 1, "chunk_index": 0, "quote_or_value": "administrators"}],
              "suggested_next_step": "Contact the administrators.", "confidence": "medium",
              "confidence_reason": "One notice; ownership inferred."}


def test_valid_card_is_accepted():
    assert OpportunityCardDraft(**VALID_CARD).evidence[0] == EvidenceRef(kind="document", id=1, chunk_index=0,
                                                                         quote_or_value="administrators")


def test_summary_word_limit_enforced():
    with pytest.raises(ValidationError):
        OpportunityCardDraft(**{**VALID_CARD, "summary": " ".join(["word"] * 81)})


def test_why_it_matters_word_limit_enforced():
    with pytest.raises(ValidationError):
        OpportunityCardDraft(**{**VALID_CARD, "why_it_matters": " ".join(["word"] * 61)})


@pytest.mark.parametrize("evidence", [[], [VALID_CARD["evidence"][0]] * 7])
def test_evidence_count_is_bounded(evidence):
    with pytest.raises(ValidationError):
        OpportunityCardDraft(**{**VALID_CARD, "evidence": evidence})


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        OpportunityCardDraft(**VALID_CARD, colour="blue")


def test_investigation_result_consistency():
    with pytest.raises(ValidationError):
        InvestigationResult(outcome="card", card=None, discard_reason=None)
    with pytest.raises(ValidationError):
        InvestigationResult(outcome="discard", card=OpportunityCardDraft(**VALID_CARD), discard_reason=None)
    assert InvestigationResult(outcome="discard", card=None, discard_reason="not London").discard_reason


def test_candidate_hypothesis_is_bounded():
    with pytest.raises(ValidationError):
        Candidate(doc_id=1, signal_type="demand", hypothesis="x" * 301)


def test_score_components():
    assert score_card("distress", 0.5, 3, "medium") == (73, {"mandate": 20, "signal": 25, "recency": 20,
                                                             "confidence": 8})


def test_recency_bands():
    assert score_card("other", 0.0, 10, "low")[1]["recency"] == 12
    assert score_card("other", 0.0, 30, "low")[1]["recency"] == 6
    assert score_card("other", 0.0, 40, "low")[1]["recency"] == 0


def test_mandate_similarity_is_clamped():
    assert score_card("demand", 1.7, 0, "high")[1]["mandate"] == 40
    assert score_card("demand", -0.2, 0, "high")[1]["mandate"] == 0

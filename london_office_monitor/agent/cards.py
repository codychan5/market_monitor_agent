"""Opportunity cards and the shapes around them. Word and size limits keep a card readable in a feed."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..vocab import CardKind, SignalType, Submarket

TITLE_MAX_CHARS = 120
SUMMARY_MAX_WORDS = 80
WHY_MAX_WORDS = 60
# Room for two or three short numbered steps (the owner asked what to do with a card, 26 Oct 2026).
NEXT_STEP_MAX_CHARS = 300
MAX_EVIDENCE_ITEMS = 6


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _within_words(text: str, limit: int) -> str:
    if len(text.split()) > limit:
        raise ValueError(f"must be at most {limit} words")
    return text


class EvidenceRef(StrictModel):
    kind: Literal["document", "fact"]
    id: int
    quote_or_value: str = Field(max_length=300)
    chunk_index: int | None = None


class OpportunityCardDraft(StrictModel):
    title: str = Field(min_length=1, max_length=TITLE_MAX_CHARS)
    signal_type: SignalType
    # None on cards written before 26 Oct 2026; the investigator is told to always set it.
    kind: CardKind | None = None
    submarket: Submarket
    assets_affected: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=MAX_EVIDENCE_ITEMS)
    suggested_next_step: str = Field(min_length=1, max_length=NEXT_STEP_MAX_CHARS)
    confidence: Literal["high", "medium", "low"]
    confidence_reason: str = Field(min_length=1, max_length=200)

    @field_validator("summary")
    @classmethod
    def _summary_words(cls, text: str) -> str:
        return _within_words(text, SUMMARY_MAX_WORDS)

    @field_validator("why_it_matters")
    @classmethod
    def _why_words(cls, text: str) -> str:
        return _within_words(text, WHY_MAX_WORDS)


class InvestigationResult(StrictModel):
    """What an investigator sub-agent returns: a card, or a discard with its reason."""
    outcome: Literal["card", "discard"]
    card: OpportunityCardDraft | None = None
    discard_reason: str | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if (self.outcome == "card") != (self.card is not None):
            raise ValueError("card must be set exactly when outcome is 'card'")
        if self.outcome == "discard" and not self.discard_reason:
            raise ValueError("a discard needs a discard_reason")
        return self


class Review(StrictModel):
    """The judge's verdict on whether a card's claims are backed by its evidence."""
    verdict: Literal["supported", "unsupported"]
    notes: str


class Candidate(StrictModel):
    doc_id: int
    signal_type: SignalType
    hypothesis: str = Field(min_length=1, max_length=300)


class ScoreInputs(StrictModel):
    """What a card's score was computed from, kept with the card so each part can be traced to its input. The
    signal type and confidence are on the card itself."""
    mandate_similarity: float = Field(ge=-1, le=1)
    closest_match: str | None
    closest_match_kind: Literal["asset", "mandate"] | None
    evidence_age_days: int = Field(ge=0)


class StoredCard(OpportunityCardDraft):
    card_id: int
    status: str
    score: int | None = None
    score_components: dict[str, int] = Field(default_factory=dict)
    # None for cards scored before the inputs were kept (25 Oct 2026).
    score_inputs: ScoreInputs | None = None
    review_notes: str | None = None
    decision_reason: str | None = None


class InvestigationOutcome(StrictModel):
    doc_id: int
    outcome: Literal["card", "discard", "failed"]
    card_id: int | None = None
    title: str | None = None
    score: int | None = None
    reason: str | None = None

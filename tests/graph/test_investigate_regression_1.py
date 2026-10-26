# Regression: ISSUE-005 — a card on real news was labelled [SYNTHETIC] because it also cited a mock internal policy
# Found by /qa on 2026-10-25
# Report: .gstack/qa-reports/qa-report-lom-2026-10-25.md
from pathlib import Path

import pytest

from london_office_monitor.agent.cards import EvidenceRef, OpportunityCardDraft
from london_office_monitor.agent.subagents import _rests_on_synthetic
from london_office_monitor.seed.load import load_internal
from tests.conftest import KINGS_CROSS_URL, SYNTHETIC_URL, TEST_DB, doc_id

pytestmark = pytest.mark.db
TEST_INTERNAL = Path(__file__).parents[1] / "data" / "internal"


def card_on(*doc_ids: int) -> OpportunityCardDraft:
    return OpportunityCardDraft(
        title="An AI firm leases King's Cross offices", signal_type="demand", submarket="King's Cross/Euston",
        summary="An AI company leased offices near King's Cross.", why_it_matters="Rent evidence for Regent Quarter.",
        evidence=[EvidenceRef(kind="document", id=i, quote_or_value="quote") for i in doc_ids],
        suggested_next_step="Brief the leasing team.", confidence="medium", confidence_reason="One report.")


def test_citing_a_mock_internal_document_does_not_make_a_real_signal_synthetic(seeded_docs, fake_embedder, settings):
    # Live (25 Oct 2026) a real FT story about loan defaults came out as "[SYNTHETIC] ..." because the investigator
    # also cited the mock investment policy, which is stored as synthetic. The label is about the market signal.
    load_internal(seeded_docs, TEST_INTERNAL, fake_embedder, settings)
    mock_policy = seeded_docs.execute("SELECT min(id) FROM documents WHERE source_kind = 'internal'").fetchone()[0]
    real, fake = doc_id(seeded_docs, KINGS_CROSS_URL), doc_id(seeded_docs, SYNTHETIC_URL)
    assert _rests_on_synthetic(settings, TEST_DB, card_on(real, mock_policy), real) is False
    assert _rests_on_synthetic(settings, TEST_DB, card_on(fake, mock_policy), fake) is True

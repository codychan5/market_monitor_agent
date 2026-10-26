import re
from datetime import UTC, datetime

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from london_office_monitor.embed import get_embedder
from london_office_monitor.facts import NotInData
from london_office_monitor.ingest.feeds import DocumentIn
from london_office_monitor.ingest.store import store_documents
from london_office_monitor.search import match_to_mandate, search_documents, similar_past_decisions
from tests.conftest import (CITY_ROAD_URL, DEMO_NOW, FUTURE_URL, KINGS_CROSS_URL, TOWER_SALE_URL, doc_id)
from tests.fakes import ConstantEmbedder


@pytest.mark.db
def test_search_excludes_documents_after_until(seeded_docs, fake_embedder):
    hits = search_documents(seeded_docs, fake_embedder, "office letting", until=DEMO_NOW)
    assert hits and all(h.published_at <= DEMO_NOW for h in hits)
    assert doc_id(seeded_docs, FUTURE_URL) not in [h.doc_id for h in hits]


@pytest.mark.db
def test_search_groups_one_hit_per_story(seeded_docs, fake_embedder):
    hits = search_documents(seeded_docs, fake_embedder, "Canary Wharf tower sale", until=DEMO_NOW)
    top = hits[0]
    assert top.doc_id == doc_id(seeded_docs, TOWER_SALE_URL) and top.also_reported_by == ["Bisnow London"]
    assert len({h.story_id for h in hits}) == len(hits)


@pytest.mark.db
def test_search_finds_name_by_full_text(seeded_docs, fake_embedder):
    top = search_documents(seeded_docs, fake_embedder, '"99 City Road"', until=DEMO_NOW)[0]
    assert top.doc_id == doc_id(seeded_docs, CITY_ROAD_URL) and "99 City Road" in top.quoted_title


@pytest.mark.db
def test_search_finds_a_word_that_is_only_in_the_headline(fresh_db, settings):
    # A news chunk is the feed summary, without the headline. With every vector equal, the earlier document wins the
    # vector rankings, so only a keyword match on the later one's headline can put it first. The two are ten days
    # apart, so their equal vectors do not make them one story.
    embedder = ConstantEmbedder()
    store_documents(fresh_db, [
        DocumentIn(source_kind="news", publisher="City AM", title="Docklands office market update",
                   summary="Leasing was steady across the area this quarter.", url="https://news.test/docklands",
                   published_at=datetime(2026, 10, 10, 9, tzinfo=UTC)),
        DocumentIn(source_kind="news", publisher="Bisnow London", title="Barclays buys Churchill Place",
                   summary="The bank bought its headquarters building.", url="https://news.test/churchill-place",
                   published_at=datetime(2026, 10, 20, 9, tzinfo=UTC))], embedder, settings)
    top = search_documents(fresh_db, embedder, "Churchill", until=DEMO_NOW)[0]
    assert top.doc_id == doc_id(fresh_db, "https://news.test/churchill-place")


@pytest.mark.db
def test_search_returns_chunk_level_citations(seeded_docs, fake_embedder):
    hits = search_documents(seeded_docs, fake_embedder, "King's Cross letting", until=DEMO_NOW)
    assert hits and all(re.fullmatch(r"D\d+#\d+", h.citation) for h in hits)
    assert hits[0].quoted_passage.startswith("An artificial intelligence company")


@pytest.mark.db
def test_search_respects_k_and_source_types(seeded_docs, fake_embedder):
    assert len(search_documents(seeded_docs, fake_embedder, "office", until=DEMO_NOW, k=2)) <= 2
    assert search_documents(seeded_docs, fake_embedder, "office", until=DEMO_NOW, source_types=["gazette"]) == []


@pytest.mark.db
def test_match_to_mandate_ranks_regent_quarter_first_for_kings_cross_doc(seeded_docs):
    ranked = match_to_mandate(seeded_docs, doc_id(seeded_docs, KINGS_CROSS_URL))
    assert ranked[0]["target"] == "Regent Quarter" and ranked[0]["kind"] == "asset"
    assert [r["similarity"] for r in ranked] == sorted((r["similarity"] for r in ranked), reverse=True)


@pytest.mark.db
def test_match_to_mandate_unknown_document(seeded_docs):
    with pytest.raises(NotInData):
        match_to_mandate(seeded_docs, 999_999)


@pytest.mark.db
def test_similar_past_decisions_returns_decided_cards_with_reasons(seeded_docs, fake_embedder):
    for title, status, reason in (("City office owner in administration", "rejected", "Too small for us"),
                                  ("Retail park sale in Kent", "approved", None),
                                  ("City office owner in administration again", "draft", None)):
        seeded_docs.execute(
            "INSERT INTO opportunities (thread_id, status, card, decision_reason, embedding) VALUES (%s, %s, %s, %s, %s)",
            ("t", status, Jsonb({"title": title}), reason, np.asarray(fake_embedder.embed_documents([title])[0],
                                                                    dtype=np.float32)))
    found = similar_past_decisions(seeded_docs, fake_embedder, "City office owner goes into administration")
    assert [(f["title"], f["decision"], f["reason"]) for f in found[:2]] == [
        ("City office owner in administration", "rejected", "Too small for us"),
        ("Retail park sale in Kent", "approved", None)]


@pytest.mark.db
@pytest.mark.embedding
def test_chinese_query_finds_english_document(seeded_docs_real_embedder, settings):
    hits = search_documents(seeded_docs_real_embedder, get_embedder(settings), "國王十字 人工智能 公司 租用 辦公室",
                            until=DEMO_NOW)
    assert doc_id(seeded_docs_real_embedder, KINGS_CROSS_URL) in [h.doc_id for h in hits[:3]]

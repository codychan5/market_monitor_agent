"""Internal documents: their own source in the vector store, chunked by headings, searchable on their own."""
from datetime import UTC, datetime
from pathlib import Path

import pytest

from london_office_monitor.ingest.feeds import DocumentIn
from london_office_monitor.ingest.store import store_documents
from london_office_monitor.search import MAX_PASSAGE_CHARS, search_documents
from london_office_monitor.seed.load import load_internal
from tests.conftest import DEMO_NOW
from tests.mcp_helpers import call

pytestmark = pytest.mark.db
TEST_INTERNAL = Path(__file__).parents[1] / "data" / "internal"
# The three passages of the test investment-committee note, as (heading path, text).
EXAMPLE_HOUSE_PASSAGES = [
    ([], "MOCK INTERNAL DOCUMENT: test data, not a Nan Fung record."),
    (["Proposal"], "Buy Example House, a 45,000 sq ft City Core office, from Example Office Holdings Limited."),
    (["Decision"], "Declined on 14 March 2024: the price was above our replacement-cost estimate."),
]


def _internal(body: str, url: str) -> DocumentIn:
    return DocumentIn(source_kind="internal", publisher="Nan Fung (internal)", title="Asset management note",
                      subtitle="Note", summary=body, url=url, published_at=datetime(2025, 1, 6, 9, tzinfo=UTC),
                      text_format="markdown")


def test_internal_documents_load_as_their_own_labelled_source(fresh_db, fake_embedder, settings):
    stored = load_internal(fresh_db, TEST_INTERNAL, fake_embedder, settings)
    rows = fresh_db.execute("SELECT source_kind, synthetic, title, url FROM documents ORDER BY id").fetchall()
    assert stored.new == 2 and {r[0] for r in rows} == {"internal"} and all(r[1] for r in rows)
    assert all(r[2].startswith("[MOCK]") and r[3].startswith("https://internal.invalid/") for r in rows)
    chunks = fresh_db.execute("SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
                              " WHERE d.title LIKE '%%Example House%%'").fetchone()[0]
    assert chunks >= 2  # chunked along its headings
    assert load_internal(fresh_db, TEST_INTERNAL, fake_embedder, settings).new == 0  # loading again adds nothing


def test_internal_documents_are_searched_on_their_own(seeded_docs, fake_embedder, settings):
    load_internal(seeded_docs, TEST_INTERNAL, fake_embedder, settings)
    internal = search_documents(seeded_docs, fake_embedder, "Example House decision", until=DEMO_NOW,
                                source_types=["internal"])
    news = search_documents(seeded_docs, fake_embedder, "Example House decision", until=DEMO_NOW,
                            source_types=["news"])
    assert internal and {h.source_kind for h in internal} == {"internal"}
    assert news and {h.source_kind for h in news} == {"news"}


async def test_the_search_tool_takes_internal_as_a_source_type(mcp_tools):
    out = await call(mcp_tools, "search_documents", query="Example House", source_types=["internal"])
    assert "error" not in out and "hits" in out


def test_an_internal_document_is_stored_whole_up_to_the_column_limit(fresh_db, fake_embedder, settings):
    # Far longer than a feed summary's 1,000 characters, and exactly the 50,000 the documents table allows an internal
    # document. The investigator reads the stored text, so it must be the document that was chunked, not a cut.
    body = ("Lease terms and covenants. " * 1852)[:50_000]
    store_documents(fresh_db, [_internal(body, "https://internal.invalid/long-note")], fake_embedder, settings)
    assert fresh_db.execute("SELECT summary FROM documents").fetchone()[0] == body


def test_a_hit_carries_the_other_passages_of_its_document(seeded_docs, fake_embedder, settings):
    # A question about a past decision can match the proposal while the answer is in the decision, so the hit brings
    # the note's other passages, each under its own citation.
    load_internal(seeded_docs, TEST_INTERNAL, fake_embedder, settings)
    hit = search_documents(seeded_docs, fake_embedder, "Example House", until=DEMO_NOW, source_types=["internal"])[0]
    passages = [(hit.citation, hit.heading_path, hit.quoted_passage)] + [
        (p.citation, p.heading_path, p.quoted_passage) for p in hit.nearby_passages]
    assert sorted((path, text) for _, path, text in passages) == sorted(EXAMPLE_HOUSE_PASSAGES)
    assert sorted(c for c, _, _ in passages) == [f"D{hit.doc_id}#0", f"D{hit.doc_id}#1", f"D{hit.doc_id}#2"]


def test_nearby_passages_are_the_closest_ones_that_fit_the_budget(fresh_db, fake_embedder, settings):
    # Eight sections of one full-length passage each, and a budget of four. Nearest the hit (section 4) first, the
    # earlier one at equal distance: 3, 5, 2 and 6 fit and 1 does not, so 2, 3, 5 and 6 come back in reading order.
    def section(i: int) -> str:
        line = f"Section {i} covers {'zanzibar' if i == 4 else 'routine'} matters."
        return f"## Part {i}\n\n{line} {'x' * (MAX_PASSAGE_CHARS - len(line) - 1)}"

    body = "\n\n".join(section(i) for i in range(8))
    store_documents(fresh_db, [_internal(body, "https://internal.invalid/eight-parts")], fake_embedder, settings)
    hit = search_documents(fresh_db, fake_embedder, "zanzibar", until=DEMO_NOW)[0]
    assert hit.chunk_index == 4
    assert [p.citation for p in hit.nearby_passages] == [f"D{hit.doc_id}#{i}" for i in (2, 3, 5, 6)]


async def test_the_search_tool_returns_nearby_passages_with_their_citations(seeded_docs, fake_embedder, settings,
                                                                             mcp_tools):
    load_internal(seeded_docs, TEST_INTERNAL, fake_embedder, settings)
    out = await call(mcp_tools, "search_documents", query="Example House", source_types=["internal"])
    hit = out["hits"][0]
    passages = {hit["citation"]: hit["quoted_passage"]} | {p["citation"]: p["quoted_passage"]
                                                           for p in hit["nearby_passages"]}
    assert sorted(passages.values()) == sorted(text for _, text in EXAMPLE_HOUSE_PASSAGES)

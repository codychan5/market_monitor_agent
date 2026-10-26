import pytest

from london_office_monitor.citations import CITATION_PATTERN, get_citation, pdf_page_url, text_fragment_url
from london_office_monitor.facts import NotInData
from tests.conftest import KINGS_CROSS_URL, SYNTHETIC_URL, TOWER_SALE_URL, doc_id, fact_id


def test_citation_pattern_matches_figures_and_passages_only():
    assert CITATION_PATTERN.findall("a [F12], b [D3#0], c [S4], d [D7]") == ["F12", "D3#0"]


def test_text_fragment_url():
    assert text_fragment_url("https://x.test/a#top", "Barclays buys One Churchill Place") == \
        "https://x.test/a#:~:text=Barclays%20buys%20One%20Churchill%20Place"


def test_text_fragment_url_encodes_dashes_and_commas():
    assert text_fragment_url("https://x.test/a", "Build-to-rent, approved") == \
        "https://x.test/a#:~:text=Build%2Dto%2Drent%2C%20approved"


def test_pdf_page_url_only_for_pdfs():
    assert pdf_page_url("https://x.test/r.pdf", "p. 3") == "https://x.test/r.pdf#page=3"
    assert pdf_page_url("https://x.test/r.pdf", "page 12") == "https://x.test/r.pdf#page=12"
    assert pdf_page_url("https://x.test/page", "p. 3") == "https://x.test/page"
    assert pdf_page_url("https://x.test/r.pdf", None) == "https://x.test/r.pdf"


@pytest.mark.db
def test_get_citation_for_figure_and_passage(seeded_docs):
    fig = get_citation(seeded_docs, f"F{fact_id(seeded_docs, 'JLL', 'prime_rent', 'City Core', '2026Q3')}")
    assert fig.kind == "figure" and fig.figure == "JLL · prime rent · City Core · 2026Q3 · £110.00 psf · headline prime"
    assert fig.open_url == "https://broker-a.test/reports/q2.pdf#page=3" and fig.publisher == "Test Broker A"
    passage = get_citation(seeded_docs, f"D{doc_id(seeded_docs, KINGS_CROSS_URL)}#0")
    assert passage.kind == "passage" and passage.open_url == (
        KINGS_CROSS_URL + "#:~:text=AI%20firm%20leases%20offices%20in%20King%27s%20Cross")
    assert passage.quoted_passage.startswith("An artificial intelligence company")


@pytest.mark.db
def test_passage_citation_lists_the_other_outlets(seeded_docs):
    view = get_citation(seeded_docs, f"D{doc_id(seeded_docs, TOWER_SALE_URL)}#0")
    assert view.also_reported_by == ["Bisnow London"]


@pytest.mark.db
def test_synthetic_citation_has_no_external_link(seeded_docs):
    view = get_citation(seeded_docs, f"D{doc_id(seeded_docs, SYNTHETIC_URL)}#0")
    assert view.open_url is None and view.synthetic


@pytest.mark.db
@pytest.mark.parametrize("bad", ["F999999", "D999999#0", "S12", "D3", "nonsense"])
def test_unknown_or_malformed_citation_raises_not_in_data(seeded_docs, bad):
    with pytest.raises(NotInData):
        get_citation(seeded_docs, bad)

"""Citations the analyst can check (spec §4.10): [F123] is one stored figure, [D12#3] is chunk 3 of document 12."""
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from urllib.parse import quote, urlsplit

import psycopg

from .facts import FACT_SELECT, NotInData, _fact, format_value
from .search import other_outlets

# The only definition of a citation marker; the grounding check and the UI both use it.
CITATION_PATTERN = re.compile(r"\[(F\d+|D\d+#\d+)\]")
CITATION_ID = re.compile(r"^(?:F(\d+)|D(\d+)#(\d+))$")
PDF_PAGE_LOCATOR = re.compile(r"^p(?:age)?\.?\s*(\d+)$", re.IGNORECASE)
# How many words of the title or passage go into a text-fragment link: enough to be unique on the page, short
# enough to survive small edits by the publisher.
FRAGMENT_WORDS = 8


@dataclass(frozen=True)
class CitationView:
    citation: str
    kind: Literal["figure", "passage"]
    title: str
    publisher: str
    published_at: date | datetime | None
    retrieved_at: date | datetime | None
    url: str
    open_url: str | None
    quoted_passage: str | None
    heading_path: list[str]
    also_reported_by: list[str]
    figure: str | None
    definition: str | None
    synthetic: bool
    source_kind: str | None = None  # a passage's document kind (news, gazette, internal); None for a figure


def text_fragment_url(url: str, phrase: str) -> str:
    """The URL with a text fragment, so supporting browsers scroll to and highlight the phrase. Dashes and commas
    have meaning in fragment syntax, so they are percent-encoded too."""
    base = url.split("#", 1)[0]
    return f"{base}#:~:text={quote(phrase, safe='').replace('-', '%2D')}"


def pdf_page_url(url: str, locator: str | None) -> str:
    """A PDF link opened at the locator's page (#page=N); any other URL or locator is returned unchanged."""
    match = PDF_PAGE_LOCATOR.match(locator.strip()) if locator else None
    if match is None or not urlsplit(url).path.lower().endswith(".pdf"):
        return url
    return f"{url.split('#', 1)[0]}#page={match.group(1)}"


def _first_words(text: str) -> str:
    return " ".join(text.split()[:FRAGMENT_WORDS])


def _figure(conn: psycopg.Connection, citation: str, fact_id: int) -> CitationView:
    row = conn.execute(f"{FACT_SELECT} WHERE m.id = %s", (fact_id,)).fetchone()
    if row is None:
        raise NotInData(f"No figure {citation}.")
    fact = _fact(row)
    source = conn.execute("SELECT s.title, s.publisher, s.published, s.url, m.checked_at FROM broker_metrics m"
                          " JOIN sources s ON s.id = m.source_id WHERE m.id = %s", (fact_id,)).fetchone()
    figure = " · ".join([fact.broker, fact.metric.replace("_", " "), fact.submarket, fact.period,
                         format_value(fact.value, fact.unit), fact.measure_note])
    return CitationView(citation=citation, kind="figure", title=source[0], publisher=source[1],
                        published_at=source[2], retrieved_at=source[4], url=source[3],
                        open_url=pdf_page_url(source[3], fact.locator), quoted_passage=None, heading_path=[],
                        also_reported_by=[], figure=figure, definition=fact.definition, synthetic=False)


def _passage(conn: psycopg.Connection, citation: str, doc_id: int, chunk_index: int) -> CitationView:
    row = conn.execute(
        "SELECT d.title, d.publisher, d.published_at, d.fetched_at, d.url, d.synthetic, d.source_kind, d.story_id,"
        " c.text,"
        " c.heading_path, (SELECT count(*) FROM chunks x WHERE x.document_id = d.id)"
        " FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.id = %s AND c.chunk_index = %s",
        (doc_id, chunk_index)).fetchone()
    if row is None:
        raise NotInData(f"No passage {citation}.")
    title, publisher, published, fetched, url, synthetic, source_kind, story_id, text, heading_path, chunk_count = row
    phrase = _first_words(title if chunk_count == 1 else text)
    return CitationView(citation=citation, kind="passage", title=title, publisher=publisher, published_at=published,
                        retrieved_at=fetched, url=url, open_url=None if synthetic else text_fragment_url(url, phrase),
                        quoted_passage=text, heading_path=list(heading_path),
                        also_reported_by=other_outlets(conn, story_id, publisher), figure=None, definition=None,
                        synthetic=synthetic, source_kind=source_kind)


def get_citation(conn: psycopg.Connection, citation_id: str) -> CitationView:
    """What a citation marker points at, with the link that opens the original."""
    match = CITATION_ID.match(citation_id.strip())
    if match is None:
        raise NotInData(f"{citation_id!r} is not a citation: use F123 for a figure or D12#3 for a passage.")
    if match.group(1):
        return _figure(conn, citation_id, int(match.group(1)))
    return _passage(conn, citation_id, int(match.group(2)), int(match.group(3)))

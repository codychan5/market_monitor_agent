import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from london_office_monitor.ingest.feeds import FEEDS, DocumentIn
from london_office_monitor.ingest.gazette import GAZETTE_NOTICE_TYPES
from london_office_monitor.ingest.http import HttpClient
from london_office_monitor.ingest.run import MAX_PAGES_PER_SOURCE, ingest
from london_office_monitor.ingest.store import store_documents
from london_office_monitor.vocab import SourceKind
from tests.conftest import TEST_DATA

SUMMARY = "Barclays has bought its Canary Wharf headquarters, One Churchill Place, for £750m."
DOC_A = DocumentIn(source_kind="news", publisher="City AM", title="Barclays buys One Churchill Place for £750m",
                   summary=SUMMARY, url="https://a.test/1", published_at=datetime(2026, 10, 20, 9, tzinfo=UTC),
                   meta={})


def _same_story(publisher: str, url: str, days_after: int) -> DocumentIn:
    return DocumentIn(source_kind="news", publisher=publisher, title=DOC_A.title, summary=SUMMARY, url=url,
                      published_at=DOC_A.published_at + timedelta(days=days_after), meta={})


def _no_sleep_client(settings) -> HttpClient:
    return HttpClient(settings, sleep=lambda s: None)


def _mock_feeds(respx_mock, **bodies: httpx.Response) -> None:
    """Every news feed 404s unless given; keyword names are publisher names with spaces as underscores."""
    for publisher, url in FEEDS.items():
        respx_mock.get(url).mock(return_value=bodies.get(publisher.replace(" ", "_"), httpx.Response(404)))


@pytest.mark.db
def test_store_creates_one_chunk_per_news_item(fresh_db, fake_embedder, settings):
    store_documents(fresh_db, [DOC_A], fake_embedder, settings)
    rows = fresh_db.execute("SELECT c.chunk_index, c.text FROM chunks c JOIN documents d ON d.id = c.document_id"
                            " WHERE d.url = %s", (DOC_A.url,)).fetchall()
    assert rows == [(0, SUMMARY)]


@pytest.mark.db
def test_exact_duplicate_is_skipped(fresh_db, fake_embedder, settings):
    r = store_documents(fresh_db, [DOC_A, DOC_A], fake_embedder, settings)
    assert (r.new, r.duplicates) == (1, 1)


@pytest.mark.db
def test_near_duplicate_joins_story(fresh_db, fake_embedder, settings):
    store_documents(fresh_db, [DOC_A, _same_story("Bisnow London", "https://b.test/1", 1),
                               _same_story("Property Week", "https://c.test/1", 5)], fake_embedder, settings)
    stories = dict(fresh_db.execute("SELECT url, story_id FROM documents"))
    assert stories["https://b.test/1"] == stories["https://a.test/1"]
    assert stories["https://c.test/1"] != stories["https://a.test/1"]


@pytest.mark.db
def test_demo_mode_never_fetches(fresh_db, settings, fake_embedder, respx_mock):
    report = ingest(settings, {SourceKind.NEWS, SourceKind.GAZETTE, SourceKind.BOE}, conn=fresh_db,
                    embedder=fake_embedder, http=_no_sleep_client(settings))
    assert report.per_source and all(o.skipped.startswith("demo mode") for o in report.per_source.values())
    assert respx_mock.calls.call_count == 0


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_html_page_instead_of_feed_fails_that_source_only(fresh_db, live_settings, fake_embedder, respx_mock):
    _mock_feeds(respx_mock, City_AM=httpx.Response(200, html="<!DOCTYPE html><html>captive portal</html>"),
                Bisnow_London=httpx.Response(200, content=TEST_DATA.joinpath("feeds/bisnow.xml").read_bytes()))
    report = ingest(live_settings, {SourceKind.NEWS}, conn=fresh_db, embedder=fake_embedder,
                    http=_no_sleep_client(live_settings))
    assert report.per_source["City AM"].error and report.per_source["Bisnow London"].new == 2
    assert fresh_db.execute("SELECT count(*) FROM documents WHERE publisher = 'City AM'").fetchone()[0] == 0
    assert fresh_db.execute("SELECT count(*) FROM ingest_runs").fetchone()[0] == len(FEEDS)


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_min_interval_skips_recent_source(fresh_db, live_settings, fake_embedder, respx_mock):
    _mock_feeds(respx_mock, Bisnow_London=httpx.Response(
        200, content=TEST_DATA.joinpath("feeds/bisnow.xml").read_bytes()))
    for _ in range(2):
        report = ingest(live_settings, {SourceKind.NEWS}, conn=fresh_db, embedder=fake_embedder,
                        http=_no_sleep_client(live_settings))
    assert report.per_source["Bisnow London"].skipped.startswith("refreshed at")
    assert respx_mock.routes[list(FEEDS).index("Bisnow London")].call_count == 1


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_gazette_keeps_only_london_property_owners(fresh_db, live_settings, fake_embedder, respx_mock):
    respx_mock.get(url__startswith="https://www.thegazette.co.uk/").mock(
        return_value=httpx.Response(200, content=TEST_DATA.joinpath("gazette_2410.xml").read_bytes()))
    ingest(live_settings, {SourceKind.GAZETTE}, conn=fresh_db, embedder=fake_embedder,
           http=_no_sleep_client(live_settings))
    rows = fresh_db.execute("SELECT DISTINCT title, meta->>'filter_reason' FROM documents").fetchall()
    assert [r[0] for r in rows] == ["EXAMPLE CITY PROPERTIES LIMITED"] and "Central London" in rows[0][1]


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_gazette_is_asked_for_a_hundred_notices_of_each_type(fresh_db, live_settings, fake_embedder, respx_mock):
    # Measured live on 25 Oct 2026: the feed's default page of 10 covered about 3 hours of winding-up petitions, and
    # 100 is the most it serves per page.
    route = respx_mock.get(url__startswith="https://www.thegazette.co.uk/").mock(
        return_value=httpx.Response(200, content=TEST_DATA.joinpath("gazette_2410.xml").read_bytes()))
    ingest(live_settings, {SourceKind.GAZETTE}, conn=fresh_db, embedder=fake_embedder,
           http=_no_sleep_client(live_settings))
    queries = [parse_qs(urlsplit(str(call.request.url)).query) for call in route.calls]
    assert {q["noticetypes"][0] for q in queries} == set(GAZETTE_NOTICE_TYPES)
    assert all(q.get("results-page-size") == ["100"] for q in queries)


PAGED_PUBLISHER = "Property Industry Eye"


def _rss(*stories: tuple[str, datetime]) -> bytes:
    """A minimal RSS feed with one item per (slug, published)."""
    items = "".join(f"<item><title>Story {slug}</title><link>https://feed.test/{slug}</link>"
                    f"<pubDate>{format_datetime(published)}</pubDate><description>Story {slug}.</description></item>"
                    for slug, published in stories)
    return (f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Test</title>{items}'
            f"</channel></rss>").encode()


def _mock_paged_feed(respx_mock, pages: dict[int, bytes]):
    """PAGED_PUBLISHER answers ?paged=N with pages[N], or 404 when there is none; every other feed 404s. Returns the
    publisher's route, whose calls show the pages requested."""
    for publisher, url in FEEDS.items():
        if publisher != PAGED_PUBLISHER:
            respx_mock.get(url__startswith=url).mock(return_value=httpx.Response(404))

    def respond(request: httpx.Request) -> httpx.Response:
        page = pages.get(int(request.url.params.get("paged", "1")))
        return httpx.Response(200, content=page) if page is not None else httpx.Response(404)

    return respx_mock.get(url__startswith=FEEDS[PAGED_PUBLISHER]).mock(side_effect=respond)


def _pages_requested(route) -> list[int]:
    return [int(call.request.url.params.get("paged", "1")) for call in route.calls]


def _ingest_news(conn, settings, embedder, **kwargs):
    return ingest(settings, {SourceKind.NEWS}, conn=conn, embedder=embedder, http=_no_sleep_client(settings),
                  **kwargs).per_source[PAGED_PUBLISHER]


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_ingest_pages_back_by_default_until_items_are_older_than_the_lookback(fresh_db, live_settings, fake_embedder,
                                                                              respx_mock):
    now = datetime.now(UTC)
    route = _mock_paged_feed(respx_mock, {1: _rss(("a", now - timedelta(hours=1)), ("b", now - timedelta(hours=12))),
                                          2: _rss(("c", now - timedelta(days=2)), ("d", now - timedelta(days=10))),
                                          3: _rss(("e", now - timedelta(days=11)))})
    outcome = _ingest_news(fresh_db, live_settings, fake_embedder)
    assert _pages_requested(route) == [1, 2]
    assert (outcome.fetched, outcome.new, outcome.note) == (4, 4, None)


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_ingest_without_a_lookback_takes_the_first_page_only(fresh_db, live_settings, fake_embedder, respx_mock):
    now = datetime.now(UTC)
    route = _mock_paged_feed(respx_mock, {1: _rss(("a", now - timedelta(hours=1))),
                                          2: _rss(("b", now - timedelta(days=2)))})
    outcome = _ingest_news(fresh_db, live_settings, fake_embedder, lookback=None)
    assert _pages_requested(route) == [1]
    assert (outcome.fetched, outcome.note) == (1, None)


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_paging_stops_when_a_page_adds_nothing_and_notes_the_shortfall(fresh_db, live_settings, fake_embedder,
                                                                      respx_mock):
    # A feed that ignores the page parameter returns its first page again.
    now = datetime.now(UTC)
    same = _rss(("a", now - timedelta(hours=1)), ("b", now - timedelta(hours=5)))
    route = _mock_paged_feed(respx_mock, {1: same, 2: same, 3: same})
    outcome = _ingest_news(fresh_db, live_settings, fake_embedder, lookback=timedelta(days=3))
    assert _pages_requested(route) == [1, 2]
    assert outcome.fetched == 2 and outcome.error is None and outcome.note is not None


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_paging_stops_at_the_page_limit_and_notes_the_shortfall(fresh_db, live_settings, fake_embedder, respx_mock):
    now = datetime.now(UTC)
    route = _mock_paged_feed(respx_mock, {n: _rss((f"p{n}", now - timedelta(hours=n)))
                                          for n in range(1, MAX_PAGES_PER_SOURCE + 2)})
    outcome = _ingest_news(fresh_db, live_settings, fake_embedder, lookback=timedelta(days=3))
    assert _pages_requested(route) == list(range(1, MAX_PAGES_PER_SOURCE + 1))
    assert outcome.fetched == MAX_PAGES_PER_SOURCE and outcome.note is not None


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_a_failed_later_page_keeps_the_pages_already_fetched(fresh_db, live_settings, fake_embedder, respx_mock):
    now = datetime.now(UTC)
    _mock_paged_feed(respx_mock, {1: _rss(("a", now - timedelta(hours=1)), ("b", now - timedelta(days=1)))})
    outcome = _ingest_news(fresh_db, live_settings, fake_embedder, lookback=timedelta(days=3))
    assert (outcome.fetched, outcome.new, outcome.error) == (2, 2, None)
    assert "page 2" in outcome.note


def _gazette_page(code: str, page: int, published: datetime) -> bytes:
    """A minimal Gazette feed page holding one notice."""
    return (f'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            f"<id>notice-{code}-{page}</id><title>NOTICE {code} {page} LIMITED</title>"
            f'<link href="https://gazette.test/{code}/{page}"/><published>{published.isoformat()}</published>'
            f"<summary>A notice.</summary></entry></feed>").encode()


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_gazette_pages_back_with_its_own_page_parameter(fresh_db, live_settings, fake_embedder, respx_mock):
    now = datetime.now(UTC)
    ages = {1: timedelta(hours=12), 2: timedelta(days=4)}

    def respond(request: httpx.Request) -> httpx.Response:
        code, page = request.url.params["noticetypes"], int(request.url.params.get("results-page", "1"))
        return httpx.Response(200, content=_gazette_page(code, page, now - ages.get(page, timedelta(days=30))))

    route = respx_mock.get(url__startswith="https://www.thegazette.co.uk/").mock(side_effect=respond)
    report = ingest(live_settings, {SourceKind.GAZETTE}, conn=fresh_db, embedder=fake_embedder,
                    http=_no_sleep_client(live_settings), lookback=timedelta(days=3))
    requested = sorted((c.request.url.params["noticetypes"], c.request.url.params.get("results-page"))
                       for c in route.calls)
    assert requested == sorted((code, page) for code in GAZETTE_NOTICE_TYPES for page in ("1", "2"))
    assert all(o.fetched == 2 and o.note is None for o in report.per_source.values())


@pytest.mark.db
@pytest.mark.respx(assert_all_called=False)
def test_boe_series_are_stored_and_capture_writes_snapshots(fresh_db, live_settings, fake_embedder, respx_mock,
                                                            tmp_path):
    respx_mock.get(url__startswith="https://www.bankofengland.co.uk/").mock(
        return_value=httpx.Response(200, text=TEST_DATA.joinpath("boe.csv").read_text()))
    _mock_feeds(respx_mock, Bisnow_London=httpx.Response(
        200, content=TEST_DATA.joinpath("feeds/bisnow.xml").read_bytes()))
    ingest(live_settings, {SourceKind.BOE, SourceKind.NEWS}, conn=fresh_db, embedder=fake_embedder,
           http=_no_sleep_client(live_settings), capture_dir=tmp_path)
    assert fresh_db.execute("SELECT count(*) FROM macro_series").fetchone()[0] == 6
    lines = [json.loads(line) for line in tmp_path.joinpath("news_snapshot.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(set(item) == {"source_kind", "publisher", "title", "summary", "url",
                                                 "published_at", "meta"} for item in lines)
    assert tmp_path.joinpath("boe_snapshot.csv").read_text().startswith("series,day,value\n")

from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest

from london_office_monitor.ingest.boe import MacroPoint, parse_boe_csv
from london_office_monitor.ingest.feeds import DocumentIn, FeedParseError, parse_feed
from london_office_monitor.ingest.gazette import parse_gazette, property_owner_reason
from london_office_monitor.ingest.http import FetchError, HttpClient
from tests.conftest import TEST_DATA

FEEDS_DIR = TEST_DATA / "feeds"
NOW = datetime(2026, 10, 23, 10, 20, tzinfo=UTC)


def _gazette_doc(title: str, summary: str, **meta) -> DocumentIn:
    return DocumentIn(source_kind="gazette", publisher="The Gazette", title=title, summary=summary, url="u",
                      published_at=NOW, meta=meta)


def test_parse_feed_strips_html_and_bounds_summary():
    docs = parse_feed(FEEDS_DIR.joinpath("cityam.xml").read_bytes(), "City AM")
    assert len(docs) == 3 and all(len(d.summary) <= 1000 and "<" not in d.summary for d in docs)
    assert docs[0].summary == "Barclays has bought its Canary Wharf headquarters, One Churchill Place, for £750m."
    assert docs[0].published_at == datetime(2026, 10, 20, 9, 15, tzinfo=UTC)


def test_parse_feed_never_keeps_the_article_body():
    docs = parse_feed(FEEDS_DIR.joinpath("cityam.xml").read_bytes(), "City AM")
    assert all("FULL ARTICLE BODY" not in d.summary for d in docs)


def test_parse_feed_refuses_html_page():
    with pytest.raises(FeedParseError):
        parse_feed(b"<!DOCTYPE html><html>captive portal</html>", "City AM")


def test_parse_feed_refuses_garbage():
    with pytest.raises(FeedParseError):
        parse_feed(b"\x00\x01 not xml at all", "City AM")


def test_parse_gazette_reads_location_and_london_time():
    docs = parse_gazette(TEST_DATA.joinpath("gazette_2410.xml").read_bytes(), "2410")
    assert [d.title for d in docs] == ["EXAMPLE CITY PROPERTIES LIMITED", "SMITH BAKERY LTD", "NORTHERN ESTATES LIMITED"]
    assert docs[0].url == "https://www.thegazette.co.uk/notice/9000001"
    assert docs[0].published_at == datetime(2026, 10, 23, 10, 20, 14, tzinfo=UTC)  # 11:20:14 BST
    assert (docs[0].meta["lat"], docs[0].meta["long"]) == (51.51794, -0.090151)
    assert "London" not in docs[2].summary  # the edition name ("London Gazette") is not notice text


def test_gazette_filter_keeps_property_company_in_ec2():
    doc = _gazette_doc("EXAMPLE PROPERTIES LIMITED", "Registered office: 1 Example St, London EC2V 7HH")
    assert "EC2V" in property_owner_reason(doc)


def test_gazette_filter_keeps_property_company_located_in_central_london():
    doc = _gazette_doc("EXAMPLE CITY PROPERTIES LIMITED", "In the High Court of Justice…", lat=51.51794,
                       long=-0.090151)
    assert "Central London" in property_owner_reason(doc)


def test_gazette_filter_drops_non_property_company():
    assert property_owner_reason(_gazette_doc("SMITH BAKERY LTD", "Registered office: Leeds LS1 4AP")) is None


def test_gazette_filter_drops_property_company_outside_london():
    doc = _gazette_doc("NORTHERN ESTATES LIMITED", "Manchester District Registry", lat=53.48095, long=-2.237431)
    assert property_owner_reason(doc) is None


def test_parse_boe_csv():
    points = parse_boe_csv(TEST_DATA.joinpath("boe.csv").read_text())
    assert MacroPoint("IUDBEDR", date(2026, 10, 23), Decimal("3.75")) in points
    assert MacroPoint("IUDMNPY", date(2026, 10, 21), Decimal("5.3993")) in points and len(points) == 6


def test_retry_then_success(respx_mock, settings):
    route = respx_mock.get("https://x.test/feed").mock(side_effect=[httpx.Response(503), httpx.Response(503),
                                                                    httpx.Response(200, content=b"ok")])
    assert HttpClient(settings, sleep=lambda s: None).get("https://x.test/feed") == b"ok" and route.call_count == 3


def test_retries_stop_after_three_attempts(respx_mock, settings):
    route = respx_mock.get("https://x.test/feed?key=SECRET").mock(return_value=httpx.Response(503))
    with pytest.raises(FetchError) as e:
        HttpClient(settings, sleep=lambda s: None).get("https://x.test/feed?key=SECRET")
    assert route.call_count == 3 and "SECRET" not in str(e.value) and "x.test" in str(e.value)


def test_client_error_is_not_retried(respx_mock, settings):
    route = respx_mock.get("https://x.test/feed").mock(return_value=httpx.Response(404))
    with pytest.raises(FetchError):
        HttpClient(settings, sleep=lambda s: None).get("https://x.test/feed")
    assert route.call_count == 1


def test_retry_after_is_honoured_and_capped(respx_mock, settings):
    waits: list[float] = []
    respx_mock.get("https://x.test/feed").mock(side_effect=[httpx.Response(429, headers={"Retry-After": "300"}),
                                                            httpx.Response(200, content=b"ok")])
    HttpClient(settings, sleep=waits.append).get("https://x.test/feed")
    assert waits == [60.0]

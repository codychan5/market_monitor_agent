"""Property news from RSS feeds. Only what a feed publishes is kept (title, summary, link, date), never the article."""
import calendar
import html
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser

# Feeds verified reachable from Hong Kong on 2026-10-23 (docs/research/2026-10-23-data-sources.md). URLs are fixed
# here: the model never supplies a URL to fetch.
FEEDS: dict[str, str] = {
    "City AM": "https://www.cityam.com/category/property/feed/",
    "Property Week": "https://www.propertyweek.com/news/feed",
    "Bisnow London": "https://www.bisnow.com/rss-feed/london",
    "Property Industry Eye": "https://propertyindustryeye.com/feed/",
    "FT Property": "https://www.ft.com/property?format=rss",
    "Construction Enquirer": "https://www.constructionenquirer.com/feed/",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
}
# Feeds that page (Property Week, Property Industry Eye and Construction Enquirer, checked live on 25 Oct 2026) serve
# older items at ?paged=2, 3 and so on; a feed that ignores the parameter returns its first page again.
PAGE_PARAMETER = "paged"
MAX_SUMMARY_CHARS = 1000
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
HTML_START = re.compile(rb"^\s*(<!doctype html|<html)", re.IGNORECASE)


class FeedParseError(Exception):
    """The bytes are not a usable feed: an HTML page (a captive portal or error page) or malformed XML."""


@dataclass(frozen=True)
class DocumentIn:
    source_kind: str
    publisher: str
    title: str
    summary: str
    url: str
    published_at: datetime
    subtitle: str | None = None
    text_format: Literal["plain", "markdown"] = "plain"
    synthetic: bool = False
    fixture_role: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def feed_page_url(url: str, page: int) -> str:
    """The URL of a feed's given page of older items; page 1 is the feed's own URL."""
    if page == 1:
        return url
    parts = urlsplit(url)
    return urlunsplit(parts._replace(query=urlencode([*parse_qsl(parts.query), (PAGE_PARAMETER, str(page))])))


def strip_html(text: str) -> str:
    return SPACE.sub(" ", html.unescape(TAG.sub(" ", text or ""))).strip()


def bounded(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """At most `limit` characters, cut at a word boundary with an ellipsis when it had to be shortened."""
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0]
    return cut + "…"


def _published(entry: Any) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC) if parsed else None


def parse_feed(xml: bytes, publisher: str) -> list[DocumentIn]:
    """One document per feed item that has a title, link and date. The item's description becomes the summary;
    its full content (content:encoded) is ignored."""
    if HTML_START.match(xml[:200]):
        raise FeedParseError(f"{publisher} returned an HTML page, not a feed")
    feed = feedparser.parse(xml)
    if feed.bozo and not feed.entries:
        raise FeedParseError(f"{publisher} returned a malformed feed ({type(feed.bozo_exception).__name__})")
    docs = []
    for entry in feed.entries:
        published = _published(entry)
        title = strip_html(entry.get("title", ""))
        link = entry.get("link", "")
        if not (title and link and published):
            continue
        docs.append(DocumentIn(source_kind="news", publisher=publisher, title=title,
                               summary=bounded(strip_html(entry.get("summary", ""))), url=link,
                               published_at=published))
    return docs

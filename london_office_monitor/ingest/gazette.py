"""Insolvency notices from The Gazette, kept only when they look like a London office owner in distress."""
import re
from datetime import UTC, datetime

import feedparser

from ..clock import LONDON
from .feeds import DocumentIn, FeedParseError, bounded, strip_html

# Administration, administrative receivers, winding-up petitions, and two liquidator notice types.
GAZETTE_NOTICE_TYPES = ("2410", "2421", "2450", "2432", "2443")
GAZETTE_URL = ("https://www.thegazette.co.uk/all-notices/notice/data.feed?noticetypes={code}"
               "&results-page-size={page_size}&results-page={page}")
# Notices per request: the most the feed serves (larger sizes still return 100, tested live on 25 Oct 2026). Its
# default of 10 covered about 3 hours of winding-up petitions; 100 cover about 3 days of those, but under a day of the
# busiest type, 2443.
GAZETTE_PAGE_SIZE = 100
GAZETTE_PUBLISHER = "The Gazette"
PROPERTY_NAME_PATTERN = re.compile(
    r"\b(PROPERT(Y|IES)|ESTATES?|REAL ESTATE|INVESTMENTS?|DEVELOPMENTS?|LAND|REIT|ASSETS?)\b", re.IGNORECASE)
LONDON_OFFICE_POSTCODE = re.compile(r"\b(EC[1-4][A-Z]?|WC[12][A-Z]?|W1[A-Z]?|SE1|E1W?|E14|N1C?|NW1)\s?\d[A-Z]{2}\b")
LONDON_WORD = re.compile(r"\bLondon\b")
# Gazette summaries are cut off before the registered address, but each notice carries the registered office's
# location; this box spans the City, West End, Midtown, King's Cross, Southbank and Canary Wharf.
CENTRAL_LONDON_BOX = {"lat_min": 51.48, "lat_max": 51.55, "long_min": -0.22, "long_max": 0.03}


def gazette_url(notice_type: str, page: int = 1) -> str:
    """The URL of one page of a notice type, newest notices first."""
    return GAZETTE_URL.format(code=notice_type, page_size=GAZETTE_PAGE_SIZE, page=page)


def _published(entry) -> datetime:
    """The Gazette's <published> has no offset and is London time; fall back to <updated>, which has one."""
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        raise FeedParseError("Gazette notice without a date")
    moment = datetime.fromisoformat(raw)
    return (moment if moment.tzinfo else moment.replace(tzinfo=LONDON)).astimezone(UTC)


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_gazette(xml: bytes, notice_type: str) -> list[DocumentIn]:
    """Every notice in the feed, unfiltered. The edition name ("London Gazette") is not part of the notice text."""
    feed = feedparser.parse(xml)
    if feed.bozo and not feed.entries:
        raise FeedParseError(f"Gazette {notice_type} returned a malformed feed")
    docs = []
    for entry in feed.entries:
        tags = entry.get("tags") or []
        docs.append(DocumentIn(
            source_kind="gazette", publisher=GAZETTE_PUBLISHER, title=strip_html(entry.get("title", "")),
            summary=bounded(strip_html(entry.get("summary", ""))), url=entry.get("link", ""),
            published_at=_published(entry),
            meta={"notice_type": notice_type, "notice_id": entry.get("id", ""),
                  "category": tags[0].get("term") if tags else None,
                  "lat": _float(entry.get("geo_lat")), "long": _float(entry.get("geo_long"))}))
    return docs


def _in_central_london(lat: float | None, long: float | None) -> bool:
    box = CENTRAL_LONDON_BOX
    return (lat is not None and long is not None and box["lat_min"] <= lat <= box["lat_max"]
            and box["long_min"] <= long <= box["long_max"])


def property_owner_reason(doc: DocumentIn) -> str | None:
    """Why a notice concerns a possible London office owner, or None. Both conditions must hold: a property
    company's name, and a London office postcode, a London address, or a registered office in Central London."""
    name = PROPERTY_NAME_PATTERN.search(doc.title)
    if not name:
        return None
    postcode = LONDON_OFFICE_POSTCODE.search(doc.summary)
    if postcode:
        return f"property company name ({name.group(0)}) with London office postcode {postcode.group(0)}"
    if LONDON_WORD.search(doc.summary):
        return f"property company name ({name.group(0)}) with a London address"
    lat, long = doc.meta.get("lat"), doc.meta.get("long")
    if _in_central_london(lat, long):
        return f"property company name ({name.group(0)}) registered in Central London ({lat}, {long})"
    return None

"""Refresh the live sources. Each source is fetched at most once per interval, in its own transaction, and one
source failing never stops the others."""
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path

import psycopg

from .. import clock
from ..clock import LONDON
from ..config import Settings
from ..embed import Embedder, EmbeddingError
from ..redact import redact
from ..vocab import SourceKind
from .boe import BOE_PUBLISHER, BoeParseError, MacroPoint, boe_url, parse_boe_csv
from .feeds import FEEDS, DocumentIn, FeedParseError, feed_page_url, parse_feed
from .gazette import GAZETTE_NOTICE_TYPES, gazette_url, parse_gazette, property_owner_reason
from .http import FetchError, HttpClient
from .store import store_documents

DEMO_SKIP = "demo mode: using the stored snapshot from {date}"
# How far back a Bank of England refresh asks for: enough history for the brief's rate commentary.
BOE_HISTORY_START = date(2025, 1, 1)
NEWS_SNAPSHOT = "news_snapshot.jsonl"
GAZETTE_SNAPSHOT = "gazette_snapshot.jsonl"
BOE_SNAPSHOT = "boe_snapshot.csv"
SOURCE_ERRORS = (FetchError, FeedParseError, BoeParseError, EmbeddingError, psycopg.Error)
# Feeds list only their newest items, so a run pages back until it reaches items older than this: the day between
# scheduled runs (ops/cron.example) plus two missed runs.
PAGING_LOOKBACK = timedelta(days=3)
# A bound on requests per source per run. The busiest Gazette notice type (2443, about 110 notices a day on
# 25 Oct 2026) needs 4 pages of 100 to cover 3 days.
MAX_PAGES_PER_SOURCE = 5


@dataclass
class SourceOutcome:
    fetched: int = 0
    new: int = 0
    error: str | None = None
    skipped: str | None = None
    # Why the items stored may not reach back the whole lookback, or None.
    note: str | None = None


@dataclass
class IngestReport:
    per_source: dict[str, SourceOutcome]


@dataclass(frozen=True)
class _Source:
    name: str
    kind: SourceKind
    url: str
    notice_type: str | None = None


def planned_sources(kinds: set[SourceKind]) -> list[_Source]:
    sources = []
    if SourceKind.NEWS in kinds:
        sources += [_Source(publisher, SourceKind.NEWS, url) for publisher, url in FEEDS.items()]
    if SourceKind.GAZETTE in kinds:
        sources += [_Source(f"Gazette {code}", SourceKind.GAZETTE, gazette_url(code), code)
                    for code in GAZETTE_NOTICE_TYPES]
    if SourceKind.BOE in kinds:
        sources.append(_Source(BOE_PUBLISHER, SourceKind.BOE, boe_url(BOE_HISTORY_START)))
    return sources


@dataclass
class _Captured:
    news: list[DocumentIn]
    gazette: list[DocumentIn]
    boe: list[MacroPoint]


@dataclass(frozen=True)
class _Fetched:
    items: list[DocumentIn]
    note: str | None = None


def _page_url(source: _Source, page: int) -> str:
    if source.kind is SourceKind.GAZETTE:
        return gazette_url(source.notice_type, page)
    return feed_page_url(source.url, page)


def _reach(items: list[DocumentIn]) -> str:
    oldest = min(d.published_at for d in items)
    return f"items reach back to {oldest.astimezone(LONDON):%d %b %H:%M}"


def _fetch_pages(source: _Source, parse: Callable[[bytes], list[DocumentIn]], *, http: HttpClient,
                 oldest_wanted: datetime | None) -> _Fetched:
    """The source's first page, then older pages until one reaches back to oldest_wanted (None: the first page only).
    Paging also stops when a page adds nothing new (the feed's last page, or a feed that ignores paging), at
    MAX_PAGES_PER_SOURCE, or when a later page fails; the note then says how far back the items reach. A failed first
    page raises, failing the source."""
    items: list[DocumentIn] = []
    seen: set[tuple[str, str]] = set()
    for page in range(1, MAX_PAGES_PER_SOURCE + 1):
        try:
            batch = parse(http.get(_page_url(source, page)))
        except (FetchError, FeedParseError) as e:
            if page == 1:
                raise
            return _Fetched(items, f"page {page} failed ({e}); {_reach(items)}")
        fresh = []
        for doc in batch:
            if (doc.url, doc.title) not in seen:
                seen.add((doc.url, doc.title))
                fresh.append(doc)
        items += fresh
        if oldest_wanted is None:
            return _Fetched(items)
        if not fresh:
            return _Fetched(items, f"the feed offers nothing older; {_reach(items)}" if items else None)
        if min(d.published_at for d in fresh) <= oldest_wanted:
            return _Fetched(items)
    return _Fetched(items, f"stopped at the {MAX_PAGES_PER_SOURCE}-page limit; {_reach(items)}")


def _ingest_one(source: _Source, *, conn, embedder, http, settings, captured: _Captured,
                oldest_wanted: datetime | None) -> SourceOutcome:
    if source.kind is SourceKind.BOE:
        points = parse_boe_csv(http.get(source.url).decode("utf-8"))
        captured.boe += points
        inserted = sum(conn.execute("INSERT INTO macro_series (series, day, value) VALUES (%s, %s, %s)"
                                    " ON CONFLICT DO NOTHING", (p.series, p.day, p.value)).rowcount for p in points)
        return SourceOutcome(fetched=len(points), new=inserted)
    if source.kind is SourceKind.NEWS:
        fetched = _fetch_pages(source, partial(parse_feed, publisher=source.name), http=http,
                               oldest_wanted=oldest_wanted)
        captured.news += fetched.items
        stored = store_documents(conn, fetched.items, embedder, settings)
    else:
        fetched = _fetch_pages(source, partial(parse_gazette, notice_type=source.notice_type), http=http,
                               oldest_wanted=oldest_wanted)
        kept = [replace(n, meta={**n.meta, "filter_reason": reason})
                for n in fetched.items if (reason := property_owner_reason(n))]
        captured.gazette += kept
        stored = store_documents(conn, kept, embedder, settings)
    return SourceOutcome(fetched=len(fetched.items), new=stored.new,
                         note=redact(fetched.note, settings) if fetched.note else None)


def _skip_reason(conn, name: str, started: datetime, interval: timedelta) -> str | None:
    last = conn.execute("SELECT max(started_at) FROM ingest_runs WHERE source_name = %s", (name,)).fetchone()[0]
    if last is None or started - last >= interval:
        return None
    return (f"refreshed at {last.astimezone(LONDON):%H:%M}; next refresh after "
            f"{(last + interval).astimezone(LONDON):%H:%M}")


def _snapshot_line(doc: DocumentIn) -> str:
    return json.dumps({"source_kind": doc.source_kind, "publisher": doc.publisher, "title": doc.title,
                       "summary": doc.summary, "url": doc.url, "published_at": doc.published_at.isoformat(),
                       "meta": doc.meta}, ensure_ascii=False)


def _write_capture(capture_dir: Path, captured: _Captured) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    (capture_dir / NEWS_SNAPSHOT).write_text("".join(_snapshot_line(d) + "\n" for d in captured.news))
    (capture_dir / GAZETTE_SNAPSHOT).write_text("".join(_snapshot_line(d) + "\n" for d in captured.gazette))
    with (capture_dir / BOE_SNAPSHOT).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["series", "day", "value"])
        writer.writerows([p.series, p.day.isoformat(), str(p.value)] for p in captured.boe)


def ingest(settings: Settings, kinds: set[SourceKind], *, conn: psycopg.Connection, embedder: Embedder,
           http: HttpClient, capture_dir: Path | None = None,
           lookback: timedelta | None = PAGING_LOOKBACK) -> IngestReport:
    """Fetch, parse and store each source of the given kinds, paging back through each feed until its items are older
    than lookback (None: each feed's first page only). Needs an autocommit connection: every source runs in its own
    transaction, and its ingest_runs row is written whether it succeeded or not."""
    sources = planned_sources(kinds)
    if not settings.live:
        message = DEMO_SKIP.format(date=settings.demo_date.isoformat())
        return IngestReport({s.name: SourceOutcome(skipped=message) for s in sources})
    if not conn.autocommit:
        raise ValueError("ingest needs an autocommit connection: each source commits on its own")
    oldest_wanted = clock.now(settings) - lookback if lookback is not None else None
    interval = timedelta(minutes=settings.refresh_min_interval_minutes)
    captured = _Captured(news=[], gazette=[], boe=[])
    report: dict[str, SourceOutcome] = {}
    for source in sources:
        started = clock.now(settings)
        reason = _skip_reason(conn, source.name, started, interval)
        if reason:
            report[source.name] = SourceOutcome(skipped=reason)
            continue
        try:
            with conn.transaction():
                outcome = _ingest_one(source, conn=conn, embedder=embedder, http=http, settings=settings,
                                      captured=captured, oldest_wanted=oldest_wanted)
        except SOURCE_ERRORS as e:
            outcome = SourceOutcome(error=redact(f"{type(e).__name__}: {e}", settings))
        conn.execute("INSERT INTO ingest_runs (source_name, started_at, finished_at, fetched, new, error)"
                     " VALUES (%s, %s, %s, %s, %s, %s)",
                     (source.name, started, clock.now(settings), outcome.fetched, outcome.new, outcome.error))
        report[source.name] = outcome
    if capture_dir is not None:
        _write_capture(capture_dir, captured)
    return IngestReport(report)

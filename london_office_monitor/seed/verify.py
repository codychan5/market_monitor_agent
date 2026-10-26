"""Check each automatically verifiable seed figure against its source page, respecting robots.txt."""
import argparse
import html
import io
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from pypdf import PdfReader

from .load import SeedBundle, SeedError, read_seed
from .models import BrokerMetricRow

USER_AGENT = "london-office-monitor/0.1 (seed verification)"
FETCH_TIMEOUT_SECONDS = 30
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
# Statuses that mean the figure was not confirmed: the script exits 1 when any row has one.
FAILING_STATUSES = frozenset({"not found", "fetch failed"})


@dataclass(frozen=True)
class RowCheck:
    status: str  # ok, not found, robots-disallowed, fetch failed, manual
    row: BrokerMetricRow
    url: str
    detail: str = ""


def value_variants(value: str | Decimal) -> list[str]:
    """The spellings a published figure may take: as written, without trailing zeros, and with a pound sign."""
    text = str(value)
    variants = [text]
    if "." in text:
        trimmed = text.rstrip("0").rstrip(".")
        if trimmed != text:
            variants.append(trimmed)
    return variants + [f"£{v}" for v in variants]


def page_text(content: bytes) -> str:
    """Plain text of a PDF or an HTML page, with entities decoded and whitespace collapsed."""
    if content.startswith(b"%PDF"):
        text = " ".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
    else:
        text = html.unescape(TAG.sub(" ", content.decode("utf-8", errors="replace")))
    return SPACE.sub(" ", text)


def contains_figure(text: str, variant: str) -> bool:
    """True when the figure appears as a whole number, not as part of a longer one (6.3 does not match 16.3)."""
    return re.search(rf"(?<![\d.]){re.escape(variant)}(?![\d])", text) is not None


class _Fetcher:
    def __init__(self, client: httpx.Client):
        self.client = client
        self.robots: dict[str, RobotFileParser] = {}
        self.pages: dict[str, bytes | Exception] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robots:
            parser = RobotFileParser()
            try:
                response = self.client.get(f"{origin}/robots.txt", headers={"User-Agent": USER_AGENT},
                                           timeout=FETCH_TIMEOUT_SECONDS)
                parser.parse(response.text.splitlines() if response.status_code == 200 else [])
            except httpx.HTTPError:
                parser.parse([])  # no readable robots.txt: the standard treats the site as open
            self.robots[origin] = parser
        return self.robots[origin].can_fetch(USER_AGENT, url)

    def get(self, url: str) -> bytes:
        if url not in self.pages:
            try:
                response = self.client.get(url, headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT_SECONDS,
                                           follow_redirects=True)
                response.raise_for_status()
                self.pages[url] = response.content
            except httpx.HTTPError as e:
                self.pages[url] = e
        result = self.pages[url]
        if isinstance(result, Exception):
            raise result
        return result


def verify_rows(bundle: SeedBundle, client: httpx.Client) -> list[RowCheck]:
    """One check per metric row, in file order. Each source page is fetched at most once."""
    urls = {s.source_key: str(s.url) for s in bundle.sources}
    fetcher = _Fetcher(client)
    checks = []
    for row in bundle.metrics:
        url = urls[row.source_key]
        if row.check_method == "manual":
            checks.append(RowCheck("manual", row, url))
        elif not fetcher.allowed(url):
            checks.append(RowCheck("robots-disallowed", row, url))
        else:
            try:
                text = page_text(fetcher.get(url))
            except httpx.HTTPError as e:
                checks.append(RowCheck("fetch failed", row, url, type(e).__name__))
                continue
            found = any(contains_figure(text, v) for v in value_variants(row.value))
            checks.append(RowCheck("ok" if found else "not found", row, url))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check seed figures against their source pages.")
    parser.add_argument("seed_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        bundle = read_seed(args.seed_dir)
    except SeedError as e:
        print(e, file=sys.stderr)
        return 1
    with httpx.Client() as client:
        checks = verify_rows(bundle, client)
    for c in checks:
        r = c.row
        print(f"{c.status:18} {r.broker.value:20} {r.metric.value:18} {r.submarket.value:22} {r.period:8} "
              f"{r.value!s:>8}  {c.url} {c.detail}")
    manual = [c for c in checks if c.status in ("manual", "robots-disallowed")]
    if manual:
        print(f"\n{len(manual)} rows need a person to check them against the source (manual or robots-disallowed).")
    failing = [c for c in checks if c.status in FAILING_STATUSES]
    print(f"{len(checks) - len(failing)} of {len(checks)} rows did not fail verification.")
    return 1 if failing else 0

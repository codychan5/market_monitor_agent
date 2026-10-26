"""Read and validate a seed directory, then load it into Postgres idempotently."""
import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import yaml
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..clock import LONDON
from ..config import Settings
from ..embed import Embedder
from ..ingest.feeds import DocumentIn
from ..ingest.store import StoreResult, store_documents
from .models import AreaMapRow, BrokerMetricRow, DefinitionRow, MandateDoc, PortfolioAssetRow, SourceRow

SOURCES_FILE = "sources.csv"
METRICS_FILE = "broker_metrics.csv"
DEFINITIONS_FILE = "metric_definitions.csv"
AREA_MAP_FILE = "broker_area_map.csv"
PORTFOLIO_FILE = "portfolio.yaml"
MANDATE_FILE = "mandate.yaml"
# The frozen demo and evaluation corpus (Task 6): live captures plus six labelled synthetic test items.
FIXTURE_DOCUMENT_FILES = ("news_snapshot.jsonl", "gazette_snapshot.jsonl", "synthetic.jsonl")
FIXTURE_MACRO_FILE = "boe_snapshot.csv"
# CSV line numbers count the header as line 1, so the first data row is line 2: what an editor shows.
FIRST_DATA_LINE = 2


class SeedError(Exception):
    """The seed directory is missing a file or holds invalid rows; the message lists every problem."""


@dataclass(frozen=True)
class SeedBundle:
    sources: tuple[SourceRow, ...]
    metrics: tuple[BrokerMetricRow, ...]
    definitions: tuple[DefinitionRow, ...]
    area_maps: tuple[AreaMapRow, ...]
    assets: tuple[PortfolioAssetRow, ...]
    mandate: MandateDoc


@dataclass(frozen=True)
class SeedCounts:
    sources: int
    metrics: int
    definitions: int
    area_maps: int
    assets: int
    mandate_statements: int


def _problems(prefix: str, error: ValidationError) -> list[str]:
    return [f"{prefix}: {'.'.join(str(p) for p in e['loc']) or 'row'}: {e['msg']}"
            for e in error.errors(include_url=False)]


def _read_csv(path: Path, model: type[BaseModel], errors: list[str]) -> list[tuple[int, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=FIRST_DATA_LINE):
            if None in raw:
                errors.append(f"{path.name}:{line}: row has more values than the header has columns")
                continue
            try:
                rows.append((line, model(**{k: v.strip() for k, v in raw.items()})))
            except ValidationError as e:
                errors.extend(_problems(f"{path.name}:{line}", e))
    return rows


def _read_yaml(path: Path, errors: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        errors.append(f"{path.name}: not valid YAML ({e})")
        return None


def read_seed(seed_dir: Path) -> SeedBundle:
    """Validate every file and cross-reference; raise SeedError listing all problems, not just the first."""
    missing = [name for name in (SOURCES_FILE, METRICS_FILE, DEFINITIONS_FILE, AREA_MAP_FILE, PORTFOLIO_FILE,
                                 MANDATE_FILE) if not (seed_dir / name).is_file()]
    if missing:
        raise SeedError(f"seed directory {seed_dir} is missing: {', '.join(missing)}")
    errors: list[str] = []
    sources = _read_csv(seed_dir / SOURCES_FILE, SourceRow, errors)
    metrics = _read_csv(seed_dir / METRICS_FILE, BrokerMetricRow, errors)
    definitions = _read_csv(seed_dir / DEFINITIONS_FILE, DefinitionRow, errors)
    area_maps = _read_csv(seed_dir / AREA_MAP_FILE, AreaMapRow, errors)

    assets: list[PortfolioAssetRow] = []
    portfolio = _read_yaml(seed_dir / PORTFOLIO_FILE, errors)
    for i, item in enumerate((portfolio or {}).get("assets") or []):
        try:
            assets.append(PortfolioAssetRow(**item))
        except (ValidationError, TypeError) as e:
            errors.extend(_problems(f"{PORTFOLIO_FILE}: assets[{i}]", e) if isinstance(e, ValidationError)
                          else [f"{PORTFOLIO_FILE}: assets[{i}]: must be a mapping"])
    mandate = None
    try:
        mandate = MandateDoc(**(_read_yaml(seed_dir / MANDATE_FILE, errors) or {}))
    except ValidationError as e:
        errors.extend(_problems(MANDATE_FILE, e))

    source_keys = {row.source_key for _, row in sources}
    if len(source_keys) != len(sources):
        errors.append(f"{SOURCES_FILE}: source_key values must be unique")
    for file, rows in ((METRICS_FILE, metrics), (DEFINITIONS_FILE, definitions)):
        errors.extend(f"{file}:{line}: source_key {row.source_key!r} is not in {SOURCES_FILE}"
                      for line, row in rows if row.source_key not in source_keys)
    errors.extend(f"{PORTFOLIO_FILE}: {asset.name}: source_key {key!r} is not in {SOURCES_FILE}"
                  for asset in assets for key in asset.source_keys if key not in source_keys)
    mapped = {(row.broker, row.broker_label): row.submarket for _, row in area_maps}
    for line, row in metrics:
        submarket = mapped.get((row.broker, row.broker_label))
        if submarket is None:
            errors.append(f"{METRICS_FILE}:{line}: no area map row for ({row.broker.value}, {row.broker_label})")
        elif submarket != row.submarket:
            errors.append(f"{METRICS_FILE}:{line}: {row.broker.value} label {row.broker_label!r} maps to "
                          f"{submarket.value} in the area map, not {row.submarket.value}")
    seen: dict[tuple, int] = {}
    for line, row in metrics:
        key = (row.broker, row.metric, row.submarket, row.period)
        if key in seen:
            errors.append(f"{METRICS_FILE}:{line}: duplicates line {seen[key]} (same broker, metric, submarket, "
                          f"period)")
        seen.setdefault(key, line)

    if errors or mandate is None:
        raise SeedError("Seed data is invalid:\n" + "\n".join(errors or ["mandate.yaml is empty"]))
    return SeedBundle(sources=tuple(r for _, r in sources), metrics=tuple(r for _, r in metrics),
                      definitions=tuple(r for _, r in definitions), area_maps=tuple(r for _, r in area_maps),
                      assets=tuple(assets), mandate=mandate)


def _vector(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def asset_text(asset: PortfolioAssetRow) -> str:
    return f"{asset.name}, {asset.address}. {asset.notes}"


def load_seed(conn: psycopg.Connection, bundle: SeedBundle, embedder: Embedder) -> SeedCounts:
    """Upsert the bundle in one transaction and delete rows the seed no longer has. Ids of unchanged figures stay
    stable, so a citation such as [F12] keeps pointing at the same figure after a reload."""
    asset_vectors = embedder.embed_documents([asset_text(a) for a in bundle.assets])
    statement_vectors = embedder.embed_documents(list(bundle.mandate.statements))
    with conn.transaction():
        source_ids = {}
        for s in bundle.sources:
            source_ids[s.source_key] = conn.execute(
                "INSERT INTO sources (source_key, publisher, title, url, published, licence_note)"
                " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (source_key) DO UPDATE SET publisher = EXCLUDED.publisher,"
                " title = EXCLUDED.title, url = EXCLUDED.url, published = EXCLUDED.published,"
                " licence_note = EXCLUDED.licence_note RETURNING id",
                (s.source_key, s.publisher, s.title, str(s.url), s.published, s.licence_note)).fetchone()[0]

        for m in bundle.metrics:
            conn.execute(
                "INSERT INTO broker_metrics (broker, metric, submarket, broker_label, period, value, unit, measure_note,"
                " source_id, locator, checked_at, check_method) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (broker, metric, submarket, period) DO UPDATE SET broker_label = EXCLUDED.broker_label,"
                " value = EXCLUDED.value, unit = EXCLUDED.unit, measure_note = EXCLUDED.measure_note,"
                " source_id = EXCLUDED.source_id, locator = EXCLUDED.locator, checked_at = EXCLUDED.checked_at,"
                " check_method = EXCLUDED.check_method",
                (m.broker.value, m.metric.value, m.submarket.value, m.broker_label, m.period, m.value, m.unit,
                 m.measure_note, source_ids[m.source_key], m.locator, m.checked_at, m.check_method))
        conn.execute(
            "DELETE FROM broker_metrics WHERE (broker, metric, submarket, period) NOT IN"
            " (SELECT * FROM unnest(%s::text[], %s::text[], %s::text[], %s::text[]))",
            ([m.broker.value for m in bundle.metrics], [m.metric.value for m in bundle.metrics],
             [m.submarket.value for m in bundle.metrics], [m.period for m in bundle.metrics]))

        for d in bundle.definitions:
            conn.execute(
                "INSERT INTO metric_definitions (broker, metric, definition, source_id) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (broker, metric) DO UPDATE SET definition = EXCLUDED.definition,"
                " source_id = EXCLUDED.source_id",
                (d.broker.value, d.metric.value, d.definition, source_ids[d.source_key]))
        conn.execute("DELETE FROM metric_definitions WHERE (broker, metric) NOT IN"
                     " (SELECT * FROM unnest(%s::text[], %s::text[]))",
                     ([d.broker.value for d in bundle.definitions], [d.metric.value for d in bundle.definitions]))

        for a in bundle.area_maps:
            conn.execute(
                "INSERT INTO broker_area_map (broker, broker_label, submarket, fit, note) VALUES (%s, %s, %s, %s, %s)"
                " ON CONFLICT (broker, broker_label) DO UPDATE SET submarket = EXCLUDED.submarket,"
                " fit = EXCLUDED.fit, note = EXCLUDED.note",
                (a.broker.value, a.broker_label, a.submarket.value, a.fit, a.note))
        conn.execute("DELETE FROM broker_area_map WHERE (broker, broker_label) NOT IN"
                     " (SELECT * FROM unnest(%s::text[], %s::text[]))",
                     ([a.broker.value for a in bundle.area_maps], [a.broker_label for a in bundle.area_maps]))

        for asset, vector in zip(bundle.assets, asset_vectors, strict=True):
            conn.execute(
                "INSERT INTO portfolio_assets (name, address, submarket, size_sqft, status, notes, source_keys,"
                " embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (name) DO UPDATE SET"
                " address = EXCLUDED.address, submarket = EXCLUDED.submarket, size_sqft = EXCLUDED.size_sqft,"
                " status = EXCLUDED.status, notes = EXCLUDED.notes, source_keys = EXCLUDED.source_keys,"
                " embedding = EXCLUDED.embedding",
                (asset.name, asset.address, asset.submarket.value, asset.size_sqft, asset.status, asset.notes,
                 list(asset.source_keys), _vector(vector)))
        conn.execute("DELETE FROM portfolio_assets WHERE name <> ALL(%s::text[])", ([a.name for a in bundle.assets],))

        conn.execute("DELETE FROM mandate")
        conn.execute("INSERT INTO mandate (label, body) VALUES (%s, %s)",
                     (bundle.mandate.label, Jsonb(bundle.mandate.model_dump(mode="json"))))
        conn.execute("DELETE FROM mandate_statements")
        for text, vector in zip(bundle.mandate.statements, statement_vectors, strict=True):
            conn.execute("INSERT INTO mandate_statements (text, embedding) VALUES (%s, %s)", (text, _vector(vector)))

    return SeedCounts(sources=len(bundle.sources), metrics=len(bundle.metrics), definitions=len(bundle.definitions),
                      area_maps=len(bundle.area_maps), assets=len(bundle.assets),
                      mandate_statements=len(bundle.mandate.statements))


def _fixture_document(record: dict) -> DocumentIn:
    return DocumentIn(source_kind=record["source_kind"], publisher=record["publisher"], title=record["title"],
                      summary=record["summary"], url=record["url"],
                      published_at=datetime.fromisoformat(record["published_at"]),
                      synthetic=bool(record.get("synthetic", False)), fixture_role=record.get("fixture_role"),
                      meta=record.get("meta") or {})


def read_fixture_documents(fixtures_dir: Path, *, demo_items: bool = True) -> list[DocumentIn]:
    """The snapshot's news items, Gazette notices and synthetic items, in file order; with demo_items False, the
    real ones only."""
    docs = []
    for name in FIXTURE_DOCUMENT_FILES:
        path = fixtures_dir / name
        if path.is_file():
            docs += [_fixture_document(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    return docs if demo_items else [doc for doc in docs if not doc.synthetic]


def load_fixtures(conn: psycopg.Connection, fixtures_dir: Path, embedder: Embedder, settings: Settings, *,
                  demo_items: bool = True) -> StoreResult:
    """Load the snapshot documents (skipping any already stored, and the synthetic ones unless demo_items) and the
    Bank of England series."""
    result = store_documents(conn, read_fixture_documents(fixtures_dir, demo_items=demo_items), embedder, settings)
    macro = fixtures_dir / FIXTURE_MACRO_FILE
    if macro.is_file():
        with conn.transaction(), macro.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                conn.execute("INSERT INTO macro_series (series, day, value) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                             (row["series"], date.fromisoformat(row["day"]), Decimal(row["value"])))
    return result


# An internal document is a Markdown file that starts with front matter between two "---" lines.
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
INTERNAL_PUBLISHER = "Nan Fung (internal)"
MOCK_INTERNAL_PUBLISHER = "Nan Fung (mock internal)"
# The most characters of an internal document the documents table holds (documents_summary_check in schema.sql).
MAX_INTERNAL_CHARS = 50_000


class InternalMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=80)
    date: date
    mock: bool = False
    url: str | None = None


def read_internal(directory: Path) -> list[DocumentIn]:
    """The internal documents in a folder (README.md aside), every problem reported together. A mock document is
    stored as a labelled test item, like the synthetic news items."""
    docs, problems = [], []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        match = FRONT_MATTER.match(path.read_text(encoding="utf-8"))
        if match is None:
            problems.append(f"{path.name}: no front matter between '---' lines")
            continue
        try:
            meta = InternalMeta(**(yaml.safe_load(match.group(1)) or {}))
        except (ValidationError, TypeError, yaml.YAMLError) as e:
            problems.append(f"{path.name}: {e}")
            continue
        body = match.group(2).strip()
        if len(body) > MAX_INTERNAL_CHARS:
            problems.append(f"{path.name}: {len(body):,} characters; an internal document can have at most "
                            f"{MAX_INTERNAL_CHARS:,}")
            continue
        docs.append(DocumentIn(
            source_kind="internal", publisher=MOCK_INTERNAL_PUBLISHER if meta.mock else INTERNAL_PUBLISHER,
            title=meta.title, subtitle=meta.kind, summary=body,
            url=meta.url or f"https://internal.invalid/{path.stem}",
            published_at=datetime.combine(meta.date, time(9), tzinfo=LONDON).astimezone(UTC),
            text_format="markdown", synthetic=meta.mock))
    if problems:
        raise SeedError("Internal documents are invalid: " + "; ".join(problems))
    return docs


def load_internal(conn: psycopg.Connection, directory: Path, embedder: Embedder, settings: Settings, *,
                  demo_items: bool = True) -> StoreResult:
    """Store the folder's internal documents (skipping any already stored, and the mock ones unless demo_items),
    chunked along their headings."""
    docs = read_internal(directory)
    return store_documents(conn, docs if demo_items else [doc for doc in docs if not doc.synthetic], embedder,
                           settings)

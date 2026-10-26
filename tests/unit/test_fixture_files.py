import json

import pytest

from london_office_monitor.paths import FIXTURES_DIR, INTERNAL_DIR
from london_office_monitor.seed.load import load_fixtures, load_internal
from london_office_monitor.vocab import FIXTURE_ROLES

ALLOWED_KEYS = {"source_kind", "publisher", "title", "summary", "url", "published_at", "meta",
                "synthetic", "fixture_role"}
SNAPSHOTS = ("news_snapshot.jsonl", "gazette_snapshot.jsonl", "synthetic.jsonl")
CANARY = "ZEBRA-7731"


def _items(name):
    return [json.loads(line) for line in (FIXTURES_DIR / name).read_text().splitlines()]


def test_synthetic_items_are_labelled_and_cover_every_role():
    items = _items("synthetic.jsonl")
    assert sorted(i["fixture_role"] for i in items) == sorted(FIXTURE_ROLES)
    assert all(i["synthetic"] and i["title"].startswith("[SYNTHETIC] ") and
               i["publisher"] == "Synthetic test fixture" for i in items)


def test_only_the_injection_item_carries_the_canary():
    carriers = [i["fixture_role"] for i in _items("synthetic.jsonl") if CANARY in i["summary"]]
    assert carriers == ["injection"]


def test_snapshot_is_not_after_demo_date():
    for name in SNAPSHOTS:
        assert all(i["published_at"] <= "2026-10-24T23:59:59+00:00" for i in _items(name))


def test_snapshot_stores_no_article_bodies():
    for name in SNAPSHOTS:
        assert all(set(i) <= ALLOWED_KEYS and len(i["summary"]) <= 1000 for i in _items(name))


def test_live_snapshot_has_news_and_boe_history():
    assert len(_items("news_snapshot.jsonl")) >= 20
    assert (FIXTURES_DIR / "boe_snapshot.csv").read_text().count("IUDBEDR") > 100


@pytest.mark.db
def test_setup_loads_fixtures(fresh_db, fake_embedder, settings):
    load_fixtures(fresh_db, FIXTURES_DIR, fake_embedder, settings)
    roles = {r[0] for r in fresh_db.execute("SELECT fixture_role FROM documents WHERE fixture_role IS NOT NULL")}
    assert roles == set(FIXTURE_ROLES)
    assert fresh_db.execute("SELECT count(*) FROM macro_series WHERE series = 'IUDBEDR'").fetchone()[0] > 100


@pytest.mark.db
def test_loading_fixtures_twice_adds_nothing(fresh_db, fake_embedder, settings):
    load_fixtures(fresh_db, FIXTURES_DIR, fake_embedder, settings)
    assert load_fixtures(fresh_db, FIXTURES_DIR, fake_embedder, settings).new == 0


@pytest.mark.db
def test_setup_can_leave_out_the_demo_items(fresh_db, fake_embedder, settings):
    # The owner asked for real data, to find real opportunities: `lom setup --no-demo-items` loads the real
    # snapshot without the made-up news items or the mock internal documents.
    load_fixtures(fresh_db, FIXTURES_DIR, fake_embedder, settings, demo_items=False)
    load_internal(fresh_db, INTERNAL_DIR, fake_embedder, settings, demo_items=False)
    assert fresh_db.execute("SELECT count(*) FROM documents WHERE synthetic").fetchone()[0] == 0
    assert fresh_db.execute("SELECT count(*) FROM documents").fetchone()[0] >= 20

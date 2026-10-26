import shutil

import pytest

from london_office_monitor.seed.load import load_seed, read_seed
from tests.conftest import SEED_MIN


@pytest.mark.db
def test_load_seed_is_idempotent(fresh_db, fake_embedder):
    bundle = read_seed(SEED_MIN)
    assert load_seed(fresh_db, bundle, fake_embedder) == load_seed(fresh_db, bundle, fake_embedder)
    assert fresh_db.execute("SELECT count(*) FROM broker_metrics").fetchone()[0] == 7
    assert fresh_db.execute("SELECT count(*) FROM mandate_statements").fetchone()[0] == 3


@pytest.mark.db
def test_reload_keeps_fact_ids_stable(fresh_db, fake_embedder):
    bundle = read_seed(SEED_MIN)
    load_seed(fresh_db, bundle, fake_embedder)
    before = dict(fresh_db.execute("SELECT broker || period, id FROM broker_metrics WHERE metric = 'prime_rent'"))
    load_seed(fresh_db, bundle, fake_embedder)
    after = dict(fresh_db.execute("SELECT broker || period, id FROM broker_metrics WHERE metric = 'prime_rent'"))
    assert before == after


@pytest.mark.db
def test_row_removed_from_seed_is_removed_from_store(fresh_db, fake_embedder, tmp_path):
    load_seed(fresh_db, read_seed(SEED_MIN), fake_embedder)
    shutil.copytree(SEED_MIN, tmp_path, dirs_exist_ok=True)
    lines = (tmp_path / "broker_metrics.csv").read_text().splitlines()
    (tmp_path / "broker_metrics.csv").write_text("\n".join(line for line in lines if not line.startswith("CBRE"))
                                                 + "\n")
    load_seed(fresh_db, read_seed(tmp_path), fake_embedder)
    assert fresh_db.execute("SELECT count(*) FROM broker_metrics WHERE broker = 'CBRE'").fetchone()[0] == 0


@pytest.mark.db
def test_assets_and_statements_are_embedded(fresh_db, fake_embedder):
    load_seed(fresh_db, read_seed(SEED_MIN), fake_embedder)
    assert fresh_db.execute("SELECT count(*) FROM portfolio_assets WHERE embedding IS NULL").fetchone()[0] == 0
    assert fresh_db.execute("SELECT count(*) FROM mandate_statements WHERE embedding IS NULL").fetchone()[0] == 0

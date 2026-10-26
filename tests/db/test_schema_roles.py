import psycopg
import pytest

from london_office_monitor.db.connect import DatabaseUnavailable, DbRole, connect
from london_office_monitor.db.setup import apply_schema
from london_office_monitor.vocab import Broker


def test_connect_reports_unreachable_database(settings):
    with pytest.raises(DatabaseUnavailable, match="docker compose up -d db"):
        connect(settings.model_copy(update={"db_port": 1}), DbRole.OWNER)


@pytest.mark.db
def test_schema_applies_twice(db):
    apply_schema(db)
    apply_schema(db)


@pytest.mark.db
def test_schema_adds_score_inputs_to_a_database_created_before_them(db):
    db.execute("ALTER TABLE opportunities DROP COLUMN IF EXISTS score_inputs")
    apply_schema(db)
    assert db.execute("SELECT 1 FROM information_schema.columns WHERE table_name = 'opportunities'"
                      " AND column_name = 'score_inputs'").fetchone() is not None


@pytest.mark.db
def test_mcp_role_cannot_write_opportunities_or_update_documents(db, settings):
    with connect(settings, DbRole.MCP) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("INSERT INTO opportunities (thread_id, status, card) VALUES ('t', 'draft', '{}')")
    with connect(settings, DbRole.MCP) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("UPDATE documents SET title = 'x'")


@pytest.mark.db
def test_mcp_role_cannot_read_checkpoints(db, settings):
    with connect(settings, DbRole.MCP) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("SELECT count(*) FROM checkpoints")


@pytest.mark.db
def test_app_role_cannot_insert_documents(db, settings):
    with connect(settings, DbRole.APP) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("INSERT INTO documents (source_kind, publisher, title, summary, url, published_at, fetched_at,"
                  " content_hash) VALUES ('news', 'p', 't', 's', 'u', now(), now(), 'h')")


def _insert_metric(db, broker: str) -> None:
    source_id = db.execute("INSERT INTO sources (source_key, publisher, title, url, published, licence_note)"
                           " VALUES (%s, 'p', 't', 'https://x.test', '2026-10-01', 'n') RETURNING id",
                           (f"k-{broker}",)).fetchone()[0]
    db.execute("INSERT INTO broker_metrics (broker, metric, submarket, broker_label, period, value, unit,"
               " measure_note, source_id, checked_at, check_method) VALUES (%s, 'prime_rent', 'City Core', 'City',"
               " '2026Q3', 1, 'GBP psf pa', 'n', %s, '2026-10-24', 'manual')", (broker, source_id))


@pytest.mark.db
def test_knight_frank_row_is_refused_by_the_database(fresh_db):
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_metric(fresh_db, "Knight Frank")


@pytest.mark.db
def test_every_vocab_broker_is_accepted_by_the_database(fresh_db):
    for broker in Broker:
        _insert_metric(fresh_db, broker.value)
    assert fresh_db.execute("SELECT count(DISTINCT broker) FROM broker_metrics").fetchone()[0] == len(Broker)

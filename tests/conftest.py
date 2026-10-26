import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from london_office_monitor.config import Settings
from london_office_monitor.db.connect import DatabaseUnavailable, DbRole, connect
from london_office_monitor.db.setup import apply_roles, apply_schema, create_database, setup_checkpointer
from london_office_monitor.embed import get_embedder
from london_office_monitor.ingest.feeds import DocumentIn
from london_office_monitor.ingest.store import store_documents
from london_office_monitor.seed.load import load_seed, read_seed
from tests.fakes import FakeEmbedder

TEST_DB = "lom_test"
TEST_DATA = Path(__file__).parent / "data"
SEED_MIN = TEST_DATA / "seed_min"
# Every application table, child tables first; checkpoints are left alone (the saver manages them).
APP_TABLES = ("chunks", "documents", "broker_metrics", "metric_definitions", "broker_area_map", "sources",
              "macro_series", "portfolio_assets", "mandate", "mandate_statements", "ingest_runs", "opportunities",
              "approvals", "model_calls", "tool_calls")
# Local development defaults, matching docker-compose.yml and .env.example; CI can override them through the environment.
LOCAL_PASSWORDS = {"LOM_OWNER_DB_PASSWORD": "lom_owner_local", "LOM_APP_DB_PASSWORD": "lom_app_local",
                   "LOM_MCP_DB_PASSWORD": "lom_mcp_local"}


def _password(name: str) -> str:
    return os.environ.get(name, LOCAL_PASSWORDS[name])


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None, db_name=TEST_DB, db_port=int(os.environ.get("LOM_DB_PORT", "55433")),
                    mcp_shared_secret="test-secret-0123456789",
                    owner_db_password=_password("LOM_OWNER_DB_PASSWORD"),
                    app_db_password=_password("LOM_APP_DB_PASSWORD"),
                    mcp_db_password=_password("LOM_MCP_DB_PASSWORD"))


@pytest.fixture(scope="session")
def live_settings(settings) -> Settings:
    return settings.model_copy(update={"live": True})


@pytest.fixture(scope="session")
def db(settings):
    """A fresh lom_test database with the schema, roles and checkpoint tables, as an OWNER autocommit connection."""
    try:
        create_database(settings, TEST_DB, reset=True)
    except DatabaseUnavailable as e:
        pytest.skip(str(e))
    conn = connect(settings, DbRole.OWNER, autocommit=True)
    apply_schema(conn)
    apply_roles(conn, settings, TEST_DB)
    setup_checkpointer(settings, TEST_DB)
    yield conn
    conn.close()


@pytest.fixture
def fresh_db(db):
    """The session database with every application table emptied and its ids restarted."""
    db.execute(f"TRUNCATE {', '.join(APP_TABLES)} RESTART IDENTITY CASCADE")
    return db


@pytest.fixture(scope="session")
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# The pinned demo clock (09:00 London on 2026-10-24) and eight documents for search tests. Ids are looked up by URL.
DEMO_NOW = datetime(2026, 10, 24, 8, 0, tzinfo=UTC)
TOWER_SALE_URL = "https://www.cityam.test/barclays-canary-wharf-tower"
TOWER_SALE_ECHO_URL = "https://www.bisnow.test/barclays-canary-wharf-tower"
CITY_ROAD_URL = "https://www.propertyweek.test/99-city-road-starts"
KINGS_CROSS_URL = "https://www.bisnow.test/ai-firm-kings-cross"
SYNTHETIC_URL = "https://synthetic.invalid/news/example-tower"
FUTURE_URL = "https://www.cityam.test/after-the-demo-clock"


def _doc(publisher, title, summary, url, day, hour=9, **extra) -> DocumentIn:
    return DocumentIn(source_kind="news", publisher=publisher, title=title, summary=summary, url=url,
                      published_at=datetime(2026, 10, day, hour, tzinfo=UTC), **extra)


TOWER_SALE = ("Barclays buys Canary Wharf tower for £750m",
              "Barclays has bought One Churchill Place, its Canary Wharf headquarters tower, for £750m.")
TEST_DOCS = [
    _doc("City AM", *TOWER_SALE, TOWER_SALE_URL, 20),
    _doc("Bisnow London", *TOWER_SALE, TOWER_SALE_ECHO_URL, 20, hour=12),
    _doc("Property Week", "99 City Road tower starts on site",
         "Construction has started on 99 City Road, a 475,000 sq ft office tower at Old Street.", CITY_ROAD_URL, 21),
    _doc("Bisnow London", "AI firm leases offices in King's Cross",
         "An artificial intelligence company leased 60,000 sq ft of offices near King's Cross station.",
         KINGS_CROSS_URL, 22),
    _doc("Construction Enquirer", "Retail park in Kent wins planning",
         "A 200,000 sq ft retail park with a supermarket has been approved.", "https://www.ce.test/kent-retail", 18),
    _doc("BBC Business", "Bank of England holds rates", "Bank Rate stays at 3.75%.", "https://www.bbc.test/rates", 16),
    _doc("Synthetic test fixture", "[SYNTHETIC] Example Tower Ltd misses a loan payment",
         "SYNTHETIC TEST FIXTURE. Example Tower Ltd missed an interest payment on its City office.", SYNTHETIC_URL, 23,
         synthetic=True, fixture_role="injection"),
    _doc("City AM", "Office letting record after the demo clock", "A record office letting.", FUTURE_URL, 25),
]


def doc_id(conn, url: str) -> int:
    return conn.execute("SELECT id FROM documents WHERE url = %s", (url,)).fetchone()[0]


def fact_id(conn, broker: str, metric: str, submarket: str, period: str) -> int:
    return conn.execute("SELECT id FROM broker_metrics WHERE broker = %s AND metric = %s AND submarket = %s"
                        " AND period = %s", (broker, metric, submarket, period)).fetchone()[0]


@pytest.fixture
def seeded(fresh_db, fake_embedder):
    """lom_test with the minimal seed loaded."""
    load_seed(fresh_db, read_seed(SEED_MIN), fake_embedder)
    return fresh_db


@pytest.fixture
def seeded_docs(seeded, fake_embedder, settings):
    """The minimal seed plus the eight test documents, embedded with the fake embedder."""
    store_documents(seeded, TEST_DOCS, fake_embedder, settings)
    return seeded


@pytest.fixture
def seeded_docs_real_embedder(fresh_db, settings):
    """The minimal seed and test documents embedded with the real model (tests marked `embedding`)."""
    embedder = get_embedder(settings)
    load_seed(fresh_db, read_seed(SEED_MIN), embedder)
    store_documents(fresh_db, TEST_DOCS, embedder, settings)
    return fresh_db


@pytest.fixture
def mcp_server(seeded_docs, settings, fake_embedder):
    """The office-data MCP server on a free port, against the seeded test database, with the fake embedder."""
    from london_office_monitor.mcp_server.server import build_app
    from tests.mcp_helpers import free_port, start_server

    server_settings = settings.model_copy(update={"mcp_port": free_port()})
    running = start_server(build_app(server_settings, embedder=fake_embedder), server_settings)
    yield running
    running.stop()


@pytest.fixture
async def mcp_tools(mcp_server):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from london_office_monitor.mcp_client import mcp_connection_config

    return await MultiServerMCPClient(mcp_connection_config(mcp_server.settings)).get_tools()

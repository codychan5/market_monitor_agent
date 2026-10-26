"""Create the database, its schema, the two least-privilege login roles and the checkpoint tables."""
import re
from pathlib import Path

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from pgvector.psycopg import register_vector
from psycopg import sql

from ..config import Settings
from ..vocab import DB_ROLE_USERS, Broker, DbRole, Metric, Submarket
from .connect import connect

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
ROLES_PATH = Path(__file__).with_name("roles.sql")
# CREATE and DROP DATABASE must run from another database; every Postgres server has this one.
MAINTENANCE_DB = "postgres"
# Database names are interpolated as identifiers, so anything beyond plain lowercase names is refused outright.
DB_NAME_PATTERN = re.compile(r"[a-z_][a-z0-9_]{0,62}")
LOGIN_ROLES = (DbRole.APP, DbRole.MCP)
# Created by PostgresSaver.setup() (spike Q7); the agent's role needs full row access to them.
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations")


def _check_db_name(db_name: str) -> str:
    if not DB_NAME_PATTERN.fullmatch(db_name):
        raise ValueError(f"database name must be lowercase letters, digits and underscores, got {db_name!r}")
    return db_name


def create_database(settings: Settings, db_name: str, *, reset: bool) -> None:
    """Create db_name if it is missing; with reset, drop it first (closing other sessions)."""
    _check_db_name(db_name)
    with connect(settings, DbRole.OWNER, MAINTENANCE_DB, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)).fetchone() is not None
        if exists and reset:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(db_name)))
            exists = False
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))


def _literal_list(conn: psycopg.Connection, values) -> str:
    return sql.SQL(", ").join(sql.Literal(str(v)) for v in values).as_string(conn)


def render_schema(conn: psycopg.Connection) -> str:
    """schema.sql with its vocabulary placeholders filled, each value quoted as an SQL literal."""
    return (SCHEMA_PATH.read_text()
            .replace("{{BROKERS}}", _literal_list(conn, Broker))
            .replace("{{METRICS}}", _literal_list(conn, Metric))
            .replace("{{SUBMARKETS}}", _literal_list(conn, Submarket)))


def apply_schema(conn: psycopg.Connection) -> None:
    """Create every table and index that does not exist yet. Safe to run repeatedly."""
    conn.execute(render_schema(conn))
    if not conn.autocommit:
        conn.commit()
    # A connection opened before the extension existed cannot adapt vectors until it is registered.
    register_vector(conn)


def apply_roles(conn: psycopg.Connection, settings: Settings, db_name: str) -> None:
    """Create the app and MCP login roles if missing, set their passwords from settings, and grant least privilege."""
    passwords = {DbRole.APP: settings.app_db_password, DbRole.MCP: settings.mcp_db_password}
    for role in LOGIN_ROLES:
        user = DB_ROLE_USERS[role]
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,)).fetchone() is None:
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(user)))
        conn.execute(sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
            sql.Identifier(user), sql.Literal(passwords[role].get_secret_value())))
    grants = ROLES_PATH.read_text().replace("{{DB}}", sql.Identifier(_check_db_name(db_name)).as_string(conn))
    conn.execute(grants)
    if not conn.autocommit:
        conn.commit()


def setup_checkpointer(settings: Settings, db_name: str) -> None:
    """Create LangGraph's checkpoint tables as the owner, then let the agent's role use them."""
    with PostgresSaver.from_conn_string(settings.database_url(DbRole.OWNER, _check_db_name(db_name))) as saver:
        saver.setup()
    with connect(settings, DbRole.OWNER, db_name, autocommit=True) as conn:
        conn.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON {} TO {}").format(
            sql.SQL(", ").join(sql.Identifier(t) for t in CHECKPOINT_TABLES),
            sql.Identifier(DB_ROLE_USERS[DbRole.APP])))

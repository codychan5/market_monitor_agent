"""Connections to Postgres as one of three roles, with pgvector registered once the extension exists."""
import psycopg
from pgvector.psycopg import register_vector, register_vector_async

from ..config import Settings
from ..redact import redact
from ..vocab import DB_ROLE_USERS, DbRole

__all__ = ["DatabaseUnavailable", "DbRole", "aconnect", "connect"]

# A reviewer whose database is not running should see the fix within seconds, not wait on a TCP timeout.
CONNECT_TIMEOUT_SECONDS = 5
VECTOR_TYPE_QUERY = "SELECT 1 FROM pg_type WHERE typname = 'vector'"


class DatabaseUnavailable(Exception):
    """The database could not be reached, or refused the login."""


def _unavailable(settings: Settings, role: DbRole, error: Exception) -> DatabaseUnavailable:
    reason = redact(str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__, settings)
    return DatabaseUnavailable(
        f"Cannot connect to the database at {settings.db_host}:{settings.db_port} as {DB_ROLE_USERS[role]}: {reason}. "
        f"Start it with `docker compose up -d db` (or set LOM_DB_PORT if it runs elsewhere), then run `lom setup`.")


def connect(settings: Settings, role: DbRole, db_name: str | None = None, *,
            autocommit: bool = False) -> psycopg.Connection:
    """A connection as `role`. Raises DatabaseUnavailable with the fix when the server is down or refuses us."""
    try:
        conn = psycopg.connect(settings.database_url(role, db_name), connect_timeout=CONNECT_TIMEOUT_SECONDS,
                               autocommit=autocommit)
    except psycopg.OperationalError as e:
        raise _unavailable(settings, role, e) from e
    # A fresh database has no vector type until the schema creates the extension; register it only when present.
    if conn.execute(VECTOR_TYPE_QUERY).fetchone():
        register_vector(conn)
    if not autocommit:
        conn.rollback()
    return conn


async def aconnect(settings: Settings, role: DbRole, db_name: str | None = None, *,
                   autocommit: bool = False) -> psycopg.AsyncConnection:
    """The async equivalent of connect()."""
    try:
        conn = await psycopg.AsyncConnection.connect(settings.database_url(role, db_name),
                                                     connect_timeout=CONNECT_TIMEOUT_SECONDS, autocommit=autocommit)
    except psycopg.OperationalError as e:
        raise _unavailable(settings, role, e) from e
    if await (await conn.execute(VECTOR_TYPE_QUERY)).fetchone():
        await register_vector_async(conn)
    if not autocommit:
        await conn.rollback()
    return conn

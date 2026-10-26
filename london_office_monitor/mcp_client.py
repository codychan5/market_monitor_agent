"""Reaching the office-data MCP server: the client config, and starting the server when nothing answers."""
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

from .config import Settings
from .paths import PROJECT_ROOT

SERVER_NAME = "office-data"
HEALTH_TIMEOUT_SECONDS = 2
# Loading the embedding model dominates start-up: a few seconds from cache, longer on a slow disk.
START_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 0.25
STOP_TIMEOUT_SECONDS = 10
LOG_DIR = PROJECT_ROOT / ".cache" / "logs"


class McpServerError(Exception):
    """The MCP server could not be reached or started."""


@dataclass
class McpHandle:
    url: str
    spawned: bool
    process: subprocess.Popen | None = None

    def close(self) -> None:
        """Stop the server, but only if this handle started it."""
        if not self.spawned or self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.kill()


def mcp_connection_config(settings: Settings) -> dict:
    """The MultiServerMCPClient configuration, with the shared secret as a bearer token (spike Q4)."""
    return {SERVER_NAME: {"transport": "streamable_http", "url": settings.mcp_url,
                          "headers": {"Authorization": f"Bearer {settings.mcp_shared_secret.get_secret_value()}"}}}


def _health(base_url: str) -> dict | None:
    try:
        response = httpx.get(f"{base_url}/healthz", timeout=HEALTH_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return None
    return response.json() if response.status_code == 200 else None


def ensure_mcp_server(settings: Settings, *, port: int | None = None, db_name: str | None = None) -> McpHandle:
    """Reuse a healthy server for the same database, or start one and wait until it reports healthy."""
    port = port or settings.mcp_port
    db_name = db_name or settings.db_name
    base_url = f"http://{settings.mcp_host}:{port}"
    health = _health(base_url)
    if health is not None:
        if health.get("db") != db_name:
            raise McpServerError(f"Port {port} already serves the MCP server for database {health.get('db')!r}, not "
                                 f"{db_name!r}: stop that server or choose another LOM_MCP_PORT.")
        return McpHandle(url=f"{base_url}/mcp", spawned=False)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"mcp-{port}.log"
    env = {**os.environ, "LOM_MCP_PORT": str(port), "LOM_DB_NAME": db_name}
    with log_path.open("ab") as log:
        process = subprocess.Popen([sys.executable, "-m", "london_office_monitor.mcp_server"], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise McpServerError(f"The MCP server stopped while starting (exit code {process.returncode}); see "
                                 f"{log_path}.")
        if _health(base_url) is not None:
            return McpHandle(url=f"{base_url}/mcp", spawned=True, process=process)
        time.sleep(POLL_INTERVAL_SECONDS)
    process.terminate()
    raise McpServerError(f"The MCP server did not become healthy within {START_TIMEOUT_SECONDS} s; see {log_path}.")

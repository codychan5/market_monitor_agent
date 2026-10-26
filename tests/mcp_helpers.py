"""Helpers for calling MCP tools in tests: results arrive as content blocks whose text is JSON (spike Q4)."""
import json
import socket
import threading
import time
from dataclasses import dataclass

import uvicorn

from london_office_monitor.config import Settings


def tool_result_text(result) -> str:
    if isinstance(result, str):
        return result
    return "".join(block.get("text", "") for block in result if isinstance(block, dict))


async def call(tools, name: str, **args) -> dict:
    tool = next(t for t in tools if t.name == name)
    return json.loads(tool_result_text(await tool.ainvoke(args)))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class McpTestServer:
    url: str
    secret: str
    settings: Settings
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def start_server(app, settings: Settings) -> McpTestServer:
    """Run the ASGI app under uvicorn in a background thread until it reports started."""
    server = uvicorn.Server(uvicorn.Config(app, host=settings.mcp_host, port=settings.mcp_port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("test MCP server did not start")
        time.sleep(0.05)
    return McpTestServer(url=settings.mcp_url, secret=settings.mcp_shared_secret.get_secret_value(),
                         settings=settings, server=server, thread=thread)

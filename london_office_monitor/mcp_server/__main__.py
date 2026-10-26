"""Run the office-data MCP server: `python -m london_office_monitor.mcp_server` (or `lom mcp`)."""
import sys

import uvicorn
from pydantic import ValidationError

from ..config import Settings, describe_validation_error
from ..embed import EmbeddingError
from .server import build_app


def main() -> int:
    try:
        settings = Settings()
    except ValidationError as e:
        print(f"Settings are invalid ({describe_validation_error(e)}).", file=sys.stderr)
        return 1
    try:
        app = build_app(settings)
    except EmbeddingError as e:
        print(str(e), file=sys.stderr)
        return 1
    uvicorn.run(app, host=settings.mcp_host, port=settings.mcp_port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

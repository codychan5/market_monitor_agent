# Regression: ISSUE-009 — lom mcp --port 70000 crashed with a 154-line OverflowError traceback
# Found by /qa on 2026-10-25
# Report: .gstack/qa-reports/qa-report-lom-2026-10-25.md
import pytest
from typer.testing import CliRunner

from london_office_monitor.cli import app


@pytest.mark.parametrize("command", ["mcp", "ui"])
def test_a_port_outside_the_tcp_range_is_a_usage_error(command, monkeypatch):
    monkeypatch.setenv("LOM_DB_NAME", "lom_test")  # never the app's database, whatever happens
    result = CliRunner().invoke(app, [command, "--port", "70000"])
    assert result.exit_code == 2 and "65535" in result.output
    assert "Traceback" not in result.output


def test_a_port_setting_outside_the_tcp_range_is_named_without_a_traceback(monkeypatch):
    monkeypatch.setenv("LOM_DB_NAME", "lom_test")
    monkeypatch.setenv("LOM_MCP_PORT", "70000")
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 1 and "Settings are invalid" in result.output and "mcp_port" in result.output

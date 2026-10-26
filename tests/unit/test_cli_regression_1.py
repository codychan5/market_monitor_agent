# Regression: ISSUE-004 — a failed chat turn printed an 80-line traceback to the analyst's terminal
# Found by /qa on 2026-10-25
# Report: .gstack/qa-reports/qa-report-lom-2026-10-25.md
import logging
from types import SimpleNamespace

from typer.testing import CliRunner

from london_office_monitor.agent.graph import TurnResult
from london_office_monitor.cli import app


def test_a_failed_chat_turn_logs_one_line_without_a_traceback(monkeypatch):
    # The CLI promises "a one-line cause and fix on failure, never a traceback"; the turn boundary logs the failure
    # with log.exception, and with no logging set up Python printed the whole traceback to stderr.
    async def failing_turn(settings, thread_id, text, **kwargs):
        try:
            raise RuntimeError("Error code: 401 - User not found.")
        except RuntimeError:
            logging.getLogger("london_office_monitor.graph").exception("turn t1 on thread %s failed", thread_id)
        return TurnResult(thread_id=thread_id, turn_id="", answer=None,
                          error="Model access failed: the OpenRouter key is missing, invalid or out of credit.")

    monkeypatch.setattr("london_office_monitor.agent.graph.run_turn", failing_turn)
    monkeypatch.setattr("london_office_monitor.mcp_client.ensure_mcp_server",
                        lambda settings: SimpleNamespace(close=lambda: None))
    result = CliRunner().invoke(app, ["chat"], input="What is City prime rent?\n\n")
    assert result.exit_code == 0 and "Model access failed" in result.stdout
    assert "Traceback" not in result.stderr
    assert "turn t1 on thread" in result.stderr and "RuntimeError: Error code: 401 - User not found." in result.stderr
    assert len([line for line in result.stderr.splitlines() if line.strip()]) == 1

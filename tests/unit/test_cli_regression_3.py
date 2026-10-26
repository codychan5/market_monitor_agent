# Regression: ISSUE-008 — lom ingest --kinds "," ran an ingest of nothing and exited 0
# Found by /qa on 2026-10-25
# Report: .gstack/qa-reports/qa-report-lom-2026-10-25.md
import pytest
from typer.testing import CliRunner

from london_office_monitor.cli import app


@pytest.mark.parametrize("kinds", [",", " ", ""])
def test_ingest_refuses_an_empty_list_of_sources(kinds, monkeypatch):
    # A cron line with an empty or mangled --kinds would otherwise stop live ingestion silently with exit 0.
    monkeypatch.setenv("LOM_DB_NAME", "lom_test")  # if this ever regresses, it must not reach the app's database
    result = CliRunner().invoke(app, ["ingest", "--kinds", kinds])
    assert result.exit_code == 1 and "at least one" in result.output

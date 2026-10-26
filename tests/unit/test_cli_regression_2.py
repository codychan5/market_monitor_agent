# Regression: ISSUE-001 — lom eval --budget-usd NaN crashed with a traceback instead of the usage message
# Found by /qa on 2026-10-25
# Report: .gstack/qa-reports/qa-report-lom-2026-10-25.md
import pytest
from typer.testing import CliRunner

from london_office_monitor.cli import app


@pytest.mark.parametrize("budget", ["NaN", "sNaN", "Infinity", "-1", "0", "abc"])
def test_eval_rejects_a_budget_that_is_not_a_positive_amount(budget):
    # Decimal("NaN") parses, and comparing it raised InvalidOperation outside the error handling.
    result = CliRunner().invoke(app, ["eval", "--models", "openrouter:z-ai/glm-5.3", "--budget-usd", budget])
    assert result.exit_code == 1 and (result.exception is None or isinstance(result.exception, SystemExit))
    assert "--budget-usd must be" in result.output

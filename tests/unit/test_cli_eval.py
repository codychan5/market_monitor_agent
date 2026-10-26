from typer.testing import CliRunner

from london_office_monitor.cli import app


def test_eval_refuses_to_reset_the_database_the_app_uses(monkeypatch):
    # lom eval drops and recreates lom_eval, so it must never run when that is the app's own database.
    monkeypatch.setenv("LOM_DB_NAME", "lom_eval")
    result = CliRunner().invoke(app, ["eval", "--models", "openrouter:z-ai/glm-5.3"])
    assert result.exit_code == 1 and "LOM_DB_NAME is lom_eval" in result.output

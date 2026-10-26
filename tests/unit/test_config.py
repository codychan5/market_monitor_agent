import pytest
from pydantic import ValidationError

from london_office_monitor.config import Settings

VALID = {"owner_db_password": "x" * 8, "app_db_password": "x" * 8, "mcp_db_password": "x" * 8}


def test_unknown_setting_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **VALID, mcp_shared_secret="s" * 16, made_up_field=1)


def test_short_mcp_secret_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **VALID, mcp_shared_secret="short")


def test_short_database_password_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{**VALID, "app_db_password": "short"}, mcp_shared_secret="s" * 16)


def test_model_without_openrouter_prefix_is_rejected():
    with pytest.raises(ValidationError, match="openrouter:"):
        Settings(_env_file=None, **VALID, mcp_shared_secret="s" * 16, model_judge="qwen/qwen3.7-plus")


def test_database_url_uses_the_role_user_and_escapes_the_password():
    s = Settings(_env_file=None, **{**VALID, "app_db_password": "p@ss/word#1"}, mcp_shared_secret="s" * 16)
    from london_office_monitor.db.connect import DbRole

    assert s.database_url(DbRole.APP, "lom_eval") == "postgresql://lom_app:p%40ss%2Fword%231@127.0.0.1:55433/lom_eval"

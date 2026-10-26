from datetime import UTC, datetime, timedelta

from london_office_monitor import clock
from london_office_monitor.redact import redact


def test_demo_clock_is_pinned(settings):
    assert clock.now(settings) == datetime(2026, 10, 24, 8, 0, tzinfo=UTC)  # 09:00 BST


def test_live_clock_is_real_time(settings):
    live = settings.model_copy(update={"live": True})
    assert abs(clock.now(live) - datetime.now(UTC)) < timedelta(seconds=5)


def test_redact_hides_every_secret(settings):
    text = f"a={settings.mcp_shared_secret.get_secret_value()} b={settings.app_db_password.get_secret_value()}"
    out = redact(text, settings)
    assert "test-secret" not in out and out.count("[redacted]") == 2


def test_redact_leaves_text_without_secrets_alone(settings):
    assert redact("vacancy 7.1% in the City", settings) == "vacancy 7.1% in the City"

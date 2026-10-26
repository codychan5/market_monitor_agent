"""Live checks against OpenRouter (cost: well under a cent). Run with: uv run pytest -m live tests/live"""
import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from london_office_monitor.agent.llm import build_chat_model, classify_model_error
from london_office_monitor.agent.telemetry import cost_from_message
from london_office_monitor.config import Settings

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_key_settings() -> Settings:
    settings = Settings()
    if settings.openrouter_api_key is None:
        pytest.skip("OPENROUTER_API_KEY is not set")
    return settings


async def test_default_orchestrator_answers_and_reports_cost(live_key_settings):
    reply = await build_chat_model(live_key_settings, "orchestrator").ainvoke(
        [HumanMessage("Reply with the single word: ok")])
    assert "ok" in reply.text.lower() and cost_from_message(reply) is not None


async def test_wrong_key_is_an_access_error(live_key_settings):
    wrong = live_key_settings.model_copy(update={"openrouter_api_key": SecretStr("sk-or-v1-not-a-real-key")})
    with pytest.raises(Exception) as caught:
        await build_chat_model(wrong, "orchestrator").ainvoke([HumanMessage("hi")])
    assert classify_model_error(caught.value) == "access"

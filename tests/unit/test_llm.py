from decimal import Decimal

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from london_office_monitor.agent.llm import build_chat_model, classify_model_error
from london_office_monitor.agent.telemetry import cost_from_message

# Observed in spike Q1 (GLM-5.3 through OpenRouter).
SPIKE_Q1_METADATA = {"token_usage": {"completion_tokens": 40, "prompt_tokens": 19, "total_tokens": 59,
                                     "cost": 0.0002026, "is_byok": False,
                                     "cost_details": {"upstream_inference_cost": 0.0002026}},
                     "model_name": "z-ai/glm-5.3", "model_provider": "openai", "finish_reason": "stop"}


def api_status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.APIStatusError("error", response=httpx.Response(status, request=request), body=None)


def test_build_chat_model_rejects_unprefixed_model(settings):
    with pytest.raises(ValueError, match="openrouter:"):
        build_chat_model(settings, "orchestrator", model_override="z-ai/glm-5.3")


def test_extra_body_requests_usage_and_no_data_collection(settings):
    m = build_chat_model(settings, "orchestrator", with_fallback=False)
    assert m.extra_body == {"usage": {"include": True}, "provider": {"data_collection": "deny"}}
    assert m.model_name == "z-ai/glm-5.3" and m.temperature == 0 and m.max_tokens == settings.llm_max_output_tokens


def test_data_collection_follows_the_setting(settings):
    m = build_chat_model(settings.model_copy(update={"provider_data_collection": "allow"}), "judge")
    assert m.extra_body["provider"] == {"data_collection": "allow"}


def test_fallback_attached_only_when_requested(settings):
    assert hasattr(build_chat_model(settings, "orchestrator"), "fallbacks")
    assert not hasattr(build_chat_model(settings, "orchestrator", with_fallback=False), "fallbacks")
    assert not hasattr(build_chat_model(settings, "judge"), "fallbacks")


def test_access_errors_do_not_trigger_the_fallback(settings):
    handled = tuple(build_chat_model(settings, "orchestrator").exceptions_to_handle)
    assert not issubclass(openai.AuthenticationError, handled) and not issubclass(openai.PermissionDeniedError, handled)
    assert issubclass(openai.APITimeoutError, handled) and issubclass(openai.InternalServerError, handled)


def test_model_override_replaces_the_role_model(settings):
    m = build_chat_model(settings, "investigator", with_fallback=False, model_override="openrouter:moonshotai/kimi-k3")
    assert m.model_name == "moonshotai/kimi-k3"


def test_judge_runs_with_reasoning_off(settings):
    # The judge's forced tool call is refused in thinking mode by Qwen3.7 Plus's only provider (live scan 2026-10-24).
    assert build_chat_model(settings, "judge").extra_body["reasoning"] == {"enabled": False}
    for role in ("orchestrator", "investigator"):
        assert "reasoning" not in build_chat_model(settings, role, with_fallback=False).extra_body


def test_classify_model_error_maps_401_and_402():
    assert classify_model_error(api_status_error(401)) == "access"
    assert classify_model_error(api_status_error(402)) == "access"
    assert classify_model_error(api_status_error(403)) == "access"
    assert classify_model_error(api_status_error(429)) == "rate_limited"
    assert classify_model_error(api_status_error(502)) == "provider"


def test_classify_model_error_handles_timeouts_and_wrapped_errors():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    assert classify_model_error(openai.APITimeoutError(request)) == "timeout"
    wrapped = RuntimeError("agent failed")
    wrapped.__cause__ = api_status_error(401)
    assert classify_model_error(wrapped) == "access"
    assert classify_model_error(ValueError("something else")) == "provider"


def test_cost_from_message_reads_openrouter_usage():
    assert cost_from_message(AIMessage("x", response_metadata=SPIKE_Q1_METADATA)) == Decimal("0.0002026")
    assert cost_from_message(AIMessage("x")) is None

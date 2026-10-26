"""One factory for every model call: OpenRouter only, one model per role, a fallback for transient failures."""
from typing import Literal

import httpx
import openai
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from ..config import MODEL_PREFIX, Settings

Role = Literal["orchestrator", "investigator", "judge"]
ModelErrorKind = Literal["access", "rate_limited", "timeout", "provider"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ACCESS_MESSAGE = "Model access failed: the OpenRouter key is missing, invalid or out of credit."
# Without a key the client is still built, so the first call fails with a 401 that becomes MODEL_ACCESS_MESSAGE.
MISSING_KEY_PLACEHOLDER = "missing-openrouter-key"
ACCESS_STATUSES = frozenset({401, 402, 403})
RATE_LIMITED_STATUS = 429
# OpenRouter sometimes reports an upstream failure inside an HTTP 200 reply as {"error": {"code": 503}}; the
# client retries only on the HTTP status, so a transient code in the body is copied onto the status.
RETRYABLE_IN_BODY_CODES = frozenset({408, 429, 500, 502, 503, 504})
# The fallback model takes over only on failures another model could avoid; a bad key or no credit is not one.
FALLBACK_ON = (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError,
               openai.UnprocessableEntityError)
# The judge answers through a forced tool call, which Qwen3.7 Plus's only provider refuses in thinking mode (HTTP 400,
# live scan 2026-10-24). A claim check needs no reasoning: with it off the probe gave the right verdicts in ~2 s.
ROLE_REASONING: dict[Role, dict] = {"judge": {"enabled": False}}


def _in_body_code(body) -> int | None:
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, int) and code in RETRYABLE_IN_BODY_CODES else None


def _promote_in_body_error(response: httpx.Response) -> None:
    if response.status_code == 200 and "json" in response.headers.get("content-type", ""):
        response.read()
        try:
            code = _in_body_code(response.json())
        except ValueError:
            return
        if code:
            response.status_code = code


async def _apromote_in_body_error(response: httpx.Response) -> None:
    if response.status_code == 200 and "json" in response.headers.get("content-type", ""):
        await response.aread()
        try:
            code = _in_body_code(response.json())
        except ValueError:
            return
        if code:
            response.status_code = code


def model_id(settings: Settings, role: Role) -> str:
    return {"orchestrator": settings.model_orchestrator, "investigator": settings.model_investigator,
            "judge": settings.model_judge}[role]


def fallback_id(settings: Settings, role: Role) -> str | None:
    return {"orchestrator": settings.model_orchestrator_fallback,
            "investigator": settings.model_investigator_fallback, "judge": settings.model_judge_fallback}[role]


def _chat_model(name: str, settings: Settings, *, reasoning: dict | None = None) -> ChatOpenAI:
    if not name.startswith(MODEL_PREFIX):
        raise ValueError(f"model must be named {MODEL_PREFIX}<id>, got {name!r}")
    key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else MISSING_KEY_PLACEHOLDER
    extra_body = {"usage": {"include": True}, "provider": {"data_collection": settings.provider_data_collection}}
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    return ChatOpenAI(
        model=name[len(MODEL_PREFIX):], base_url=OPENROUTER_BASE_URL, api_key=key, temperature=0,
        timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_output_tokens, extra_body=extra_body,
        http_client=httpx.Client(event_hooks={"response": [_promote_in_body_error]}),
        http_async_client=httpx.AsyncClient(event_hooks={"response": [_apromote_in_body_error]}))


def build_chat_model(settings: Settings, role: Role, *, with_fallback: bool = True,
                     model_override: str | None = None) -> Runnable:
    """The role's model (or model_override), with the role's fallback attached when asked for and configured."""
    reasoning = ROLE_REASONING.get(role)
    primary = _chat_model(model_override or model_id(settings, role), settings, reasoning=reasoning)
    fallback = fallback_id(settings, role)
    if with_fallback and fallback:
        return primary.with_fallbacks([_chat_model(fallback, settings, reasoning=reasoning)],
                                      exceptions_to_handle=FALLBACK_ON)
    return primary


def _status(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None) if isinstance(exc, openai.APIStatusError) else None


def classify_model_error(exc: BaseException) -> ModelErrorKind:
    """What kind of model failure this is, looking through wrapped causes: access problems are the user's to fix,
    the rest are the provider's."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = _status(current)
        if status in ACCESS_STATUSES:
            return "access"
        if status == RATE_LIMITED_STATUS:
            return "rate_limited"
        if isinstance(current, (openai.APITimeoutError, httpx.TimeoutException)):
            return "timeout"
        if status is not None:
            return "provider"
        current = current.__cause__ or current.__context__
    return "provider"

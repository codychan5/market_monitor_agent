"""Settings, read from the environment and `.env`. Unknown keys are rejected so a typo fails fast."""
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import AliasChoices, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .vocab import DB_ROLE_USERS, DbRole

MODEL_PREFIX = "openrouter:"
# Shortest acceptable database password and MCP secret; shorter values are typos or placeholders.
MIN_PASSWORD_LENGTH = 8
MIN_MCP_SECRET_LENGTH = 16
MAX_PORT = 65535  # the highest TCP port; a larger one failed deep in the socket layer with a traceback


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOM_", env_file=".env", extra="forbid")

    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_api_key"))

    db_host: str = "127.0.0.1"
    db_port: int = Field(55433, ge=1, le=MAX_PORT)
    db_name: str = "lom"
    owner_db_password: SecretStr
    app_db_password: SecretStr
    mcp_db_password: SecretStr

    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(8765, ge=1, le=MAX_PORT)
    mcp_shared_secret: SecretStr

    live: bool = False
    demo_date: date = date(2026, 10, 24)

    model_orchestrator: str = "openrouter:z-ai/glm-5.3"
    model_orchestrator_fallback: str | None = "openrouter:deepseek/deepseek-v4.1-flash"
    model_investigator: str = "openrouter:deepseek/deepseek-v4.1-flash"
    model_investigator_fallback: str | None = "openrouter:z-ai/glm-5.3"
    model_judge: str = "openrouter:qwen/qwen3.7-plus"
    model_judge_fallback: str | None = None
    provider_data_collection: Literal["deny", "allow"] = "deny"

    llm_timeout_seconds: float = 90
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 8192

    chat_max_model_calls: int = 45
    chat_max_tool_calls: int = 60
    chat_deadline_seconds: int = 180
    scan_max_model_calls: int = 90
    scan_max_tool_calls: int = 120
    scan_deadline_seconds: int = 600
    subagent_max_model_calls: int = 12
    subagent_max_tool_calls: int = 24
    subagent_deadline_seconds: int = 90
    # An investigator's final call starts this long before its deadline: longer than the slowest investigator call
    # seen (23.7 s, live scan 2026-10-24), so the call that returns the result finishes in time.
    subagent_wrap_up_seconds: int = 30
    max_candidates: int = 5
    subagent_concurrency: int = 5

    refresh_min_interval_minutes: int = 15
    store_stale_minutes: int = 60
    dedup_similarity: float = 0.90
    dedup_window_days: int = 3

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B-Q"
    embedding_dim: int = 1024
    embedding_cache_dir: Path = Path(".cache/fastembed")
    chunk_max_tokens: int = 350

    eval_budget_usd: Decimal = Decimal("5")
    user_agent: str = "london-office-monitor/0.1 (proof of concept)"

    @field_validator("owner_db_password", "app_db_password", "mcp_db_password")
    @classmethod
    def _password_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"must be at least {MIN_PASSWORD_LENGTH} characters")
        return value

    @field_validator("mcp_shared_secret")
    @classmethod
    def _secret_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < MIN_MCP_SECRET_LENGTH:
            raise ValueError(f"must be at least {MIN_MCP_SECRET_LENGTH} characters")
        return value

    @field_validator("model_orchestrator", "model_orchestrator_fallback", "model_investigator",
                     "model_investigator_fallback", "model_judge", "model_judge_fallback")
    @classmethod
    def _openrouter_model(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(MODEL_PREFIX):
            raise ValueError(f"model must be named {MODEL_PREFIX}<id>, got {value!r}")
        return value

    def database_url(self, role: DbRole, db_name: str | None = None) -> str:
        password = {DbRole.OWNER: self.owner_db_password, DbRole.APP: self.app_db_password,
                    DbRole.MCP: self.mcp_db_password}[role].get_secret_value()
        return (f"postgresql://{DB_ROLE_USERS[role]}:{quote(password, safe='')}@{self.db_host}:{self.db_port}/"
                f"{db_name or self.db_name}")

    @property
    def mcp_url(self) -> str:
        return f"http://{self.mcp_host}:{self.mcp_port}/mcp"


def describe_validation_error(error: ValidationError) -> str:
    """Field names and messages only: input values are left out because some of them are secrets."""
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                     for e in error.errors(include_input=False, include_url=False))

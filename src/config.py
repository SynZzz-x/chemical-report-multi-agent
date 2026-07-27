"""Runtime configuration helpers.

All secrets must come from environment variables. This module keeps startup
errors actionable and prevents source files from carrying private keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v1"


@dataclass(frozen=True)
class AppConfig:
    """Provider configuration shared by every model consumer."""

    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    dashscope_api_key: str | None
    dashscope_embedding_model: str

def get_cache_root() -> Path:
    """
    返回项目缓存根目录。

    可通过 AGENT_CACHE_ROOT 指定自定义目录；
    未配置时默认使用项目根目录下的 cache。
    """
    configured = get_env("AGENT_CACHE_ROOT")

    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent
        root = project_root / "cache"

    root.mkdir(parents=True, exist_ok=True)
    return root


def get_local_user_id() -> str:
    """Return the stable local identity used to namespace persisted jobs."""

    return get_env("AGENT_USER_ID", "local-user") or "local-user"


def get_langgraph_data_dir() -> Path:
    """Return the private runtime directory for LangGraph SQLite files."""

    directory = get_cache_root() / "langgraph"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Load model-provider settings once for consistent process-wide access."""

    return AppConfig(
        deepseek_api_key=get_env("DEEPSEEK_API_KEY"),
        deepseek_base_url=(
            get_env("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
            or DEFAULT_DEEPSEEK_BASE_URL
        ),
        deepseek_model=(
            get_env("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
            or DEFAULT_DEEPSEEK_MODEL
        ),
        dashscope_api_key=get_env("DASHSCOPE_API_KEY"),
        dashscope_embedding_model=(
            get_env(
                "DASHSCOPE_EMBEDDING_MODEL",
                DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
            )
            or DEFAULT_DASHSCOPE_EMBEDDING_MODEL
        ),
    )


def missing_key_message(name: str) -> str:
    return (
        f"Missing required environment variable {name}. "
        "Copy .env.example to .env, fill in your local credentials, and export "
        "the variables before starting the app."
    )


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise RuntimeError(missing_key_message(name))
    return value


def _number(configurable: dict[str, Any], name: str, default: float | int) -> float | int:
    value = configurable.get(name, default)
    try:
        if isinstance(default, int):
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return default


def get_llm_settings(configurable: dict[str, Any] | None = None) -> dict[str, Any]:
    configurable = configurable or {}
    app_config = get_app_config()
    return {
        "api_key": app_config.deepseek_api_key,
        "base_url": app_config.deepseek_base_url,
        "model_name": app_config.deepseek_model,
        "max_tokens": _number(configurable, "max_tokens", 2048),
        "temperature": _number(configurable, "temperature", 0.2),
        "top_p": _number(configurable, "top_p", 1.0),
        "frequency_penalty": _number(configurable, "frequency_penalty", 0.0),
        "presence_penalty": _number(configurable, "presence_penalty", 0.0),
    }


def configure_langsmith_from_env() -> None:
    tracing = get_env("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_TRACING"] = tracing
    for name in ("LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY"):
        value = get_env(name)
        if value:
            os.environ[name] = value

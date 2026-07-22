"""Runtime configuration helpers.

All secrets must come from environment variables. This module keeps startup
errors actionable and prevents source files from carrying private keys.
"""

from __future__ import annotations

import os
from typing import Any
from pathlib import Path

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
DEFAULT_MODEL = "deepseek-v3"

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


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


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
    return {
        "api_key": configurable.get("api_key") or get_env("OPENAI_API_KEY"),
        "base_url": configurable.get("base_url") or get_env("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        "model_name": configurable.get("model_name") or get_env("OPENAI_MODEL", DEFAULT_MODEL),
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

"""Runtime configuration helpers.

All secrets must come from environment variables. This module keeps startup
errors actionable and prevents source files from carrying private keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_MODEL_REVISION = "66e95e324bebb9453d3b5be447c898dca1ba0eb0"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 30.0
DEFAULT_CHILD_TARGET_TOKENS = 450
DEFAULT_CHILD_MAX_TOKENS = 700
DEFAULT_CHILD_OVERLAP_TOKENS = 70
DEFAULT_PARENT_TARGET_TOKENS = 1200
DEFAULT_PARENT_MAX_TOKENS = 1600
DEFAULT_BM25_TOP_K = 40
DEFAULT_DENSE_TOP_K = 40
DEFAULT_DENSE_OVERFETCH_FACTOR = 3
DEFAULT_RRF_CHILD_TOP_K = 12
DEFAULT_FINAL_TOP_K = 5
DEFAULT_MAX_HITS_PER_PARENT = 2
DEFAULT_RRF_K = 60
DEFAULT_MAX_CONTEXT_TOKENS = 5000


@dataclass(frozen=True)
class RAGSettings:
    """Immutable configuration for the hybrid chemical RAG pipeline."""

    embedding_base_url: str
    embedding_api_key: str | None = field(repr=False)
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    embedding_timeout_seconds: float
    child_target_tokens: int
    child_max_tokens: int
    child_overlap_tokens: int
    parent_target_tokens: int
    parent_max_tokens: int
    bm25_top_k: int
    dense_top_k: int
    dense_overfetch_factor: int
    rrf_child_top_k: int
    final_top_k: int
    max_hits_per_parent: int
    rrf_k: int
    max_context_tokens: int
    storage_root: Path


@dataclass(frozen=True)
class AppConfig:
    """Provider configuration shared by every model consumer."""

    deepseek_api_key: str | None = field(repr=False)
    deepseek_base_url: str
    deepseek_model: str
    rag_settings: RAGSettings

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


def _positive_int_from_env(name: str, default: int) -> int:
    value = get_env(name, str(default))
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _non_negative_int_from_env(name: str, default: int) -> int:
    value = get_env(name, str(default))
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _positive_float_from_env(name: str, default: float) -> float:
    value = get_env(name, str(default))
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return parsed


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Load model-provider settings once for consistent process-wide access."""

    child_target_tokens = _positive_int_from_env(
        "RAG_CHILD_TARGET_TOKENS", DEFAULT_CHILD_TARGET_TOKENS
    )
    child_max_tokens = _positive_int_from_env(
        "RAG_CHILD_MAX_TOKENS", DEFAULT_CHILD_MAX_TOKENS
    )
    child_overlap_tokens = _non_negative_int_from_env(
        "RAG_CHILD_OVERLAP_TOKENS", DEFAULT_CHILD_OVERLAP_TOKENS
    )
    parent_target_tokens = _positive_int_from_env(
        "RAG_PARENT_TARGET_TOKENS", DEFAULT_PARENT_TARGET_TOKENS
    )
    parent_max_tokens = _positive_int_from_env(
        "RAG_PARENT_MAX_TOKENS", DEFAULT_PARENT_MAX_TOKENS
    )
    if not child_overlap_tokens < child_target_tokens < child_max_tokens:
        raise ValueError(
            "RAG chunk settings must satisfy 0 <= RAG_CHILD_OVERLAP_TOKENS "
            "< RAG_CHILD_TARGET_TOKENS < RAG_CHILD_MAX_TOKENS."
        )
    if parent_target_tokens > parent_max_tokens:
        raise ValueError(
            "RAG_PARENT_TARGET_TOKENS must be less than or equal to "
            "RAG_PARENT_MAX_TOKENS."
        )

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
        rag_settings=RAGSettings(
            embedding_base_url=(
                get_env("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
                or DEFAULT_EMBEDDING_BASE_URL
            ),
            embedding_api_key=get_env("EMBEDDING_API_KEY"),
            embedding_model=(
                get_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
                or DEFAULT_EMBEDDING_MODEL
            ),
            embedding_model_revision=(
                get_env("EMBEDDING_MODEL_REVISION", DEFAULT_EMBEDDING_MODEL_REVISION)
                or DEFAULT_EMBEDDING_MODEL_REVISION
            ),
            embedding_dimension=_positive_int_from_env(
                "EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION
            ),
            embedding_timeout_seconds=_positive_float_from_env(
                "EMBEDDING_TIMEOUT_SECONDS", DEFAULT_EMBEDDING_TIMEOUT_SECONDS
            ),
            child_target_tokens=child_target_tokens,
            child_max_tokens=child_max_tokens,
            child_overlap_tokens=child_overlap_tokens,
            parent_target_tokens=parent_target_tokens,
            parent_max_tokens=parent_max_tokens,
            bm25_top_k=_positive_int_from_env(
                "RAG_BM25_TOP_K", DEFAULT_BM25_TOP_K
            ),
            dense_top_k=_positive_int_from_env(
                "RAG_DENSE_TOP_K", DEFAULT_DENSE_TOP_K
            ),
            dense_overfetch_factor=_positive_int_from_env(
                "RAG_DENSE_OVERFETCH_FACTOR", DEFAULT_DENSE_OVERFETCH_FACTOR
            ),
            rrf_child_top_k=_positive_int_from_env(
                "RAG_RRF_CHILD_TOP_K", DEFAULT_RRF_CHILD_TOP_K
            ),
            final_top_k=_positive_int_from_env(
                "RAG_FINAL_TOP_K", DEFAULT_FINAL_TOP_K
            ),
            max_hits_per_parent=_positive_int_from_env(
                "RAG_MAX_HITS_PER_PARENT", DEFAULT_MAX_HITS_PER_PARENT
            ),
            rrf_k=_positive_int_from_env("RAG_RRF_K", DEFAULT_RRF_K),
            max_context_tokens=_positive_int_from_env(
                "RAG_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS
            ),
            storage_root=get_cache_root() / "rag",
        ),
    )


def get_rag_settings() -> RAGSettings:
    """Return the process-wide immutable RAG settings."""

    return get_app_config().rag_settings


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

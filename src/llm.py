from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from .config import get_llm_settings, missing_key_message


logger = logging.getLogger("src.llm.observability")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _log_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return re.sub(r"\s+", "_", str(value).strip())[:160] or "-"


def _model_name(runnable: Any) -> str:
    """Best-effort model identification without changing runnable behavior."""
    candidates = [runnable, getattr(runnable, "bound", None)]
    steps = getattr(runnable, "steps", None)
    if isinstance(steps, (list, tuple)):
        candidates.extend(reversed(steps))
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("model_name", "model"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "-"


def _token_value(mapping: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def extract_token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Read token usage from common LangChain/OpenAI-compatible response shapes."""
    try:
        usage = _mapping(getattr(response, "usage_metadata", None))
        if not usage:
            response_metadata = _mapping(
                getattr(response, "response_metadata", None)
            )
            usage = _mapping(
                response_metadata.get("token_usage")
                or response_metadata.get("usage")
            )
        input_tokens = _token_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _token_value(
            usage, "output_tokens", "completion_tokens"
        )
        total_tokens = _token_value(usage, "total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens
    except Exception:
        return None, None, None


def invoke_llm(
    runnable: Any,
    value: Any,
    *,
    config: RunnableConfig | None = None,
    node: str,
    purpose: str,
    attempt: int = 1,
    iteration: int | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
    plan_revision: int | None = None,
    task_revision: int | None = None,
    json_mode: bool | None = None,
) -> Any:
    """Invoke an LLM runnable with metadata-only request observability."""
    config_map = _mapping(config)
    metadata = _mapping(config_map.get("metadata"))
    configurable = _mapping(config_map.get("configurable"))
    resolved_job_id = job_id or metadata.get("job_id") or configurable.get("thread_id")
    thread_id = configurable.get("thread_id") or metadata.get("thread_id")
    call_id = uuid.uuid4().hex
    common = (
        f"call_id={_log_value(call_id)} "
        f"job_id={_log_value(resolved_job_id)} "
        f"thread_id={_log_value(thread_id)} "
        f"node={_log_value(node)} "
        f"purpose={_log_value(purpose)} "
        f"task_id={_log_value(task_id)} "
        f"plan_revision={_log_value(plan_revision)} "
        f"task_revision={_log_value(task_revision)} "
        f"attempt={_log_value(attempt)} "
        f"iteration={_log_value(iteration)} "
        f"model={_log_value(_model_name(runnable))} "
        f"json_mode={str(json_mode).lower() if json_mode is not None else '-'}"
    )
    logger.info("LLM_CALL_START %s", common)
    started = time.perf_counter()
    try:
        if config is None:
            response = runnable.invoke(value)
        else:
            response = runnable.invoke(value, config=config)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000)
        logger.warning(
            "LLM_CALL_ERROR %s latency_ms=%s error_type=%s",
            common,
            latency_ms,
            _log_value(type(exc).__name__),
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000)
    input_tokens, output_tokens, total_tokens = extract_token_usage(response)
    logger.info(
        "LLM_CALL_END %s status=ok latency_ms=%s input_tokens=%s "
        "output_tokens=%s total_tokens=%s",
        common,
        latency_ms,
        _log_value(input_tokens),
        _log_value(output_tokens),
        _log_value(total_tokens),
    )
    return response


def get_llm(config: RunnableConfig, json_mode: bool = True) -> ChatOpenAI:
    """Create a ChatOpenAI-compatible client from runtime configuration."""
    configurable = config.get("configurable", {}) if config else {}
    settings = get_llm_settings(configurable)
    api_key = settings.pop("api_key", None)
    if not api_key:
        raise RuntimeError(missing_key_message("DEEPSEEK_API_KEY"))

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        api_key=api_key,
        model_kwargs=model_kwargs,
        **settings,
    )

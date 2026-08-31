from __future__ import annotations

import logging
import re
import time
import uuid
from copy import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from openai import LengthFinishReasonError

from .config import get_app_config, get_llm_settings, missing_key_message


logger = logging.getLogger("src.llm.observability")


_PURPOSE_COMPLETION_BUDGETS = {
    "canonical_intake_generation": 1200,
    "assessment_contract_repair": 900,
    "concept_graph_extraction": 1200,
    "citation_binding": 1600,
    "report_synthesis": 4000,
    "report_evaluation": 800,
    "plan_patch": 1800,
    "feedback_analysis": 1000,
    "chart_description": 1000,
    "chart_description_fallback": 800,
    "table_description": 800,
}
_BUDGET_PARAMETER_NAMES = frozenset(
    {"max_tokens", "max_completion_tokens", "max_output_tokens"}
)


def _maximum_length(description: str | None) -> int | None:
    text = str(description or "").replace(",", "").replace("，", "")
    if re.search(r"(?:不少于|至少)\s*\d+\s*(?:个)?字", text):
        return None
    range_match = re.search(r"\d+\s*(?:-|–|—|~|～|至|到)\s*(\d+)\s*(?:个)?字", text)
    if range_match:
        return int(range_match.group(1))
    maximum_match = re.search(r"(?:不超过|至多|最多|压缩到不超过)\s*(\d+)\s*(?:个)?字", text)
    if maximum_match:
        return int(maximum_match.group(1))
    approximate = re.search(r"(?:约\s*)?(\d+)\s*(?:个)?字(?:左右)?", text)
    return int(approximate.group(1)) if approximate else None


def completion_budget(purpose: str, *, task_description: str | None = None) -> int:
    """Return a bounded output-token budget for one semantic purpose."""

    normalized = str(purpose or "").strip()
    if normalized == "plan_generation":
        return get_app_config().planner_max_completion_tokens
    if normalized == "assessment":
        return get_app_config().verifier_max_completion_tokens
    maximum = _maximum_length(task_description)
    if normalized == "task_generation":
        return max(1200, min(4096, (maximum * 2 + 800) if maximum else 3200))
    if normalized == "length_rewrite":
        return max(800, min(2400, (maximum * 3 // 2 + 600) if maximum else 1800))
    return _PURPOSE_COMPLETION_BUDGETS.get(normalized, 2048)


def _deepseek_extra_body(
    existing: Mapping[str, Any] | None, *, max_tokens: int
) -> dict[str, Any]:
    """Keep DeepSeek completion budgets in its supported request body."""

    return {
        **{
            key: value
            for key, value in dict(existing or {}).items()
            if key not in _BUDGET_PARAMETER_NAMES
        },
        "max_tokens": int(max_tokens),
    }


def _bound_request_layers(runnable: Any) -> tuple[list[Any], Any]:
    """Return RunnableBinding-like layers from outermost to innermost."""

    layers: list[Any] = []
    base = runnable
    while (
        isinstance(getattr(base, "kwargs", None), Mapping)
        and getattr(base, "bound", None) is not None
    ):
        layers.append(base)
        base = base.bound

    return layers, base


def _copy_binding(layer: Any, *, bound: Any, kwargs: Mapping[str, Any]) -> Any:
    """Copy a binding while retaining its concrete class and metadata."""

    model_copy = getattr(layer, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"bound": bound, "kwargs": dict(kwargs)})
    cloned = copy(layer)
    cloned.bound = bound
    cloned.kwargs = dict(kwargs)
    return cloned


def _with_deepseek_budget(runnable: Any, *, max_tokens: int) -> Any:
    """Apply one budget without dropping binding metadata or request options."""

    layers, base = _bound_request_layers(runnable)
    extra_body = dict(getattr(base, "extra_body", None) or {})
    for layer in reversed(layers):
        extra_body.update(dict(getattr(layer, "kwargs", {}).get("extra_body") or {}))
    extra_body = _deepseek_extra_body(extra_body, max_tokens=max_tokens)

    if not layers:
        return base.bind(extra_body=extra_body)

    rebound = base
    for position, layer in reversed(list(enumerate(layers))):
        layer_kwargs = {
            key: value
            for key, value in dict(layer.kwargs).items()
            if key not in _BUDGET_PARAMETER_NAMES
        }
        if "extra_body" in layer_kwargs:
            layer_kwargs["extra_body"] = _deepseek_extra_body(
                _mapping(layer_kwargs["extra_body"]), max_tokens=max_tokens
            )
        if position == 0:
            layer_kwargs["extra_body"] = extra_body
        rebound = _copy_binding(layer, bound=rebound, kwargs=layer_kwargs)
    return rebound


def with_completion_budget(
    runnable: Any, purpose: str, *, task_description: str | None = None
) -> tuple[Any, int]:
    """Bind a budget to direct model/binding runnables without altering chains."""

    budget = completion_budget(purpose, task_description=task_description)
    binder = getattr(runnable, "bind", None)
    if (
        callable(binder)
        and _model_name(runnable) != "-"
        and not isinstance(getattr(runnable, "steps", None), (list, tuple))
    ):
        return _with_deepseek_budget(runnable, max_tokens=budget), budget
    return runnable, budget


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


@dataclass(frozen=True)
class ProviderTokenUsage:
    """Provider-reported token usage with its original field semantics."""

    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_reasoning_tokens: int | None = None
    provider_total_tokens: int | None = None


def extract_provider_token_usage(response: Any) -> ProviderTokenUsage:
    """Read named usage fields from common LangChain/provider response shapes."""

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
        prompt_tokens = _token_value(usage, "input_tokens", "prompt_tokens")
        completion_tokens = _token_value(
            usage, "output_tokens", "completion_tokens"
        )
        total_tokens = _token_value(usage, "total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        output_details = _mapping(usage.get("output_token_details"))
        completion_details = _mapping(usage.get("completion_tokens_details"))
        reasoning_tokens = _token_value(usage, "reasoning_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = _token_value(
                output_details, "reasoning", "reasoning_tokens"
            )
        if reasoning_tokens is None:
            reasoning_tokens = _token_value(
                completion_details, "reasoning_tokens", "reasoning"
            )
        return ProviderTokenUsage(
            provider_prompt_tokens=prompt_tokens,
            provider_completion_tokens=completion_tokens,
            provider_reasoning_tokens=reasoning_tokens,
            provider_total_tokens=total_tokens,
        )
    except Exception:
        return ProviderTokenUsage()


def extract_truncation_token_usage(error: LengthFinishReasonError) -> ProviderTokenUsage:
    """Read SDK usage only, without inspecting content or inferring missing tokens."""

    try:
        usage = getattr(error.completion, "usage", None)
        if callable(getattr(usage, "model_dump", None)):
            usage = usage.model_dump()
        usage = _mapping(usage)
        details = _mapping(usage.get("completion_tokens_details"))
        return ProviderTokenUsage(
            provider_prompt_tokens=_token_value(usage, "prompt_tokens"),
            provider_completion_tokens=_token_value(usage, "completion_tokens"),
            provider_reasoning_tokens=_token_value(details, "reasoning_tokens"),
            provider_total_tokens=_token_value(usage, "total_tokens"),
        )
    except Exception:
        # Telemetry must never hide or replace the original SDK failure.
        return ProviderTokenUsage()


def extract_token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Compatibility adapter for the original input/output/total tuple."""

    usage = extract_provider_token_usage(response)
    return (
        usage.provider_prompt_tokens,
        usage.provider_completion_tokens,
        usage.provider_total_tokens,
    )


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
    max_completion_tokens: int | None = None,
) -> Any:
    """Invoke an LLM runnable with metadata-only request observability."""
    config_map = _mapping(config)
    metadata = _mapping(config_map.get("metadata"))
    configurable = _mapping(config_map.get("configurable"))
    resolved_job_id = job_id or metadata.get("job_id") or configurable.get("thread_id")
    thread_id = configurable.get("thread_id") or metadata.get("thread_id")
    if max_completion_tokens is None:
        max_completion_tokens = completion_budget(purpose)
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
        f"requested_max_completion_tokens={_log_value(max_completion_tokens)} "
        f"max_completion_tokens={_log_value(max_completion_tokens)} "
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
        usage_fields = ""
        if isinstance(exc, LengthFinishReasonError):
            usage = extract_truncation_token_usage(exc)
            usage_fields = (
                f" provider_prompt_tokens={_log_value(usage.provider_prompt_tokens)}"
                f" provider_completion_tokens={_log_value(usage.provider_completion_tokens)}"
                f" provider_reasoning_tokens={_log_value(usage.provider_reasoning_tokens)}"
                f" provider_total_tokens={_log_value(usage.provider_total_tokens)}"
            )
        logger.warning(
            "LLM_CALL_ERROR %s latency_ms=%s error_type=%s%s",
            common,
            latency_ms,
            _log_value(type(exc).__name__),
            usage_fields,
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000)
    usage = extract_provider_token_usage(response)
    # Provider completion usage may include provider-accounted reasoning and is
    # therefore not guaranteed to equal the visible response-text token count.
    logger.info(
        "LLM_CALL_END %s status=ok latency_ms=%s provider_prompt_tokens=%s "
        "provider_completion_tokens=%s provider_reasoning_tokens=%s "
        "provider_total_tokens=%s input_tokens=%s output_tokens=%s total_tokens=%s",
        common,
        latency_ms,
        _log_value(usage.provider_prompt_tokens),
        _log_value(usage.provider_completion_tokens),
        _log_value(usage.provider_reasoning_tokens),
        _log_value(usage.provider_total_tokens),
        _log_value(usage.provider_prompt_tokens),
        _log_value(usage.provider_completion_tokens),
        _log_value(usage.provider_total_tokens),
    )
    return response


def get_llm(
    config: RunnableConfig,
    json_mode: bool = True,
    *,
    purpose: str | None = None,
    task_description: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI-compatible client from runtime configuration."""
    configurable = config.get("configurable", {}) if config else {}
    settings = get_llm_settings(configurable)
    api_key = settings.pop("api_key", None)
    if not api_key:
        raise RuntimeError(missing_key_message("DEEPSEEK_API_KEY"))

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    budget = settings.pop("max_tokens", 2048)
    if purpose:
        budget = completion_budget(purpose, task_description=task_description)
    if purpose == "assessment":
        verifier_settings = get_app_config()
        if verifier_settings.verifier_model:
            settings["model_name"] = verifier_settings.verifier_model
        if verifier_settings.verifier_reasoning_effort:
            settings["reasoning_effort"] = verifier_settings.verifier_reasoning_effort
    settings["extra_body"] = _deepseek_extra_body(
        settings.pop("extra_body", None), max_tokens=int(budget)
    )

    return ChatOpenAI(
        api_key=api_key,
        model_kwargs=model_kwargs,
        **settings,
    )

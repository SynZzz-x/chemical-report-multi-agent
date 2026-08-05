"""Shared bounded LangGraph execution-budget helpers."""

from __future__ import annotations

from typing import Any, Mapping

from src.limits import MAX_PLAN_TASKS
from src.recovery.policy import (
    MAX_CONTENT_RETRIES,
    MAX_EVIDENCE_RECOVERIES,
    MAX_JOB_PATCHES,
    MAX_VERIFIER_RETRIES,
)


_FIXED_SUPERSTEPS = 8
_BASE_TASK_SUPERSTEPS = 3  # Worker -> Verifier -> DecisionPolicy
_CONTENT_RETRY_SUPERSTEPS = 3
_EVIDENCE_RECOVERY_SUPERSTEPS = 4
_VERIFIER_RETRY_SUPERSTEPS = 2
_PLAN_PATCH_SUPERSTEPS = 4


def recursion_limit_for_tasks(task_count: int | None = None) -> int:
    """Return the fixed budget covering the largest accepted bounded plan."""
    tasks = MAX_PLAN_TASKS
    estimated = (
        _FIXED_SUPERSTEPS
        + tasks
        * (
            _BASE_TASK_SUPERSTEPS
            + _CONTENT_RETRY_SUPERSTEPS * MAX_CONTENT_RETRIES
            + _EVIDENCE_RECOVERY_SUPERSTEPS * MAX_EVIDENCE_RECOVERIES
            + _VERIFIER_RETRY_SUPERSTEPS * MAX_VERIFIER_RETRIES
        )
        + _PLAN_PATCH_SUPERSTEPS * MAX_JOB_PATCHES
    )
    margin = max(16, tasks * 2)
    return estimated + margin


def execution_config(
    config: Mapping[str, Any], state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Copy a graph config and attach the shared task-aware recursion budget."""
    tasks = state.get("tasks") if isinstance(state, Mapping) else None
    task_count = len(tasks) if isinstance(tasks, list) else None
    return {**dict(config), "recursion_limit": recursion_limit_for_tasks(task_count)}

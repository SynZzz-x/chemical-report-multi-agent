"""Shared semantic checks for planned and restored task metadata."""

from __future__ import annotations

from typing import Any, Mapping


def task_allows_web(task: Mapping[str, Any]) -> bool:
    """Return whether task metadata explicitly authorizes public-web access."""
    visualization = task.get("visualization")
    return bool(
        task.get("use_web") is True
        or task.get("allow_web_fallback") is True
        or (
            isinstance(visualization, Mapping)
            and visualization.get("allow_web_fallback") is True
        )
    )


def effective_web_allowed(
    task: Mapping[str, Any], web_authorized: bool | None
) -> bool:
    """Apply the job-level web gate to a task's explicit source policy."""

    if web_authorized is False:
        return False
    return task_allows_web(task)

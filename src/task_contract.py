"""Shared semantic checks for planned and restored task metadata."""

from __future__ import annotations

from typing import Any, Mapping


def task_allows_web(
    task: Mapping[str, Any], *, job_authorized: bool = False
) -> bool:
    """Require both immutable job consent and task-scoped web metadata."""
    if job_authorized is not True:
        return False
    visualization = task.get("visualization")
    return bool(
        task.get("use_web") is True
        or task.get("allow_web_fallback") is True
        or (
            isinstance(visualization, Mapping)
            and visualization.get("allow_web_fallback") is True
        )
    )

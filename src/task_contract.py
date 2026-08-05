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

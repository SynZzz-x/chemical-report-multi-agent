"""Shared semantic checks for planned and restored task metadata."""

from __future__ import annotations

import re
from typing import Any, Mapping


_SYNTHESIS_SECTION_PATTERN = re.compile(
    r"^(?:(?:主要|总体|报告|全文|综合)?(?:结论|总结|结语)(?:与(?:建议|展望))?|conclusions?)$",
    re.IGNORECASE,
)


def _normalized_section_title(value: Any) -> str:
    text = re.sub(r"^\s*#{1,6}\s*", "", str(value or "").strip())
    return re.sub(
        r"^\s*(?:第?[一二三四五六七八九十百千万\d]+[章节、.．:]|[（(]?[一二三四五六七八九十\d]+[）).、．])\s*",
        "",
        text,
    ).casefold()


def is_synthesis_section(task: Mapping[str, Any]) -> bool:
    """Recognize only controlled conclusion/summary aggregate titles."""

    titles = [task.get("task_name"), *(task.get("covers_sections") or [])]
    return any(
        _SYNTHESIS_SECTION_PATTERN.fullmatch(_normalized_section_title(value))
        for value in titles
    )


def synthesis_semantic_error(
    task: Mapping[str, Any], *, has_prior_task: bool
) -> str | None:
    """Return a shared Planner/PlanPatch synthesis-contract violation."""

    aggregate = is_synthesis_section(task)
    synthesis = task.get("task_type") == "synthesis"
    task_id = task.get("task_id")
    if aggregate and not synthesis:
        return (
            f"task {task_id} conclusion/summary section must use "
            "task_type=synthesis"
        )
    if synthesis and not has_prior_task:
        return f"task {task_id} synthesis requires at least one prior task"
    if synthesis and not aggregate:
        return (
            f"task {task_id} task_type=synthesis is reserved for "
            "conclusion/summary aggregate sections"
        )
    return None


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

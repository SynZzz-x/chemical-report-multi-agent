"""Deterministic section acceptance and report admission rules."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


VERIFIED_PASS = "VERIFIED_PASS"
USER_ACCEPTED_GAP = "USER_ACCEPTED_GAP"
USER_ACCEPTED_WARNING = "USER_ACCEPTED_WARNING"
ACCEPT_WITH_WARNING = "ACCEPT_WITH_WARNING"
BLOCKED = "BLOCKED"
EXTERNAL_BLOCKER = "EXTERNAL_BLOCKER"

READY_FOR_FINAL = "READY_FOR_FINAL"
DRAFT_WITH_GAPS = "DRAFT_WITH_GAPS"

_DRAFT_STATUSES = {USER_ACCEPTED_GAP, USER_ACCEPTED_WARNING}
_BLOCKING_STATUSES = {ACCEPT_WITH_WARNING, BLOCKED, EXTERNAL_BLOCKER}


def _task_ids(tasks: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(task["task_id"])
        for task in tasks
        if isinstance(task, Mapping) and task.get("task_id") is not None
    ]


def _current_task_id(state: Mapping[str, Any]) -> str:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], Mapping):
        task_id = tasks[cursor].get("task_id")
        if task_id is not None:
            return str(task_id)
    current_result = state.get("current_result") or {}
    task_id = current_result.get("task_id") if isinstance(current_result, Mapping) else None
    return str(task_id if task_id is not None else cursor)


def record_section_status(
    state: Mapping[str, Any],
    status: str,
    *,
    accepted_by: str,
    issues: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a copied section-status map with the current task updated."""

    task_id = _current_task_id(state)
    task_revisions = state.get("task_revisions") or {}
    try:
        task_revision = int(task_revisions.get(task_id, 1) or 1)
    except (AttributeError, TypeError, ValueError):
        task_revision = 1
    statuses = {
        str(key): dict(value)
        for key, value in (state.get("section_status") or {}).items()
        if isinstance(value, Mapping)
    }
    statuses[task_id] = {
        "status": str(status),
        "accepted_by": str(accepted_by),
        "issues": [dict(issue) for issue in (issues or []) if isinstance(issue, Mapping)],
        "plan_revision": int(state.get("plan_revision", 1) or 1),
        "task_revision": task_revision,
    }
    return statuses


def derive_report_status(
    tasks: Iterable[Mapping[str, Any]],
    section_status: Mapping[str, Mapping[str, Any]] | None,
) -> str:
    """Derive one job-level report status without trusting committed results."""

    task_ids = _task_ids(tasks)
    statuses = section_status or {}
    if not task_ids:
        return BLOCKED

    values: list[str] = []
    for task_id in task_ids:
        entry = statuses.get(task_id)
        if not isinstance(entry, Mapping):
            return BLOCKED
        status = str(entry.get("status") or "")
        if not status or status in _BLOCKING_STATUSES:
            return BLOCKED
        values.append(status)

    if all(status == VERIFIED_PASS for status in values):
        return READY_FOR_FINAL
    if all(status == VERIFIED_PASS or status in _DRAFT_STATUSES for status in values):
        return DRAFT_WITH_GAPS
    return BLOCKED


def eligible_task_ids(
    tasks: Iterable[Mapping[str, Any]],
    section_status: Mapping[str, Mapping[str, Any]] | None,
    report_status: str,
) -> list[str]:
    """Return report-admitted task IDs in Planner order."""

    if report_status == READY_FOR_FINAL:
        allowed = {VERIFIED_PASS}
    elif report_status == DRAFT_WITH_GAPS:
        allowed = {VERIFIED_PASS, *_DRAFT_STATUSES}
    else:
        return []
    statuses = section_status or {}
    return [
        task_id
        for task_id in _task_ids(tasks)
        if isinstance(statuses.get(task_id), Mapping)
        and str(statuses[task_id].get("status") or "") in allowed
    ]

"""Task-revision-scoped acceptance of evidence availability gaps."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


WAIVABLE_EVIDENCE_GAP_CODES = {
    "EVIDENCE_GAP",
    "INSUFFICIENT_EVIDENCE",
    "MISSING_EVIDENCE",
    "RAG_COVERAGE_GAP",
    "RAG_INSUFFICIENT",
}


def current_task_id(state: Mapping[str, Any]) -> str:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], Mapping):
        task_id = tasks[cursor].get("task_id")
        if task_id is not None:
            return str(task_id)
    current_result = state.get("current_result") or {}
    if isinstance(current_result, Mapping) and current_result.get("task_id") is not None:
        return str(current_result["task_id"])
    return str(cursor)


def current_task_scope(state: Mapping[str, Any]) -> tuple[str, int, int]:
    task_id = current_task_id(state)
    revisions = state.get("task_revisions") or {}
    try:
        task_revision = int(revisions.get(task_id, 1) or 1)
    except (AttributeError, TypeError, ValueError):
        task_revision = 1
    return (
        task_id,
        int(state.get("plan_revision", 1) or 1),
        task_revision,
    )


def is_waivable_evidence_gap(issue: Mapping[str, Any]) -> bool:
    return str(issue.get("code") or "").strip().upper() in WAIVABLE_EVIDENCE_GAP_CODES


def split_waivable_evidence_gaps(
    issues: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        target = accepted if is_waivable_evidence_gap(issue) else remaining
        target.append(dict(issue))
    return accepted, remaining


def record_evidence_gap_acceptance(
    state: Mapping[str, Any], issues: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    task_id, plan_revision, task_revision = current_task_scope(state)
    accepted, _ = split_waivable_evidence_gaps(issues)
    waivers = {
        str(key): deepcopy(value)
        for key, value in (state.get("accepted_evidence_gaps") or {}).items()
        if isinstance(value, Mapping)
    }
    waivers[task_id] = {
        "plan_revision": plan_revision,
        "task_revision": task_revision,
        "issues": accepted,
    }
    return waivers


def matching_evidence_gap_acceptance(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    task_id, plan_revision, task_revision = current_task_scope(state)
    record = (state.get("accepted_evidence_gaps") or {}).get(task_id)
    if not isinstance(record, Mapping):
        return None
    try:
        record_plan_revision = int(record.get("plan_revision", 0) or 0)
        record_task_revision = int(record.get("task_revision", 0) or 0)
    except (TypeError, ValueError):
        return None
    if (
        record_plan_revision != plan_revision
        or record_task_revision != task_revision
    ):
        return None
    return deepcopy(dict(record))


def apply_evidence_gap_acceptance(
    assessment: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    updated = deepcopy(dict(assessment))
    if matching_evidence_gap_acceptance(state) is None:
        return updated
    waived, remaining = split_waivable_evidence_gaps(updated.get("issues") or [])
    if not waived:
        return updated
    updated["issues"] = remaining
    updated["waived_evidence_issues"] = waived
    if remaining:
        updated["status"] = "FAILED"
    else:
        updated["status"] = "PASS"
        updated["requirements_missing"] = []
    return updated

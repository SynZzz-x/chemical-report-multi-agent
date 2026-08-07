"""Pure, bounded policy for recovery after a task assessment."""

from __future__ import annotations

from enum import Enum
from os.path import basename
from typing import Any, Dict, Iterable, List

from src.workflow_records import ensure_task_records, set_task_status


class WorkflowAction(str, Enum):
    PASS = "PASS"
    NEXT = "NEXT"
    DONE = "DONE"
    REWORK = "REWORK"
    EVIDENCE_RECOVERY = "EVIDENCE_RECOVERY"
    PLAN_PATCH = "PLAN_PATCH"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    RETRY_VERIFIER = "RETRY_VERIFIER"


class IssueCategory(str, Enum):
    CONTENT_DEFECT = "CONTENT_DEFECT"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    DATA_DEFECT = "DATA_DEFECT"
    VISUAL_DEFECT = "VISUAL_DEFECT"
    WORKER_FAILURE = "WORKER_FAILURE"
    LOCAL_PLAN_DEFECT = "LOCAL_PLAN_DEFECT"
    SAFETY_BOUNDARY = "SAFETY_BOUNDARY"
    REQUIREMENT_MISSING = "REQUIREMENT_MISSING"
    EXTERNAL_BLOCKER = "EXTERNAL_BLOCKER"
    REVIEW_FAILURE = "REVIEW_FAILURE"
    # Legacy assessment compatibility. New QualityReview emits REVIEW_FAILURE.
    VERIFIER_FAILURE = "VERIFIER_FAILURE"


MAX_CONTENT_RETRIES = 2
MAX_EVIDENCE_RECOVERIES = 1
MAX_TASK_PATCHES = 1
MAX_JOB_PATCHES = 3
MAX_VERIFIER_RETRIES = 1

_CATEGORY_PRIORITY = {
    IssueCategory.CONTENT_DEFECT: 0,
    IssueCategory.DATA_DEFECT: 0,
    IssueCategory.VISUAL_DEFECT: 0,
    IssueCategory.WORKER_FAILURE: 0,
    IssueCategory.REQUIREMENT_MISSING: 0,
    IssueCategory.EVIDENCE_GAP: 1,
    IssueCategory.LOCAL_PLAN_DEFECT: 2,
    IssueCategory.EXTERNAL_BLOCKER: 3,
    IssueCategory.SAFETY_BOUNDARY: 4,
    IssueCategory.VERIFIER_FAILURE: 5,
    IssueCategory.REVIEW_FAILURE: 5,
}

_CONTENT_CODES = {
    "CONTENT_DEFECT",
    "FORMAT_ERROR",
    "INCOMPLETE_CONTENT",
    "MISSING_FIGURE",
    "MISSING_TABLE",
    "TOO_SHORT",
    "REQUIREMENT_MISSING",
}
_VERIFIER_CODES = {
    "ASSESSMENT_CONTRACT_ERROR",
    "LLM_ERROR",
    "LLM_NOT_ENABLED",
    "VERIFIER_ERROR",
    "VERIFIER_SERVICE_ERROR",
}
_EVIDENCE_CODES = {
    "EVIDENCE_GAP",
    "INSUFFICIENT_EVIDENCE",
    "MISSING_CITATION",
    "MISSING_EVIDENCE",
    "RAG_COVERAGE_GAP",
    "RAG_INSUFFICIENT",
    "SOURCE_UNSUPPORTED",
}
_LOCAL_PLAN_CODES = {
    "INVALID_TASK_ORDER",
    "MISSING_DEPENDENCY",
    "RESOURCE_NOT_ASSIGNED",
    "TASK_GRANULARITY",
}
_EXTERNAL_CODES = {
    "BAD_PLAN",
    "CONTRADICTORY_REQUIREMENTS",
    "EXTERNAL_BLOCKER",
    "INVALID_PLAN",
    "PERMISSION_DENIED",
    "REQUIREMENTS_CONFLICT",
    "RESOURCE_UNAVAILABLE",
    "UNEXECUTABLE_TASK",
}


def _current_task(state: Dict[str, Any]) -> Dict[str, Any]:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], dict):
        return tasks[cursor]
    return state.get("current_task") or {}


def _current_task_id(state: Dict[str, Any]) -> str:
    task = _current_task(state)
    task_id = task.get("task_id") or (state.get("current_result") or {}).get("task_id")
    if task_id is not None:
        return str(task_id)
    return str(int(state.get("cursor", 0) or 0))


def _normalise_counter(counter: Any, state: Dict[str, Any]) -> Dict[str, int]:
    """Convert old cursor-keyed checkpoints into stable task-id keyed counts."""
    normalized: Dict[str, int] = {}
    if not isinstance(counter, dict):
        return normalized
    tasks = state.get("tasks") or []
    active_task_ids = {
        str(task.get("task_id"))
        for task in tasks
        if isinstance(task, dict) and task.get("task_id") is not None
    }
    for key, value in counter.items():
        task_id = None
        cursor_key = None
        if isinstance(key, str) and key in active_task_ids:
            task_id = key
        elif isinstance(key, int) and not isinstance(key, bool):
            cursor_key = key
        elif isinstance(key, str) and key.isdecimal():
            cursor_key = int(key)
        if task_id is None and cursor_key is not None and 0 <= cursor_key < len(tasks):
            task = tasks[cursor_key]
            if isinstance(task, dict) and task.get("task_id") is not None:
                task_id = str(task["task_id"])
        if task_id is None:
            task_id = str(key)
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        normalized[task_id] = max(normalized.get(task_id, 0), count)
    return normalized


def _counter_update(state: Dict[str, Any], field: str) -> Dict[str, int]:
    return _normalise_counter(state.get(field), state)


def _as_strings(values: Iterable[Any]) -> set[str]:
    identifiers: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            identifiers.add(text)
            identifiers.add(basename(text))
    return identifiers


def _resource_identifiers(resource: Dict[str, Any]) -> set[str]:
    return _as_strings(
        resource.get(key)
        for key in ("name", "path", "file_id", "resource_id")
    )


def _missing_resource_category(issue: Dict[str, Any], state: Dict[str, Any]) -> IssueCategory:
    requested = str(issue.get("resource_name") or "").strip()
    if not requested:
        return IssueCategory.EXTERNAL_BLOCKER

    matching_resource = next(
        (
            resource
            for resource in state.get("docs") or []
            if isinstance(resource, dict) and requested in _resource_identifiers(resource)
        ),
        None,
    )
    if matching_resource is None:
        return IssueCategory.EXTERNAL_BLOCKER

    assigned = _as_strings(_current_task(state).get("use_resources") or [])
    if _resource_identifiers(matching_resource) & assigned:
        return IssueCategory.CONTENT_DEFECT
    return IssueCategory.LOCAL_PLAN_DEFECT


def _classify_issue(issue: Dict[str, Any], state: Dict[str, Any]) -> IssueCategory:
    code = str(issue.get("code") or "").strip().upper()
    if code == "MISSING_RESOURCE":
        return _missing_resource_category(issue, state)
    if code in _VERIFIER_CODES:
        return (
            IssueCategory.REVIEW_FAILURE
            if str(issue.get("category") or "").strip().upper()
            == IssueCategory.REVIEW_FAILURE.value
            else IssueCategory.VERIFIER_FAILURE
        )
    if code in _EVIDENCE_CODES:
        return IssueCategory.EVIDENCE_GAP
    if code in _EXTERNAL_CODES:
        return IssueCategory.EXTERNAL_BLOCKER
    if code in _LOCAL_PLAN_CODES:
        return IssueCategory.LOCAL_PLAN_DEFECT
    if code in _CONTENT_CODES:
        return IssueCategory.CONTENT_DEFECT

    try:
        code_category = IssueCategory(code)
    except ValueError:
        code_category = None
    if code_category in {
        IssueCategory.DATA_DEFECT,
        IssueCategory.VISUAL_DEFECT,
        IssueCategory.WORKER_FAILURE,
        IssueCategory.SAFETY_BOUNDARY,
        IssueCategory.REQUIREMENT_MISSING,
        IssueCategory.REVIEW_FAILURE,
        IssueCategory.VERIFIER_FAILURE,
    }:
        return code_category

    category = str(issue.get("category") or "").strip().upper()
    if category == IssueCategory.LOCAL_PLAN_DEFECT.value:
        return IssueCategory.EXTERNAL_BLOCKER
    try:
        return IssueCategory(category)
    except ValueError:
        return IssueCategory.CONTENT_DEFECT


def classify_assessment(assessment: Dict[str, Any], state: Dict[str, Any]) -> IssueCategory:
    """Return the highest-priority deterministic category in an assessment."""
    categories = [
        _classify_issue(issue, state)
        for issue in assessment.get("issues") or []
        if isinstance(issue, dict)
    ]
    return max(categories or [IssueCategory.CONTENT_DEFECT], key=_CATEGORY_PRIORITY.get)


def commit_current_result(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Replace-or-append the active Artifact in the compatibility result list."""
    results = list(state.get("results") or [])
    current_result = state.get("current_result") or {}
    if not current_result:
        return results

    task_id = _current_task_id(state)
    committed = dict(current_result)
    committed["task_id"] = task_id
    task_revisions = _normalise_counter(state.get("task_revisions"), state)
    committed.setdefault("task_revision", task_revisions.get(task_id, 1))
    committed.setdefault("plan_revision", int(state.get("plan_revision", 1) or 1))
    existing_index = next(
        (
            index
            for index, result in enumerate(results)
            if isinstance(result, dict) and str(result.get("task_id")) == task_id
        ),
        None,
    )
    if existing_index is None:
        results.append(committed)
    else:
        results[existing_index] = committed
    return results


def _pending_user_action(
    category: IssueCategory,
    state: Dict[str, Any],
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "category": category.value,
        "task_id": _current_task_id(state),
        "artifact_id": assessment.get("artifact_id")
        or (state.get("current_result") or {}).get("artifact_id"),
        "issues": list(assessment.get("issues") or []),
    }


def _task_records(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return ensure_task_records(state)


def _block_task(
    update: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
    task_id: str,
    category: IssueCategory,
    state: Dict[str, Any],
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    update.update(
        {
            "workflow_action": WorkflowAction.NEEDS_USER_INPUT.value,
            "task_records": set_task_status(records, task_id, "BLOCKED"),
            "pending_user_action": _pending_user_action(
                category, state, assessment
            ),
        }
    )
    return update


def decide_recovery_action(state: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Choose a bounded recovery action without invoking models or graph nodes."""
    task_id = _current_task_id(state)
    task_retry_count = _counter_update(state, "task_retry_count")
    evidence_recovery_count = _counter_update(state, "evidence_recovery_count")
    task_patch_count = _counter_update(state, "task_patch_count")
    verifier_retry_count = _counter_update(state, "verifier_retry_count")
    job_patch_count = int(state.get("job_patch_count", 0) or 0)
    records = _task_records(state)
    update: Dict[str, Any] = {
        "task_retry_count": task_retry_count,
        "evidence_recovery_count": evidence_recovery_count,
        "task_patch_count": task_patch_count,
        "verifier_retry_count": verifier_retry_count,
        "job_patch_count": job_patch_count,
        "pending_user_action": {},
        "verification_warnings": list(state.get("verification_warnings") or []),
    }

    if str(assessment.get("status") or "").upper() == "PASS":
        active_artifact_id = (
            assessment.get("artifact_id")
            or (state.get("active_artifact_ids") or {}).get(task_id)
            or (state.get("current_result") or {}).get("artifact_id")
        )
        update["workflow_action"] = WorkflowAction.NEXT.value
        update["task_records"] = set_task_status(
            records,
            task_id,
            "PASSED",
            active_artifact_id=active_artifact_id,
        )
        update["results"] = commit_current_result(state)
        return update

    category = classify_assessment(assessment, state)
    if category in {IssueCategory.REVIEW_FAILURE, IssueCategory.VERIFIER_FAILURE}:
        retries = verifier_retry_count.get(task_id, 0)
        if retries < MAX_VERIFIER_RETRIES:
            verifier_retry_count[task_id] = retries + 1
            update["workflow_action"] = WorkflowAction.RETRY_VERIFIER.value
            return update

    if category in {
        IssueCategory.CONTENT_DEFECT,
        IssueCategory.DATA_DEFECT,
        IssueCategory.VISUAL_DEFECT,
        IssueCategory.WORKER_FAILURE,
        IssueCategory.REQUIREMENT_MISSING,
    }:
        retries = task_retry_count.get(task_id, 0)
        if retries < MAX_CONTENT_RETRIES:
            task_retry_count[task_id] = retries + 1
            update["workflow_action"] = WorkflowAction.REWORK.value
            update["task_records"] = set_task_status(
                records, task_id, "REVISE_REQUIRED"
            )
            return update

    if category is IssueCategory.EVIDENCE_GAP:
        recoveries = evidence_recovery_count.get(task_id, 0)
        if recoveries < MAX_EVIDENCE_RECOVERIES:
            evidence_recovery_count[task_id] = recoveries + 1
            update["workflow_action"] = WorkflowAction.EVIDENCE_RECOVERY.value
            update["task_records"] = set_task_status(
                records, task_id, "EVIDENCE_REQUIRED"
            )
            return update

    if category is IssueCategory.LOCAL_PLAN_DEFECT:
        patches = task_patch_count.get(task_id, 0)
        if patches < MAX_TASK_PATCHES and job_patch_count < MAX_JOB_PATCHES:
            update["workflow_action"] = WorkflowAction.PLAN_PATCH.value
            update["task_records"] = set_task_status(
                records, task_id, "REVISE_REQUIRED"
            )
            return update

    return _block_task(
        update,
        records,
        task_id,
        category,
        state,
        assessment,
    )

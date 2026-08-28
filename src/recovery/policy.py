"""Pure, bounded policy for recovery after a task assessment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from enum import Enum
from hashlib import sha256
import logging
from os.path import basename
from typing import Any, Dict, List

from src.failure_semantics import (
    FailureAction,
    FailureClass,
    validate_failure_action_pair,
    normalize_failure_state,
)
from src.failure_registry import build_degraded_issue, upsert_degraded_issue
from src.blocker_registry import (
    affected_task_closure,
    build_user_blocker,
    runnable_task_ids,
    upsert_user_blocker,
)
from src.requirements import reconstruct_legacy_requirements
from src.evidence_waivers import (
    is_waivable_evidence_gap,
    matching_evidence_gap_acceptance,
)
from src.report_acceptance import (
    ACCEPT_WITH_WARNING,
    BLOCKED,
    EXTERNAL_BLOCKER,
    USER_ACCEPTED_GAP,
    VERIFIED_PASS,
    derive_report_status,
    record_section_status,
)
from src.report_validation import parse_length_target, safe_deterministic_trim
from src.tool_capabilities import public_web_runtime_available
from src.task_contract import effective_web_allowed


logger = logging.getLogger(__name__)


class WorkflowAction(str, Enum):
    PASS = "PASS"
    NEXT = "NEXT"
    DONE = "DONE"
    REWORK = "REWORK"
    ASSET_RECOVERY = "ASSET_RECOVERY"
    EVIDENCE_RECOVERY = "EVIDENCE_RECOVERY"
    PLAN_PATCH = "PLAN_PATCH"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    ACCEPT_WITH_WARNING = "ACCEPT_WITH_WARNING"
    RETRY_VERIFIER = "RETRY_VERIFIER"
    LENGTH_REWRITE = "LENGTH_REWRITE"
    SYNTHESIS_REWRITE = "SYNTHESIS_REWRITE"
    FATAL_SYSTEM = "FATAL_SYSTEM"
    CANCEL_JOB = "CANCEL_JOB"


class IssueCategory(str, Enum):
    CONTENT_DEFECT = "CONTENT_DEFECT"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    LOCAL_PLAN_DEFECT = "LOCAL_PLAN_DEFECT"
    EXTERNAL_BLOCKER = "EXTERNAL_BLOCKER"
    VERIFIER_FAILURE = "VERIFIER_FAILURE"


MAX_CONTENT_RETRIES = 2
MAX_ASSET_RETRIES = 1
MAX_EVIDENCE_RECOVERIES = 1
MAX_TASK_PATCHES = 1
MAX_JOB_PATCHES = 3
MAX_VERIFIER_RETRIES = 1

NEXT_RESUME_ALIASES = frozenset(
    {"NEXT", "CONTINUE", "继续", "带限制继续", "接受当前结果", "跳过"}
)

_EVIDENCE_USER_CHOICES = [
    "UPLOAD_RESOURCES",
    "AUTHORIZE_WEB",
    "ADJUST_REQUIREMENT",
    "ACCEPT_EVIDENCE_GAP",
]
_CATEGORY_PRIORITY = {
    IssueCategory.CONTENT_DEFECT: 0,
    IssueCategory.EVIDENCE_GAP: 1,
    IssueCategory.LOCAL_PLAN_DEFECT: 2,
    IssueCategory.EXTERNAL_BLOCKER: 3,
    IssueCategory.VERIFIER_FAILURE: 4,
}

_CONTENT_CODES = {
    "CONTENT_DEFECT",
    "FORMAT_ERROR",
    "INCOMPLETE_CONTENT",
    "MISSING_FIGURE",
    "MISSING_REQUIRED_FIGURE",
    "MISSING_TABLE",
    "TOO_SHORT",
    "TOO_LONG",
    "OUT_OF_SCOPE",
    "SCOPE_VIOLATION",
    "UNSUPPORTED_RECOMMENDATION",
    "REQUIREMENT_MISSING",
    "REPORT_SOURCE_INCONSISTENT",
}
_ASSET_ISSUE_CODES = {
    "MISSING_FIGURE",
    "MISSING_REQUIRED_FIGURE",
    "MISSING_TABLE",
}
_VERIFIER_CODES = {
    "ASSESSMENT_CONTRACT_ERROR",
    "LLM_ERROR",
    "LLM_NOT_ENABLED",
    "VERIFIER_UNAVAILABLE",
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
    "INVALID_CITATION_ID",
    "MISSING_INLINE_CITATION",
}
PLAN_PATCH_SUBTYPES = frozenset(
    {
        "MISSING_TASK",
        "INVALID_DEPENDENCY",
        "INVALID_TASK_ORDER",
        "MISSING_DEPENDENCY",
        "RESOURCE_NOT_ASSIGNED",
        "TASK_GRANULARITY",
        "UNEXECUTABLE_TASK",
        "UNEXECUTABLE_TASK_OBJECTIVE",
        "TASK_CONFLICT",
        "INVALID_TASK_DECOMPOSITION",
    }
)
_LOCAL_PLAN_CODES = PLAN_PATCH_SUBTYPES
_EXTERNAL_CODES = {
    "BAD_PLAN",
    "CONTRADICTORY_REQUIREMENTS",
    "EXTERNAL_BLOCKER",
    "INVALID_PLAN",
    "PERMISSION_DENIED",
    "REQUIREMENTS_CONFLICT",
    "RESOURCE_UNAVAILABLE",
}


def _has_authorized_evidence_retrieval(state: Dict[str, Any]) -> bool:
    task = _current_task(state)
    if "use_rag" not in task and "use_web" not in task:
        # Legacy checkpoints predate explicit source authorization fields and
        # historically resumed through the configured knowledge base.
        return True
    if task.get("use_rag") is True:
        return True
    web_authorized = (
        state.get("web_authorized")
        if isinstance(state.get("web_authorized"), bool)
        else None
    )
    return bool(
        public_web_runtime_available()
        and effective_web_allowed(task, web_authorized)
    )


def _current_task(state: Mapping[str, Any]) -> Mapping[str, Any]:
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


def _assessment_subtype(assessment: Mapping[str, Any], default: str) -> str:
    for issue in assessment.get("issues") or []:
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code") or "").strip().upper()
        if code:
            return code
    return default


def _requirement_scope(
    state: Mapping[str, Any], assessment: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """Resolve severity from explicit task/registry linkage, never issue prose."""

    task_requirement_ids = {
        str(requirement_id).strip()
        for requirement_id in _current_task(state).get("requirement_ids") or []
        if str(requirement_id).strip()
    }
    issue_requirement_ids = {
        str(requirement_id).strip()
        for issue in assessment.get("issues") or []
        if isinstance(issue, Mapping)
        for requirement_id in issue.get("requirement_ids") or []
        if str(requirement_id).strip()
    }
    current_task = _current_task(state)
    if str(current_task.get("task_type") or "").strip().lower() == "synthesis":
        affected = issue_requirement_ids
    else:
        affected = task_requirement_ids & issue_requirement_ids
    active_registry = {
        str(requirement.get("requirement_id") or "").strip(): requirement
        for requirement in state.get("requirement_registry") or []
        if isinstance(requirement, Mapping)
        and str(requirement.get("requirement_id") or "").strip()
        and str(requirement.get("status") or "active") == "active"
    }
    requirement_ids = sorted(
        requirement_id
        for requirement_id in affected
        if requirement_id in active_registry
    )
    hard_requirement_ids = [
        requirement_id
        for requirement_id in requirement_ids
        if str(active_registry[requirement_id].get("severity") or "soft") == "hard"
    ]
    return requirement_ids, hard_requirement_ids


def _failure_decision(
    state: Mapping[str, Any],
    assessment: Mapping[str, Any],
    *,
    failure_class: FailureClass,
    action: FailureAction,
    subtype: str | None = None,
    repair_attempt: int = 0,
    repair_budget: int = 0,
    retryable: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    validate_failure_action_pair(failure_class, action)
    requirement_ids, hard_requirement_ids = _requirement_scope(state, assessment)
    normalized_subtype = subtype or _assessment_subtype(assessment, "UNKNOWN")
    return {
        "failure_class": failure_class.value,
        "subtype": normalized_subtype,
        "reason": normalized_subtype,
        "task_id": _current_task_id(dict(state)),
        "action": action.value,
        "retryable": retryable,
        "repair_attempt": repair_attempt,
        "repair_budget": repair_budget,
        "user_blocker": failure_class is FailureClass.USER_DECISION_REQUIRED,
        "requirement_ids": requirement_ids,
        "hard_requirement_ids": hard_requirement_ids,
        "metadata": dict(metadata or {}),
    }


def _set_decision(
    update: Dict[str, Any],
    state: Mapping[str, Any],
    assessment: Mapping[str, Any],
    **decision: Any,
) -> Dict[str, Any]:
    canonical = _failure_decision(
        state,
        assessment,
        **decision,
    )
    update["failure_decision"] = canonical
    if canonical["failure_class"] == FailureClass.USER_DECISION_REQUIRED.value:
        _register_user_blocker(update, state, assessment, canonical)
    degraded_issue_id = next(
        (
            str(issue.get("issue_id") or "")
            for issue in reversed(update.get("degraded_issue_registry") or [])
            if isinstance(issue, Mapping) and issue.get("status") == "active"
        ),
        "-",
    )
    blocker_id = next(
        (
            str(blocker.get("blocker_id") or "")
            for blocker in reversed(update.get("pending_user_blockers") or [])
            if isinstance(blocker, Mapping)
            and str(blocker.get("status") or "") in {"pending", "retry_pending"}
        ),
        "-",
    )
    logger.info(
        "FAILURE_POLICY_DECISION task_id=%s failure_class=%s subtype=%s "
        "policy_action=%s repair_attempt=%s repair_budget=%s "
        "degraded_issue_id=%s blocker_id=%s requirement_id=%s hard_or_soft=%s",
        canonical.get("task_id") or "-",
        canonical.get("failure_class") or "-",
        canonical.get("subtype") or "-",
        canonical.get("action") or "-",
        canonical.get("repair_attempt", 0),
        canonical.get("repair_budget", 0),
        degraded_issue_id,
        blocker_id,
        ",".join(canonical.get("requirement_ids") or []) or "-",
        "hard" if canonical.get("hard_requirement_ids") else "soft",
    )
    return update


def _tasks_with_legacy_dependencies(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use the conservative historical serial contract for old checkpoints."""

    normalized: list[dict[str, Any]] = []
    prior_ids: list[str] = []
    for raw in state.get("tasks") or []:
        task = dict(raw)
        task_id = str(task.get("task_id") or "")
        if not isinstance(task.get("depends_on_task_ids"), list):
            task["depends_on_task_ids"] = (
                list(prior_ids)
                if str(task.get("task_type") or "") == "synthesis"
                else ([prior_ids[-1]] if prior_ids else [])
            )
        normalized.append(task)
        if task_id:
            prior_ids.append(task_id)
    return normalized


def _canonical_blocker_projection(
    blockers: Iterable[Mapping[str, Any]], *, task_id: str = ""
) -> Dict[str, Any]:
    """Project active canonical blockers into the legacy UI state field."""

    active = [
        deepcopy(dict(blocker))
        for blocker in blockers
        if str(blocker.get("status") or "") in {"pending", "retry_pending"}
    ]
    if not active:
        return {}
    choices = (
        list(active[0].get("available_options") or [])
        if len(active) == 1
        else []
    )
    return {
        "category": "CONSOLIDATED_BLOCKERS",
        "task_id": task_id or str(active[0].get("task_id") or ""),
        "blocker_id": str(active[-1].get("blocker_id") or ""),
        "blockers": active,
        "guidance": "请逐项处理以下未解决的硬性要求。",
        "accepted_choices": choices,
    }


def _register_user_blocker(
    update: Dict[str, Any],
    state: Mapping[str, Any],
    assessment: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    task_id = str(decision.get("task_id") or _current_task_id(dict(state)))
    subtype = str(decision.get("subtype") or "HARD_CONTRACT_BLOCKED")
    tasks = _tasks_with_legacy_dependencies(state)
    affected = affected_task_closure(tasks, task_id)
    missing_resource_id = next(
        (
            str(
                issue.get("resource_id")
                or issue.get("resource_name")
                or issue.get("missing_resource")
                or ""
            ).strip()
            for issue in assessment.get("issues") or []
            if isinstance(issue, Mapping)
            and str(
                issue.get("resource_id")
                or issue.get("resource_name")
                or issue.get("missing_resource")
                or ""
            ).strip()
        ),
        None,
    )
    options = ["MODIFY_REQUIREMENT", "CANCEL_JOB"]
    if subtype in _EVIDENCE_CODES or subtype == "MISSING_RESOURCE":
        options.insert(0, "UPLOAD_RESOURCES")
    requirement_ids = list(decision.get("hard_requirement_ids") or [])
    requirements = {
        str(requirement.get("requirement_id") or ""): requirement
        for requirement in state.get("requirement_registry") or []
        if isinstance(requirement, Mapping)
    }
    if any(
        str(requirements.get(requirement_id, {}).get("kind") or "")
        in {"human_approval", "waiver"}
        for requirement_id in requirement_ids
    ):
        options.insert(-1, "APPROVE_EXCEPTION")
    task_revision = _normalise_counter(state.get("task_revisions"), dict(state)).get(
        task_id, 1
    )
    record = build_user_blocker(
        job_scope=str(
            state.get("_job_id")
            or state.get("job_id")
            or state.get("thread_id")
            or "legacy-job"
        ),
        task_id=task_id,
        subtype=subtype,
        requirement_ids=requirement_ids,
        affected_task_ids=affected,
        missing_resource_id=missing_resource_id,
        reason=subtype,
        attempted_repairs=[
            {
                "repair_type": str(decision.get("action") or ""),
                "attempt": int(decision.get("repair_attempt", 0) or 0),
                "budget": int(decision.get("repair_budget", 0) or 0),
                "outcome": "exhausted",
                "diagnostic_code": subtype,
            }
        ],
        available_options=options,
        metadata={
            "task_revision": task_revision,
            "plan_revision": int(state.get("plan_revision", 1) or 1),
        },
    )
    blockers = upsert_user_blocker(
        state.get("pending_user_blockers") or [], record
    )
    outcomes = deepcopy(dict(state.get("task_outcome_registry") or {}))
    result_ids = {
        str(result.get("task_id") or "")
        for result in state.get("results") or []
        if isinstance(result, Mapping)
    }
    degraded_task_ids = {
        str(issue.get("task_id") or "")
        for issue in state.get("degraded_issue_registry") or []
        if isinstance(issue, Mapping) and issue.get("status") == "active"
    }
    revisions = _normalise_counter(state.get("task_revisions"), dict(state))
    for task in tasks:
        candidate_id = str(task.get("task_id") or "")
        existing = dict(outcomes.get(candidate_id) or {})
        status = str(existing.get("status") or "pending")
        if candidate_id in result_ids:
            status = "degraded" if candidate_id in degraded_task_ids else "committed"
        if candidate_id == task_id:
            status = "blocked_user"
        elif candidate_id in affected:
            status = "blocked_dependency"
        blocker_ids = list(existing.get("blocker_ids") or [])
        if candidate_id in affected and record["blocker_id"] not in blocker_ids:
            blocker_ids.append(record["blocker_id"])
        outcomes[candidate_id] = {
            "task_id": candidate_id,
            "status": status,
            "dependency_ids": list(task.get("depends_on_task_ids") or []),
            "blocker_ids": blocker_ids,
            "task_revision": revisions.get(candidate_id, 1),
        }
    update["pending_user_blockers"] = blockers
    update["task_outcome_registry"] = outcomes
    if isinstance(update.get("pending_user_action"), dict):
        update["pending_user_action"]["blocker_id"] = record["blocker_id"]
    runnable = runnable_task_ids(tasks, outcomes)
    if runnable:
        next_task_id = runnable[0]
        next_index = next(
            index
            for index, task in enumerate(tasks)
            if str(task.get("task_id") or "") == next_task_id
        )
        update["cursor"] = max(0, next_index - 1)
        update["workflow_action"] = WorkflowAction.NEXT.value
        update["pending_user_action"] = {}
    else:
        update["pending_user_action"] = _canonical_blocker_projection(
            blockers, task_id=task_id
        )


def _terminal_scheduler_update(
    state: Mapping[str, Any], *, terminal_status: str
) -> Dict[str, Any]:
    """Route after a terminal task using only explicit dependency outcomes."""

    tasks = _tasks_with_legacy_dependencies(state)
    task_id = _current_task_id(dict(state))
    revisions = _normalise_counter(state.get("task_revisions"), dict(state))
    outcomes = deepcopy(dict(state.get("task_outcome_registry") or {}))
    result_ids = {
        str(result.get("task_id") or "")
        for result in state.get("results") or []
        if isinstance(result, Mapping)
    }
    for task in tasks:
        candidate_id = str(task.get("task_id") or "")
        existing = dict(outcomes.get(candidate_id) or {})
        status = str(existing.get("status") or "pending")
        if candidate_id in result_ids and status == "pending":
            status = "committed"
        outcomes[candidate_id] = {
            "task_id": candidate_id,
            "status": status,
            "dependency_ids": list(task.get("depends_on_task_ids") or []),
            "blocker_ids": list(existing.get("blocker_ids") or []),
            "task_revision": revisions.get(candidate_id, 1),
        }
    if task_id in outcomes:
        outcomes[task_id]["status"] = terminal_status

    blockers = deepcopy(list(state.get("pending_user_blockers") or []))
    resolved_ids = list(dict.fromkeys(state.get("resolved_user_blocker_ids") or []))
    newly_resolved: set[str] = set()
    for blocker in blockers:
        if (
            str(blocker.get("task_id") or "") == task_id
            and str(blocker.get("status") or "") == "retry_pending"
        ):
            blocker["status"] = "resolved"
            blocker_id = str(blocker.get("blocker_id") or "")
            newly_resolved.add(blocker_id)
            if blocker_id and blocker_id not in resolved_ids:
                resolved_ids.append(blocker_id)
    if newly_resolved:
        for outcome in outcomes.values():
            outcome["blocker_ids"] = [
                blocker_id
                for blocker_id in outcome.get("blocker_ids") or []
                if blocker_id not in newly_resolved
            ]
            if (
                outcome.get("status") == "blocked_dependency"
                and not outcome["blocker_ids"]
            ):
                outcome["status"] = "pending"

    transition: Dict[str, Any] = {
        "task_outcome_registry": outcomes,
        "pending_user_blockers": blockers,
        "resolved_user_blocker_ids": resolved_ids,
    }
    runnable = runnable_task_ids(tasks, outcomes)
    if runnable:
        next_task_id = runnable[0]
        next_index = next(
            index
            for index, task in enumerate(tasks)
            if str(task.get("task_id") or "") == next_task_id
        )
        transition.update(
            {
                "workflow_action": WorkflowAction.NEXT.value,
                "cursor": max(0, next_index - 1),
                "pending_user_action": {},
            }
        )
        return transition
    unresolved = [
        blocker
        for blocker in blockers
        if str(blocker.get("status") or "") == "pending"
    ]
    if unresolved:
        transition.update(
            {
                "workflow_action": WorkflowAction.NEEDS_USER_INPUT.value,
                "pending_user_action": _canonical_blocker_projection(unresolved),
            }
        )
        return transition
    transition.update(
        {
            "workflow_action": WorkflowAction.DONE.value,
            "pending_user_action": {},
        }
    )
    return transition


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


def _resource_identifiers(resource: Mapping[str, Any]) -> set[str]:
    return _as_strings(
        resource.get(key)
        for key in ("name", "path", "file_id", "resource_id")
    )


def _missing_resource_category(
    issue: Mapping[str, Any], state: Mapping[str, Any]
) -> IssueCategory:
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


def _classify_issue(
    issue: Mapping[str, Any], state: Mapping[str, Any]
) -> IssueCategory:
    code = str(issue.get("code") or "").strip().upper()
    if code == "MISSING_RESOURCE":
        return _missing_resource_category(issue, state)
    if code in _VERIFIER_CODES:
        return IssueCategory.VERIFIER_FAILURE
    if code in _EVIDENCE_CODES:
        return IssueCategory.EVIDENCE_GAP
    if code in _EXTERNAL_CODES:
        return IssueCategory.EXTERNAL_BLOCKER
    if code in _LOCAL_PLAN_CODES:
        return IssueCategory.LOCAL_PLAN_DEFECT
    if code in _CONTENT_CODES:
        return IssueCategory.CONTENT_DEFECT
    category = str(issue.get("category") or "").strip().upper()
    if category == IssueCategory.LOCAL_PLAN_DEFECT.value:
        return IssueCategory.EXTERNAL_BLOCKER
    try:
        return IssueCategory(category)
    except ValueError:
        return IssueCategory.CONTENT_DEFECT


def classify_issue(
    issue: Mapping[str, Any], state: Mapping[str, Any]
) -> IssueCategory:
    """Classify one normalized issue for execution-level recovery planning."""

    return _classify_issue(issue, state)


def _allows_accept_as_draft(
    state: Mapping[str, Any],
    issues: Iterable[Mapping[str, Any]],
) -> bool:
    """Allow draft acceptance only when every evidence issue is waivable."""

    issue_list = list(issues)
    if any(
        classify_issue(issue, state) is IssueCategory.VERIFIER_FAILURE
        for issue in issue_list
    ):
        return False
    evidence_issues = [
        issue
        for issue in issue_list
        if classify_issue(issue, state) is IssueCategory.EVIDENCE_GAP
    ]
    return not any(
        not is_waivable_evidence_gap(issue) for issue in evidence_issues
    )


def sanitize_blocker_choices(
    state: Mapping[str, Any],
    issues: Iterable[Mapping[str, Any]],
    choices: Iterable[Any] | None,
    category: str,
) -> list[str]:
    """Normalize current or persisted blocker choices through policy safety rules."""

    issue_list = list(issues)
    normalized = list(
        dict.fromkeys(
            str(choice or "").strip().upper()
            for choice in (choices or [])
            if str(choice or "").strip()
        )
    )
    if not normalized:
        normalized = (
            ["UPLOAD_RESOURCES", "AUTHORIZE_WEB", "ADJUST_REQUIREMENT", "DONE"]
            if str(category or "").upper() == IssueCategory.EVIDENCE_GAP.value
            else (
                ["RETRY_VERIFIER", "DONE"]
                if str(category or "").upper()
                == IssueCategory.VERIFIER_FAILURE.value
                else ["REWORK", "DONE"]
            )
        )
    if str(category or "").upper() == IssueCategory.VERIFIER_FAILURE.value:
        normalized = ["RETRY_VERIFIER", "DONE"]

    evidence_issues = [
        issue
        for issue in issue_list
        if classify_issue(issue, state) is IssueCategory.EVIDENCE_GAP
    ]
    has_waivable_gap = any(
        is_waivable_evidence_gap(issue) for issue in evidence_issues
    )
    allow_draft = _allows_accept_as_draft(state, issue_list)
    disallowed: set[str] = set()
    current_task_id = str(_current_task(state).get("task_id") or "").strip()
    current_result = state.get("current_result") or {}
    result_task_id = (
        str(current_result.get("task_id") or "").strip()
        if isinstance(current_result, Mapping)
        else ""
    )
    allow_asset_retry = bool(
        str(category or "").upper() == IssueCategory.CONTENT_DEFECT.value
        and _asset_only_issues({"issues": issue_list})
        and current_task_id
        and result_task_id == current_task_id
    )
    if not has_waivable_gap:
        disallowed.add("ACCEPT_EVIDENCE_GAP")
    if not allow_draft:
        disallowed.update({"ACCEPT_AS_DRAFT", *NEXT_RESUME_ALIASES})
    if not allow_asset_retry:
        disallowed.add("RETRY_ASSET")
    if not public_web_runtime_available():
        disallowed.add("AUTHORIZE_WEB")
    return [choice for choice in normalized if choice not in disallowed]


def classify_assessment(assessment: Dict[str, Any], state: Dict[str, Any]) -> IssueCategory:
    """Return the highest-priority actionable category in an assessment.

    Any Verifier failure invalidates semantic issues that coexist in a legacy
    assessment because the candidate result was never assessed reliably.
    """
    classified = [
        _classify_issue(issue, state)
        for issue in assessment.get("issues") or []
        if isinstance(issue, dict)
    ]
    if IssueCategory.VERIFIER_FAILURE in classified:
        return IssueCategory.VERIFIER_FAILURE
    return max(
        classified or [IssueCategory.CONTENT_DEFECT],
        key=_CATEGORY_PRIORITY.get,
    )


def _asset_only_issues(assessment: Mapping[str, Any]) -> bool:
    codes = {
        str(issue.get("code") or "").strip().upper()
        for issue in assessment.get("issues") or []
        if isinstance(issue, Mapping)
        and str(issue.get("code") or "").strip().upper() not in _VERIFIER_CODES
    }
    return bool(codes) and codes <= _ASSET_ISSUE_CODES


def _continuation_action(state: Dict[str, Any]) -> WorkflowAction:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    return WorkflowAction.DONE if cursor + 1 >= len(tasks) else WorkflowAction.NEXT


def commit_current_result(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Commit the current accepted revision once for its stable task ID."""
    results = list(state.get("results") or [])
    current_result = state.get("current_result") or {}
    if not current_result:
        return results

    task_id = _current_task_id(state)
    committed = dict(current_result)
    committed["task_id"] = task_id
    task_revisions = _normalise_counter(state.get("task_revisions"), state)
    committed["task_revision"] = task_revisions.get(task_id, 1)
    committed["plan_revision"] = int(state.get("plan_revision", 1) or 1)
    for index, result in enumerate(results):
        if isinstance(result, dict) and str(result.get("task_id")) == task_id:
            results[index] = committed
            return results
    results.append(committed)
    return results


def _pending_user_action(
    category: IssueCategory,
    state: Dict[str, Any],
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    task_id = _current_task_id(state)
    task_revision = int((state.get("task_revisions") or {}).get(task_id, 1) or 1)
    plan_revision = int(state.get("plan_revision", 1) or 1)
    blocker_sequence = max(
        int((state.get("task_retry_count") or {}).get(task_id, 0) or 0),
        int((state.get("asset_retry_count") or {}).get(task_id, 0) or 0),
        int((state.get("evidence_recovery_count") or {}).get(task_id, 0) or 0),
        int((state.get("verifier_retry_count") or {}).get(task_id, 0) or 0),
        1,
    )
    pending = {
        "category": category.value,
        "task_id": task_id,
        "issues": list(assessment.get("issues") or []),
        "blocker_id": (
            f"{task_id}:p{plan_revision}:t{task_revision}:"
            f"{category.value}:{blocker_sequence}"
        ),
        "blocker_status": "ACTIVE",
    }
    issues = [
        issue
        for issue in assessment.get("issues") or []
        if isinstance(issue, Mapping)
    ]
    allow_draft = _allows_accept_as_draft(state, issues)
    if category is IssueCategory.EVIDENCE_GAP:
        evidence_issues = [
            issue
            for issue in assessment.get("issues") or []
            if isinstance(issue, Mapping)
            and classify_issue(issue, state) is IssueCategory.EVIDENCE_GAP
        ]
        accepted_choices = list(_EVIDENCE_USER_CHOICES[:-1])
        web_available = public_web_runtime_available()
        if not web_available:
            accepted_choices = [
                choice for choice in accepted_choices if choice != "AUTHORIZE_WEB"
            ]
        if any(is_waivable_evidence_gap(issue) for issue in evidence_issues):
            accepted_choices.append("ACCEPT_EVIDENCE_GAP")
        if evidence_issues and allow_draft:
            accepted_choices.append("ACCEPT_AS_DRAFT")
        descriptions = [
            str(issue.get("description") or "").strip()
            for issue in evidence_issues
            if str(issue.get("description") or "").strip()
        ]
        gap_list = "\n".join(f"- {description}" for description in descriptions)
        choices_for_guidance = ["上传补充资料"]
        if "AUTHORIZE_WEB" in accepted_choices:
            choices_for_guidance.append("授权公开网络检索")
        choices_for_guidance.append("调整任务要求")
        if "ACCEPT_EVIDENCE_GAP" in accepted_choices:
            choices_for_guidance.append("接受证据缺口并继续修复其他问题")
        if "ACCEPT_AS_DRAFT" in accepted_choices:
            choices_for_guidance.append("接受为带风险草稿")
        choice_guidance = "、".join(choices_for_guidance) + "。"
        runtime_guidance = (
            "\n当前服务器未提供可用的公开网络检索工具；请先安装并初始化该能力，"
            "或选择上传资料/调整要求。"
            if not web_available
            else ""
        )
        can_retrieve = _has_authorized_evidence_retrieval(state)
        pending.update(
            {
                "accepted_choices": accepted_choices,
                "guidance": (
                    (
                        "当前任务仍存在证据缺口，但未授权可执行的证据检索能力：\n"
                        if not can_retrieve
                        else "当前任务的自动证据恢复已达上限，仍存在以下证据缺口：\n"
                    )
                    + f"{gap_list or '- 当前授权来源不足以满足硬性证据要求。'}\n"
                    + (
                        "“接受证据缺口”仅接受证据可得性缺口；篇幅、格式、内容及"
                        "引用完整性错误仍会继续处理。\n"
                        if "ACCEPT_EVIDENCE_GAP" in accepted_choices
                        else ""
                    )
                    + f"请在页面的阻塞处理区选择：{choice_guidance}"
                    + f"{runtime_guidance}"
                ),
            }
        )
    elif category is IssueCategory.CONTENT_DEFECT:
        asset_only = _asset_only_issues(assessment)
        accepted_choices = (
            ["RETRY_ASSET", "ADJUST_REQUIREMENT"]
            if asset_only
            else ["REWORK", "ADJUST_REQUIREMENT"]
        )
        if allow_draft:
            accepted_choices.append("ACCEPT_AS_DRAFT")
        accepted_choices.append("DONE")
        if asset_only:
            choice_guidance = (
                "重新生成资产、调整资产要求、明确接受为带风险草稿，或结束工作流。"
                if allow_draft
                else "重新生成资产、调整资产要求，或结束工作流。"
            )
            guidance_prefix = (
                "当前正文已保留，但所需图形或表格资产自动恢复失败；正文不会重新生成。"
            )
        else:
            choice_guidance = (
                "按反馈再次返工、调整任务要求、明确接受为带风险草稿，或结束工作流。"
                if allow_draft
                else "按反馈再次返工、调整任务要求，或结束工作流。"
            )
            guidance_prefix = "当前任务已达到自动返工上限，但仍未通过验收。"
        pending.update(
            {
                "accepted_choices": accepted_choices,
                "guidance": (
                    f"{guidance_prefix}请在页面的阻塞处理区选择："
                    f"{choice_guidance}"
                ),
            }
        )
    elif category is IssueCategory.VERIFIER_FAILURE:
        accepted_choices = ["RETRY_VERIFIER", "DONE"]
        descriptions = [
            str(issue.get("description") or "").strip()
            for issue in issues
            if str(issue.get("description") or "").strip()
        ]
        failure_detail = descriptions[0] if descriptions else "自动校验未能完成。"
        pending.update(
            {
                "accepted_choices": accepted_choices,
                "guidance": (
                    f"{failure_detail}请在页面的阻塞处理区选择："
                    "仅重试自动校验器，或结束工作流。"
                ),
            }
        )
    return pending


def _content_retry_warning(
    warnings: List[Dict[str, Any]],
    task_id: str,
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    for warning in warnings:
        if (
            warning.get("code") == "CONTENT_RETRY_LIMIT_REACHED"
            and str(warning.get("task_id")) == task_id
        ):
            return warning
    return {
        "code": "CONTENT_RETRY_LIMIT_REACHED",
        "category": IssueCategory.CONTENT_DEFECT.value,
        "task_id": task_id,
        "issues": list(assessment.get("issues") or []),
    }


def _commit_degraded_result(
    update: Dict[str, Any],
    state: Dict[str, Any],
    assessment: Dict[str, Any],
    *,
    warning_code: str,
) -> Dict[str, Any]:
    """Commit a supported partial result as a terminal non-Human outcome."""

    task_id = _current_task_id(state)
    warning = {
        "code": warning_code,
        "category": FailureClass.DEGRADABLE_QUALITY.value,
        "task_id": task_id,
        "issues": list(assessment.get("issues") or []),
    }
    warnings = [
        existing
        for existing in update.get("verification_warnings") or []
        if not (
            str(existing.get("code") or "") == warning_code
            and str(existing.get("task_id") or "") == task_id
        )
    ]
    warnings.append(warning)
    subtype = _assessment_subtype(assessment, warning_code)
    requirement_ids, _ = _requirement_scope(state, assessment)
    affected_claims = [
        claim
        for issue in assessment.get("issues") or []
        if isinstance(issue, Mapping)
        for claim in issue.get("affected_claims") or []
    ]
    if subtype in _EVIDENCE_CODES:
        repair_type = FailureAction.RECOVER_EVIDENCE.value
        attempt = _normalise_counter(state.get("evidence_recovery_count"), state).get(
            task_id, 0
        )
        budget = MAX_EVIDENCE_RECOVERIES
    elif subtype in _ASSET_ISSUE_CODES:
        repair_type = FailureAction.RECOVER_ASSET.value
        attempt = _normalise_counter(state.get("asset_retry_count"), state).get(
            task_id, 0
        )
        budget = MAX_ASSET_RETRIES
    elif subtype in {"TOO_SHORT", "TOO_LONG"}:
        repair_type = FailureAction.REPAIR_CONTRACT.value
        attempt = 1
        budget = 1
    elif subtype in PLAN_PATCH_SUBTYPES:
        repair_type = FailureAction.PATCH_PLAN.value
        attempt = _normalise_counter(state.get("task_patch_count"), state).get(
            task_id, 0
        )
        budget = MAX_TASK_PATCHES
    else:
        repair_type = FailureAction.RETRY_TASK.value
        attempt = _normalise_counter(state.get("task_retry_count"), state).get(
            task_id, 0
        )
        budget = MAX_CONTENT_RETRIES
    task_revision = _normalise_counter(state.get("task_revisions"), state).get(
        task_id, 1
    )
    degraded_record = build_degraded_issue(
        task_id=task_id,
        task_revision=task_revision,
        subtype=subtype,
        affected_claims=affected_claims,
        affected_requirement_ids=requirement_ids,
        attempted_repairs=[
            {
                "repair_type": repair_type,
                "attempt": attempt,
                "budget": budget,
                "outcome": "exhausted",
                "diagnostic_code": subtype,
            }
        ],
        final_fallback="commit_supported_content_with_warning",
    )
    statuses = record_section_status(
        state,
        ACCEPT_WITH_WARNING,
        accepted_by="system",
        issues=assessment.get("issues") or [],
    )
    update.update(
        {
            "results": commit_current_result(state),
            "verification_warning": warning,
            "verification_warnings": warnings,
            "pending_user_action": {},
            "degraded_issue_registry": upsert_degraded_issue(
                state.get("degraded_issue_registry") or [], degraded_record
            ),
            "section_status": statuses,
            "report_status": derive_report_status(state.get("tasks") or [], statuses),
        }
    )
    update.update(_terminal_scheduler_update(state, terminal_status="degraded"))
    return _set_decision(
        update,
        state,
        assessment,
        failure_class=FailureClass.DEGRADABLE_QUALITY,
        action=FailureAction.COMMIT_WITH_WARNING,
        retryable=False,
    )


def _fatal_system_update(
    update: Dict[str, Any],
    state: Dict[str, Any],
    assessment: Dict[str, Any],
    *,
    subtype: str,
    component: str,
    operation: str,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    task_id = _current_task_id(state)
    safe_metadata = {
        str(key): value
        for key, value in (metadata or {}).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    identity = "|".join(
        (
            str(state.get("_job_id") or ""),
            task_id,
            subtype,
            component,
            operation,
        )
    )
    update.update(
        {
            "workflow_action": WorkflowAction.FATAL_SYSTEM.value,
            "pending_user_action": {},
            "fatal_system_error": {
                "failure_id": "fatal-" + sha256(identity.encode("utf-8")).hexdigest()[:24],
                "failure_class": FailureClass.FATAL_SYSTEM.value,
                "subtype": subtype,
                "origin": "graph",
                "component": component,
                "operation": operation,
                "task_id": task_id,
                "diagnostic_code": subtype,
                "retryable": False,
                "metadata": safe_metadata,
            },
        }
    )
    return _set_decision(
        update,
        state,
        assessment,
        failure_class=FailureClass.FATAL_SYSTEM,
        action=FailureAction.FAIL_JOB,
        subtype=subtype,
        retryable=False,
        metadata=safe_metadata,
    )


def _fatal_verifier_update(
    update: Dict[str, Any],
    state: Dict[str, Any],
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    return _fatal_system_update(
        update,
        state,
        assessment,
        subtype="VERIFIER_UNAVAILABLE",
        component="Verifier",
        operation="assessment",
    )


def decide_recovery_action(state: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Choose a bounded recovery action without invoking models or graph nodes."""
    had_requirement_registry = isinstance(state.get("requirement_registry"), list)
    state = normalize_failure_state(state)
    if not had_requirement_registry:
        state["requirement_registry"] = reconstruct_legacy_requirements(state)
    task_id = _current_task_id(state)
    task_retry_count = _counter_update(state, "task_retry_count")
    asset_retry_count = _counter_update(state, "asset_retry_count")
    evidence_recovery_count = _counter_update(state, "evidence_recovery_count")
    task_patch_count = _counter_update(state, "task_patch_count")
    verifier_retry_count = _counter_update(state, "verifier_retry_count")
    job_patch_count = int(state.get("job_patch_count", 0) or 0)
    legacy_length_attempts = "length_rewrite_attempts" not in state
    length_rewrite_attempts = {
        str(key): int(value or 0)
        for key, value in (state.get("length_rewrite_attempts") or {}).items()
        if str(key).strip()
    }
    update: Dict[str, Any] = {
        "task_retry_count": task_retry_count,
        "asset_retry_count": asset_retry_count,
        "evidence_recovery_count": evidence_recovery_count,
        "task_patch_count": task_patch_count,
        "verifier_retry_count": verifier_retry_count,
        "job_patch_count": job_patch_count,
        "length_rewrite_attempts": length_rewrite_attempts,
        "pending_user_action": {},
        "failure_decision": {},
        "fatal_system_error": {},
        "requirement_registry": deepcopy(state.get("requirement_registry") or []),
        "degraded_issue_registry": deepcopy(state.get("degraded_issue_registry") or []),
        "pending_user_blockers": deepcopy(state.get("pending_user_blockers") or []),
        "blocker_resolution_registry": deepcopy(
            state.get("blocker_resolution_registry") or []
        ),
        "resolved_user_blocker_ids": list(
            state.get("resolved_user_blocker_ids") or []
        ),
        "task_outcome_registry": deepcopy(state.get("task_outcome_registry") or {}),
        "verification_warnings": list(state.get("verification_warnings") or []),
    }

    verifier_failure = state.get("verifier_failure") or {}
    if isinstance(verifier_failure, Mapping) and str(
        verifier_failure.get("code") or ""
    ).strip():
        failure_assessment = {
            "status": "FAILED",
            "issues": [
                {
                    "code": str(verifier_failure.get("code")),
                    "category": IssueCategory.VERIFIER_FAILURE.value,
                    "description": str(
                        verifier_failure.get("message")
                        or "自动校验器本身未能产生合法校验结果。"
                    ),
                    "suggestion": "仅重试自动校验器，或检查模型服务配置。",
                    "severity": "error",
                }
            ],
        }
        retries = verifier_retry_count.get(task_id, 0)
        if bool(verifier_failure.get("retryable")) and retries < MAX_VERIFIER_RETRIES:
            verifier_retry_count[task_id] = retries + 1
            update["workflow_action"] = WorkflowAction.RETRY_VERIFIER.value
            return _set_decision(
                update,
                state,
                failure_assessment,
                failure_class=FailureClass.RETRYABLE_EXECUTION,
                action=FailureAction.RETRY_VERIFIER,
                subtype="VERIFIER_UNAVAILABLE",
                repair_attempt=retries + 1,
                repair_budget=MAX_VERIFIER_RETRIES,
                retryable=True,
            )
        return _fatal_verifier_update(
            update,
            state,
            failure_assessment,
        )

    if str(assessment.get("status") or "").upper() == "PASS":
        accepted_gap = matching_evidence_gap_acceptance(state)
        status = USER_ACCEPTED_GAP if accepted_gap is not None else VERIFIED_PASS
        statuses = record_section_status(
            state,
            status,
            accepted_by="user" if accepted_gap is not None else "verifier",
            issues=(accepted_gap or {}).get("issues") or [],
        )
        update["results"] = commit_current_result(state)
        update["section_status"] = statuses
        update["report_status"] = derive_report_status(
            state.get("tasks") or [], statuses
        )
        update.update(_terminal_scheduler_update(state, terminal_status="committed"))
        return update

    category = classify_assessment(assessment, state)
    task = _current_task(state)
    is_synthesis = str(task.get("task_type") or "") == "synthesis"
    issue_codes = {
        str(issue.get("code") or "").strip().upper()
        for issue in assessment.get("issues") or []
        if isinstance(issue, dict)
        and str(issue.get("code") or "").strip().upper() not in _VERIFIER_CODES
    }
    if "PLAN_PATCH_INTERNAL_ERROR" in issue_codes:
        error_type = next(
            (
                str(issue.get("error_type") or "PlanPatchError")
                for issue in assessment.get("issues") or []
                if isinstance(issue, Mapping)
                and str(issue.get("code") or "").strip().upper()
                == "PLAN_PATCH_INTERNAL_ERROR"
            ),
            "PlanPatchError",
        )
        return _fatal_system_update(
            update,
            state,
            assessment,
            subtype="PLAN_PATCH_INTERNAL_ERROR",
            component="PlanPatcher",
            operation="plan_patch",
            metadata={"error_type": error_type},
        )
    in_progress_statuses = record_section_status(
        state,
        BLOCKED,
        accepted_by="system",
        issues=assessment.get("issues") or [],
    )
    update["section_status"] = in_progress_statuses
    update["report_status"] = derive_report_status(
        state.get("tasks") or [], in_progress_statuses
    )
    if category is IssueCategory.VERIFIER_FAILURE:
        retries = verifier_retry_count.get(task_id, 0)
        if retries < MAX_VERIFIER_RETRIES:
            verifier_retry_count[task_id] = retries + 1
            update["workflow_action"] = WorkflowAction.RETRY_VERIFIER.value
            return _set_decision(
                update,
                state,
                assessment,
                failure_class=FailureClass.RETRYABLE_EXECUTION,
                action=FailureAction.RETRY_VERIFIER,
                repair_attempt=retries + 1,
                repair_budget=MAX_VERIFIER_RETRIES,
                retryable=True,
            )
        return _fatal_verifier_update(update, state, assessment)

    if is_synthesis and category in {
        IssueCategory.CONTENT_DEFECT,
        IssueCategory.EVIDENCE_GAP,
    }:
        retries = task_retry_count.get(task_id, 0)
        if retries < MAX_CONTENT_RETRIES:
            task_retry_count[task_id] = retries + 1
            update["workflow_action"] = WorkflowAction.SYNTHESIS_REWRITE.value
            return _set_decision(
                update,
                state,
                assessment,
                failure_class=FailureClass.RETRYABLE_EXECUTION,
                action=FailureAction.RETRY_TASK,
                repair_attempt=retries + 1,
                repair_budget=MAX_CONTENT_RETRIES,
                retryable=True,
            )
        # Synthesis never falls through into ordinary evidence recovery.
        category = IssueCategory.CONTENT_DEFECT

    if category is IssueCategory.CONTENT_DEFECT:
        if issue_codes == {"REPORT_SOURCE_INCONSISTENT"}:
            source_issue = next(
                issue
                for issue in assessment.get("issues") or []
                if isinstance(issue, Mapping)
                and str(issue.get("code") or "").strip().upper()
                == "REPORT_SOURCE_INCONSISTENT"
            )
            current_result = dict(state.get("current_result") or {})
            current_result["report_sources"] = [
                str(source).strip()
                for source in source_issue.get("expected_sources") or []
                if str(source).strip()
            ]
            update["current_result"] = current_result
            update["workflow_action"] = WorkflowAction.RETRY_VERIFIER.value
            return _set_decision(
                update,
                state,
                assessment,
                failure_class=FailureClass.REPAIRABLE_CONTRACT,
                action=FailureAction.REPAIR_CONTRACT,
                repair_attempt=1,
                repair_budget=1,
            )
        if _asset_only_issues(assessment) and state.get("current_result"):
            retries = asset_retry_count.get(task_id, 0)
            if retries < MAX_ASSET_RETRIES:
                asset_retry_count[task_id] = retries + 1
                update["workflow_action"] = WorkflowAction.ASSET_RECOVERY.value
                return _set_decision(
                    update,
                    state,
                    assessment,
                    failure_class=FailureClass.REPAIRABLE_CONTRACT,
                    action=FailureAction.RECOVER_ASSET,
                    repair_attempt=retries + 1,
                    repair_budget=MAX_ASSET_RETRIES,
                )

            _, hard_requirement_ids = _requirement_scope(state, assessment)
            if not hard_requirement_ids:
                return _commit_degraded_result(
                    update,
                    state,
                    assessment,
                    warning_code="ASSET_RETRY_LIMIT_REACHED",
                )

            statuses = record_section_status(
                state, BLOCKED, accepted_by="system", issues=assessment.get("issues") or []
            )
            update.update(
                {
                    "workflow_action": WorkflowAction.NEEDS_USER_INPUT.value,
                    "pending_user_action": _pending_user_action(
                        category, state, assessment
                    ),
                    "section_status": statuses,
                    "report_status": derive_report_status(
                        state.get("tasks") or [], statuses
                    ),
                }
            )
            return _set_decision(
                update,
                state,
                assessment,
                failure_class=FailureClass.USER_DECISION_REQUIRED,
                action=FailureAction.REGISTER_BLOCKER,
            )

        length_only = bool(issue_codes) and issue_codes <= {"TOO_SHORT", "TOO_LONG"}
        if length_only:
            task_revision = int(
                (state.get("task_revisions") or {}).get(task_id, 1) or 1
            )
            attempt_key = f"{task_id}:length_rewrite:t{task_revision}"
            if (
                legacy_length_attempts
                and attempt_key not in length_rewrite_attempts
                and task_retry_count.get(task_id, 0) > 0
            ):
                # Legacy checkpoints only persisted the shared content counter.
                # Treat an existing retry as already consuming the semantic rewrite.
                length_rewrite_attempts[attempt_key] = 1
            if int(length_rewrite_attempts.get(attempt_key, 0) or 0) < 1:
                length_rewrite_attempts[attempt_key] = 1
                task_retry_count[task_id] = task_retry_count.get(task_id, 0) + 1
                update["workflow_action"] = WorkflowAction.LENGTH_REWRITE.value
                return _set_decision(
                    update,
                    state,
                    assessment,
                    failure_class=FailureClass.REPAIRABLE_CONTRACT,
                    action=FailureAction.REPAIR_CONTRACT,
                    repair_attempt=1,
                    repair_budget=1,
                )
            if issue_codes == {"TOO_LONG"}:
                target = parse_length_target(str(task.get("task_description") or ""))
                if target and target.get("max") is not None:
                    current_result = dict(state.get("current_result") or {})
                    content_field = (
                        "content" if current_result.get("content") is not None else "text_output"
                    )
                    trimmed = safe_deterministic_trim(
                        str(current_result.get(content_field) or ""),
                        maximum=int(target["max"]),
                        minimum=target.get("min"),
                    )
                    if trimmed is not None:
                        current_result[content_field] = trimmed
                        update["current_result"] = current_result
                        update["workflow_action"] = WorkflowAction.RETRY_VERIFIER.value
                        return _set_decision(
                            update,
                            state,
                            assessment,
                            failure_class=FailureClass.REPAIRABLE_CONTRACT,
                            action=FailureAction.REPAIR_CONTRACT,
                            repair_attempt=1,
                            repair_budget=1,
                        )
            retries = MAX_CONTENT_RETRIES
        else:
            retries = task_retry_count.get(task_id, 0)
        if retries < MAX_CONTENT_RETRIES:
            task_retry_count[task_id] = retries + 1
            update["workflow_action"] = WorkflowAction.REWORK.value
            return _set_decision(
                update,
                state,
                assessment,
                failure_class=FailureClass.RETRYABLE_EXECUTION,
                action=FailureAction.RETRY_TASK,
                repair_attempt=retries + 1,
                repair_budget=MAX_CONTENT_RETRIES,
                retryable=True,
            )

        _, hard_requirement_ids = _requirement_scope(state, assessment)
        if not hard_requirement_ids:
            return _commit_degraded_result(
                update,
                state,
                assessment,
                warning_code="CONTENT_RETRY_LIMIT_REACHED",
            )

        warning = _content_retry_warning(
            update["verification_warnings"], task_id, assessment
        )
        warnings = update["verification_warnings"]
        if warning not in warnings:
            warnings = [*warnings, warning]
        statuses = record_section_status(
            state,
            BLOCKED,
            accepted_by="system",
            issues=assessment.get("issues") or [],
        )
        update.update(
            {
                "workflow_action": WorkflowAction.NEEDS_USER_INPUT.value,
                "verification_warning": warning,
                "verification_warnings": warnings,
                "pending_user_action": _pending_user_action(
                    category, state, assessment
                ),
                "section_status": statuses,
                "report_status": derive_report_status(
                    state.get("tasks") or [], statuses
                ),
            }
        )
        return _set_decision(
            update,
            state,
            assessment,
            failure_class=FailureClass.USER_DECISION_REQUIRED,
            action=FailureAction.REGISTER_BLOCKER,
        )

    if category is IssueCategory.EVIDENCE_GAP:
        recoveries = evidence_recovery_count.get(task_id, 0)
        if (
            _has_authorized_evidence_retrieval(state)
            and recoveries < MAX_EVIDENCE_RECOVERIES
        ):
            evidence_recovery_count[task_id] = recoveries + 1
            update["workflow_action"] = WorkflowAction.EVIDENCE_RECOVERY.value
            return _set_decision(
                update,
                state,
                assessment,
                failure_class=FailureClass.RETRYABLE_EXECUTION,
                action=FailureAction.RECOVER_EVIDENCE,
                repair_attempt=recoveries + 1,
                repair_budget=MAX_EVIDENCE_RECOVERIES,
                retryable=True,
            )
        _, hard_requirement_ids = _requirement_scope(state, assessment)
        if not hard_requirement_ids:
            return _commit_degraded_result(
                update,
                state,
                assessment,
                warning_code="EVIDENCE_RECOVERY_EXHAUSTED",
            )

    if category is IssueCategory.LOCAL_PLAN_DEFECT:
        patches = task_patch_count.get(task_id, 0)
        if patches < MAX_TASK_PATCHES and job_patch_count < MAX_JOB_PATCHES:
            update["workflow_action"] = WorkflowAction.PLAN_PATCH.value
            return _set_decision(
                update,
                state,
                assessment,
                failure_class=FailureClass.REPAIRABLE_CONTRACT,
                action=FailureAction.PATCH_PLAN,
                repair_attempt=patches + 1,
                repair_budget=MAX_TASK_PATCHES,
            )
        _, hard_requirement_ids = _requirement_scope(state, assessment)
        if not hard_requirement_ids:
            if state.get("current_result"):
                return _commit_degraded_result(
                    update,
                    state,
                    assessment,
                    warning_code="PLAN_PATCH_BUDGET_EXHAUSTED",
                )
            return _fatal_system_update(
                update,
                state,
                assessment,
                subtype="PLAN_PATCH_BUDGET_EXHAUSTED",
                component="DecisionPolicy",
                operation="plan_patch_budget",
            )

    blocking_status = (
        EXTERNAL_BLOCKER
        if category is IssueCategory.EXTERNAL_BLOCKER
        else BLOCKED
    )
    statuses = record_section_status(
        state,
        blocking_status,
        accepted_by="system",
        issues=assessment.get("issues") or [],
    )
    update.update(
        {
            "workflow_action": WorkflowAction.NEEDS_USER_INPUT.value,
            "pending_user_action": _pending_user_action(category, state, assessment),
            "section_status": statuses,
            "report_status": derive_report_status(
                state.get("tasks") or [], statuses
            ),
        }
    )
    return _set_decision(
        update,
        state,
        assessment,
        failure_class=FailureClass.USER_DECISION_REQUIRED,
        action=FailureAction.REGISTER_BLOCKER,
    )

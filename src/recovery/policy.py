"""Pure, bounded policy for recovery after a task assessment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from os.path import basename
from typing import Any, Dict, List

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
from src.tool_capabilities import public_web_runtime_available
from src.task_contract import effective_web_allowed


class WorkflowAction(str, Enum):
    PASS = "PASS"
    NEXT = "NEXT"
    DONE = "DONE"
    REWORK = "REWORK"
    EVIDENCE_RECOVERY = "EVIDENCE_RECOVERY"
    PLAN_PATCH = "PLAN_PATCH"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    ACCEPT_WITH_WARNING = "ACCEPT_WITH_WARNING"
    RETRY_VERIFIER = "RETRY_VERIFIER"
    LENGTH_REWRITE = "LENGTH_REWRITE"
    SYNTHESIS_REWRITE = "SYNTHESIS_REWRITE"


class IssueCategory(str, Enum):
    CONTENT_DEFECT = "CONTENT_DEFECT"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    LOCAL_PLAN_DEFECT = "LOCAL_PLAN_DEFECT"
    EXTERNAL_BLOCKER = "EXTERNAL_BLOCKER"
    VERIFIER_FAILURE = "VERIFIER_FAILURE"


MAX_CONTENT_RETRIES = 2
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
    "MISSING_TABLE",
    "TOO_SHORT",
    "TOO_LONG",
    "OUT_OF_SCOPE",
    "SCOPE_VIOLATION",
    "UNSUPPORTED_RECOMMENDATION",
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
    "INVALID_CITATION_ID",
    "MISSING_INLINE_CITATION",
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

    evidence_issues = [
        issue
        for issue in issues
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
            else ["REWORK", "DONE"]
        )

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
    if not has_waivable_gap:
        disallowed.add("ACCEPT_EVIDENCE_GAP")
    if not allow_draft:
        disallowed.update({"ACCEPT_AS_DRAFT", *NEXT_RESUME_ALIASES})
    if not public_web_runtime_available():
        disallowed.add("AUTHORIZE_WEB")
    return [choice for choice in normalized if choice not in disallowed]


def classify_assessment(assessment: Dict[str, Any], state: Dict[str, Any]) -> IssueCategory:
    """Return the highest-priority actionable category in an assessment.

    A contract error is a Verifier health signal, not a defect in the candidate
    result. It controls recovery only when no other issue survived assessment.
    Runtime/service failures retain their existing priority because semantic
    verification did not run successfully at all.
    """
    classified = [
        (
            str(issue.get("code") or "").strip().upper(),
            _classify_issue(issue, state),
        )
        for issue in assessment.get("issues") or []
        if isinstance(issue, dict)
    ]
    actionable = [
        category
        for code, category in classified
        if code != "ASSESSMENT_CONTRACT_ERROR"
    ]
    return max(
        actionable
        or [category for _, category in classified]
        or [IssueCategory.CONTENT_DEFECT],
        key=_CATEGORY_PRIORITY.get,
    )


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
        accepted_choices = ["REWORK", "ADJUST_REQUIREMENT"]
        if allow_draft:
            accepted_choices.append("ACCEPT_AS_DRAFT")
        accepted_choices.append("DONE")
        choice_guidance = (
            "按反馈再次返工、调整任务要求、明确接受为带风险草稿，或结束工作流。"
            if allow_draft
            else "按反馈再次返工、调整任务要求，或结束工作流。"
        )
        pending.update(
            {
                "accepted_choices": accepted_choices,
                "guidance": (
                    "当前任务已达到自动返工上限，但仍未通过验收。请在页面的阻塞处理区选择："
                    f"{choice_guidance}"
                ),
            }
        )
    elif category is IssueCategory.VERIFIER_FAILURE:
        accepted_choices = ["REWORK"]
        if allow_draft:
            accepted_choices.append("ACCEPT_AS_DRAFT")
        accepted_choices.append("DONE")
        choice_guidance = (
            "重新生成后再校验、明确接受当前未验证内容为带风险草稿，或结束工作流。"
            if allow_draft
            else "重新生成后再校验，或结束工作流。"
        )
        pending.update(
            {
                "accepted_choices": accepted_choices,
                "guidance": (
                    "自动校验未能完成。请在页面的阻塞处理区选择："
                    f"{choice_guidance}"
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


def decide_recovery_action(state: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Choose a bounded recovery action without invoking models or graph nodes."""
    task_id = _current_task_id(state)
    task_retry_count = _counter_update(state, "task_retry_count")
    evidence_recovery_count = _counter_update(state, "evidence_recovery_count")
    task_patch_count = _counter_update(state, "task_patch_count")
    verifier_retry_count = _counter_update(state, "verifier_retry_count")
    job_patch_count = int(state.get("job_patch_count", 0) or 0)
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
        accepted_gap = matching_evidence_gap_acceptance(state)
        status = USER_ACCEPTED_GAP if accepted_gap is not None else VERIFIED_PASS
        statuses = record_section_status(
            state,
            status,
            accepted_by="user" if accepted_gap is not None else "verifier",
            issues=(accepted_gap or {}).get("issues") or [],
        )
        update["workflow_action"] = _continuation_action(state).value
        update["results"] = commit_current_result(state)
        update["section_status"] = statuses
        update["report_status"] = derive_report_status(
            state.get("tasks") or [], statuses
        )
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
            return update

    if is_synthesis and category in {
        IssueCategory.CONTENT_DEFECT,
        IssueCategory.EVIDENCE_GAP,
    }:
        retries = task_retry_count.get(task_id, 0)
        if retries < MAX_CONTENT_RETRIES:
            task_retry_count[task_id] = retries + 1
            update["workflow_action"] = WorkflowAction.SYNTHESIS_REWRITE.value
            return update
        # Synthesis never falls through into ordinary evidence recovery.
        category = IssueCategory.CONTENT_DEFECT

    if category is IssueCategory.CONTENT_DEFECT:
        retries = task_retry_count.get(task_id, 0)
        if retries < MAX_CONTENT_RETRIES:
            task_retry_count[task_id] = retries + 1
            update["workflow_action"] = (
                WorkflowAction.LENGTH_REWRITE.value
                if issue_codes and issue_codes <= {"TOO_SHORT", "TOO_LONG"}
                else WorkflowAction.REWORK.value
            )
            return update

        warning = _content_retry_warning(
            update["verification_warnings"], task_id, assessment
        )
        warnings = update["verification_warnings"]
        if warning not in warnings:
            warnings = [*warnings, warning]
        statuses = record_section_status(
            state,
            ACCEPT_WITH_WARNING,
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
        return update

    if category is IssueCategory.EVIDENCE_GAP:
        recoveries = evidence_recovery_count.get(task_id, 0)
        if (
            _has_authorized_evidence_retrieval(state)
            and recoveries < MAX_EVIDENCE_RECOVERIES
        ):
            evidence_recovery_count[task_id] = recoveries + 1
            update["workflow_action"] = WorkflowAction.EVIDENCE_RECOVERY.value
            return update

    if category is IssueCategory.LOCAL_PLAN_DEFECT:
        patches = task_patch_count.get(task_id, 0)
        if patches < MAX_TASK_PATCHES and job_patch_count < MAX_JOB_PATCHES:
            update["workflow_action"] = WorkflowAction.PLAN_PATCH.value
            return update

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
    return update

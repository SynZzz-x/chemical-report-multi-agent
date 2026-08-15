"""Bounded recovery nodes for the automatic verification workflow."""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.evidence_waivers import (
    record_evidence_gap_acceptance,
    split_waivable_evidence_gaps,
)
from src.llm import get_llm
from src.nodes.planner import planner as planner_node
from src.recovery.plan_patch import apply_plan_patch, validate_plan_patch
from src.recovery.policy import (
    IssueCategory,
    WorkflowAction,
    _continuation_action,
    classify_issue,
    commit_current_result,
    decide_recovery_action,
)
from src.report_validation import parse_length_target
from src.report_acceptance import (
    USER_ACCEPTED_GAP,
    USER_ACCEPTED_WARNING,
    derive_report_status,
    record_section_status,
)
from src.state import State
from src.task_contract import effective_web_allowed
from src.tool_capabilities import public_web_runtime_available


logger = logging.getLogger(__name__)


_RESUME_ACTION_ALIASES = {
    "NEXT": WorkflowAction.NEXT.value,
    "CONTINUE": WorkflowAction.NEXT.value,
    "继续": WorkflowAction.NEXT.value,
    "带限制继续": WorkflowAction.NEXT.value,
    "接受当前结果": WorkflowAction.NEXT.value,
    "跳过": WorkflowAction.NEXT.value,
    "DONE": WorkflowAction.DONE.value,
    "完成": WorkflowAction.DONE.value,
    "结束": WorkflowAction.DONE.value,
    "REWORK": WorkflowAction.REWORK.value,
    "RETRY": WorkflowAction.REWORK.value,
    "返工": WorkflowAction.REWORK.value,
    "重试": WorkflowAction.REWORK.value,
    "EVIDENCE_RECOVERY": WorkflowAction.EVIDENCE_RECOVERY.value,
    "证据恢复": WorkflowAction.EVIDENCE_RECOVERY.value,
    "扩大检索": WorkflowAction.EVIDENCE_RECOVERY.value,
}

_FULL_REPLAN_ACTIONS = {
    "FULL_REPLAN",
    "FULL_REPLAN_RETRY",
    "FULL_REPLAN_REFINED",
    "FULL_REPLAN_ERROR",
}
_FULL_REPLAN_DECISIONS = {"REPLAN", "FULL_REPLAN"}
_RETRIEVAL_EVIDENCE_CODES = {
    "EVIDENCE_GAP",
    "INSUFFICIENT_EVIDENCE",
    "MISSING_CITATION",
    "MISSING_EVIDENCE",
    "RAG_COVERAGE_GAP",
    "RAG_INSUFFICIENT",
    "SOURCE_UNSUPPORTED",
}
_RETRIEVAL_QUERY_MAX_CHARS = 200
_EVIDENCE_USER_CHOICES = [
    "UPLOAD_RESOURCES",
    "AUTHORIZE_WEB",
    "ADJUST_REQUIREMENT",
    "ACCEPT_EVIDENCE_GAP",
]
_SPECIAL_RESUME_CHOICES = set(_EVIDENCE_USER_CHOICES)
_SPECIAL_RESUME_CHOICES.add("ACCEPT_AS_DRAFT")
_SPECIAL_CHOICE_ALIASES = {
    "上传补充资料": "UPLOAD_RESOURCES",
    "上传资料": "UPLOAD_RESOURCES",
    "授权公开网络检索": "AUTHORIZE_WEB",
    "授权WEB检索": "AUTHORIZE_WEB",
    "授权网络检索": "AUTHORIZE_WEB",
    "调整任务要求": "ADJUST_REQUIREMENT",
    "缩小任务要求": "ADJUST_REQUIREMENT",
    "接受仅报告现有证据及缺口": "ACCEPT_EVIDENCE_GAP",
    "接受证据缺口": "ACCEPT_EVIDENCE_GAP",
    "接受为带风险草稿": "ACCEPT_AS_DRAFT",
    "接受当前缺陷并作为带风险草稿继续": "ACCEPT_AS_DRAFT",
}


def _canonical_resume_action(value: Any) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    return _RESUME_ACTION_ALIASES.get(normalized.upper())


def _resume_action(resumed: Any) -> str | None:
    if isinstance(resumed, dict):
        candidates = (resumed.get("action"), resumed.get("text"))
    else:
        candidates = (resumed,)
    return next(
        (action for value in candidates if (action := _canonical_resume_action(value))),
        None,
    )


def _resume_choice(resumed: Any) -> str:
    if isinstance(resumed, dict):
        value = resumed.get("action") or resumed.get("text")
    else:
        value = resumed
    normalized = str(value or "").strip()
    upper = normalized.upper()
    if upper in _SPECIAL_RESUME_CHOICES:
        return upper
    if normalized in _SPECIAL_CHOICE_ALIASES:
        return _SPECIAL_CHOICE_ALIASES[normalized]
    for prefix in ("调整任务要求：", "调整任务要求:", "缩小任务要求：", "缩小任务要求:"):
        if normalized.startswith(prefix):
            return "ADJUST_REQUIREMENT"
    return upper


def _adjusted_requirement_text(value: str) -> str:
    text = value.strip()
    if text in {"调整任务要求", "缩小任务要求"}:
        return ""
    for prefix in ("调整任务要求：", "调整任务要求:", "缩小任务要求：", "缩小任务要求:"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _is_replan_control_message(message: Any) -> bool:
    try:
        payload = json.loads(str(getattr(message, "content", "") or "").strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("to") == "Planner"
        and str(payload.get("type") or "").strip().upper()
        in {"REPLAN", "FULL_REPLAN"}
    )


def automatic_planner(
    state: State, config: Optional[RunnableConfig] = None, **kwargs
) -> dict[str, Any]:
    """Run Planner on an auto-safe copy that cannot invoke full replanning."""
    sanitized_state = dict(state)
    planner_action = str(state.get("planner_action") or "")
    decision = str(state.get("decision") or "")
    has_full_replan_staging = bool(
        state.get("full_replan_previous_task_ids")
        or state.get("full_replan_reason")
        or state.get("full_replan_candidate_tasks")
        or planner_action in _FULL_REPLAN_ACTIONS
        or decision in _FULL_REPLAN_DECISIONS
    )
    if has_full_replan_staging:
        # A restored manual checkpoint may still be waiting to retry or
        # confirm a replacement plan.  Automatic recovery discards only that
        # staging state; normal Planner actions (notably INTAKE_SUMMARY) and
        # their confirmation guidance remain Planner's responsibility.
        sanitized_state.update(
            {
                "full_replan_previous_task_ids": [],
                "full_replan_reason": "",
                "full_replan_candidate_tasks": [],
                "guidance": {},
            }
        )
        if planner_action in _FULL_REPLAN_ACTIONS:
            sanitized_state["planner_action"] = ""
        if decision in _FULL_REPLAN_DECISIONS:
            sanitized_state["decision"] = WorkflowAction.NEXT.value
    sanitized_state["messages"] = [
        message
        for message in state.get("messages", []) or []
        if not _is_replan_control_message(message)
    ]
    update = planner_node(sanitized_state, config, **kwargs)
    if not has_full_replan_staging:
        return update

    # LangGraph state updates are partial, so return the clears persistently.
    # Do not overwrite Planner's normal action or guidance: an automatic
    # initial plan must still route through Planner_Confirm.
    update = {
        **update,
        "full_replan_previous_task_ids": [],
        "full_replan_reason": "",
        "full_replan_candidate_tasks": [],
    }
    update.setdefault("guidance", {})
    return update


def _current_task(state: State) -> dict[str, Any]:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], dict):
        return tasks[cursor]
    return state.get("current_task") or {}


def _current_task_id(state: State) -> str:
    task = _current_task(state)
    return str(
        task.get("task_id")
        or (state.get("current_result") or {}).get("task_id")
        or state.get("cursor", 0)
    )


def _worker_state_with_feedback(
    state: State, feedback: dict[str, Any]
) -> dict[str, Any]:
    worker_state = deepcopy(state.get("worker_state") or {})
    worker_state["execution_feedback"] = feedback
    return worker_state


def _issue_instructions(issues: list[dict[str, Any]]) -> str:
    instructions = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        suggestion = str(issue.get("suggestion") or "").strip()
        description = str(issue.get("description") or "").strip()
        if suggestion:
            instructions.append(suggestion)
        elif description:
            instructions.append(description)
    return "\n".join(f"- {instruction}" for instruction in instructions)


def _recovery_sequence(state: State, mode: str, task_id: str) -> int:
    field = (
        "evidence_recovery_count"
        if mode == "evidence_recovery"
        else "task_retry_count"
    )
    counter = state.get(field) or {}
    try:
        return max(1, int(counter.get(task_id, 0) or 0))
    except (AttributeError, TypeError, ValueError):
        return 1


def _normalise_retrieval_query(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    query = re.sub(r"\s+", " ", value).strip(" \t\r\n\"'`，。；;")
    if not query:
        return ""
    return query[:_RETRIEVAL_QUERY_MAX_CHARS].rstrip()


def _retrieval_query_tokens(query: str) -> list[str]:
    return [
        token.casefold()
        for token in re.split(r"[\s,，、;；:：/|]+", query)
        if token.strip()
    ]


def _merge_retrieval_queries(values: list[Any]) -> list[str]:
    """Deterministically normalize and collapse exact/contained evidence queries."""

    merged: list[str] = []
    for value in values:
        candidate = _normalise_retrieval_query(value)
        if not candidate:
            continue
        candidate_tokens = _retrieval_query_tokens(candidate)
        candidate_set = set(candidate_tokens)
        handled = False
        for index, existing in enumerate(merged):
            existing_tokens = _retrieval_query_tokens(existing)
            existing_set = set(existing_tokens)
            if candidate.casefold() == existing.casefold() or (
                candidate_set and candidate_set <= existing_set
            ):
                handled = True
                break
            if existing_set and existing_set < candidate_set:
                merged[index] = candidate
                handled = True
                break
            union = candidate_set | existing_set
            overlap = len(candidate_set & existing_set) / len(union) if union else 0.0
            if overlap >= 0.75:
                combined_tokens = list(existing_tokens)
                combined_tokens.extend(
                    token for token in candidate_tokens if token not in existing_set
                )
                merged[index] = _normalise_retrieval_query(" ".join(combined_tokens))
                handled = True
                break
        if not handled:
            merged.append(candidate)

    changed = True
    while changed:
        changed = False
        for left_index in range(len(merged)):
            left_tokens = _retrieval_query_tokens(merged[left_index])
            left_set = set(left_tokens)
            for right_index in range(left_index + 1, len(merged)):
                right_tokens = _retrieval_query_tokens(merged[right_index])
                right_set = set(right_tokens)
                if left_set == right_set or (right_set and right_set <= left_set):
                    merged.pop(right_index)
                    changed = True
                    break
                if left_set and left_set < right_set:
                    merged.pop(left_index)
                    changed = True
                    break
                union = left_set | right_set
                overlap = len(left_set & right_set) / len(union) if union else 0.0
                if overlap >= 0.75:
                    combined = list(left_tokens)
                    combined.extend(token for token in right_tokens if token not in left_set)
                    merged[left_index] = _normalise_retrieval_query(" ".join(combined))
                    merged.pop(right_index)
                    changed = True
                    break
            if changed:
                break
    return merged


def _build_recovery_plan(
    state: State,
    issues: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    """Build an execution-only, version-bound plan from all verifier issues."""

    task = _current_task(state)
    task_id = _current_task_id(state)
    plan_revision = int(state.get("plan_revision", 1) or 1)
    task_revision = int((state.get("task_revisions") or {}).get(task_id, 1) or 1)
    sequence = _recovery_sequence(state, mode, task_id)
    evidence_issues = [
        issue
        for issue in issues
        if isinstance(issue, dict)
        and classify_issue(issue, state) is IssueCategory.EVIDENCE_GAP
    ]

    raw_evidence_queries: list[Any] = []
    for issue in evidence_issues:
        code = str(issue.get("code") or "").strip().upper()
        if code not in _RETRIEVAL_EVIDENCE_CODES:
            continue
        raw_evidence_queries.append(issue.get("retrieval_query"))

    evidence_queries = _merge_retrieval_queries(raw_evidence_queries)

    has_evidence_root = bool(evidence_queries)
    asset_actions: list[dict[str, Any]] = []
    scope_constraints: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code") or "").strip().upper()
        if code in {"MISSING_FIGURE", "MISSING_IMAGE"}:
            visualization = task.get("visualization")
            causal = (
                isinstance(visualization, dict)
                and str(visualization.get("kind") or "").strip().lower() == "causal"
            )
            action: dict[str, Any] = {
                "asset": "causal_figure" if causal else "figure",
                "action": "regenerate",
            }
            if causal and has_evidence_root:
                action["after"] = "evidence_recovery"
            if action not in asset_actions:
                asset_actions.append(action)
        elif code == "MISSING_TABLE":
            action = {"asset": "table", "action": "materialize"}
            if action not in asset_actions:
                asset_actions.append(action)
        elif code in {
            "OUT_OF_SCOPE",
            "SCOPE_VIOLATION",
            "UNSUPPORTED_RECOMMENDATION",
        }:
            constraint = str(
                issue.get("suggestion") or issue.get("description") or ""
            ).strip()
            if constraint and constraint not in scope_constraints:
                scope_constraints.append(constraint)

    length_target = (
        parse_length_target(str(task.get("task_description") or ""))
        if any(
            str(issue.get("code") or "").strip().upper()
            in {"TOO_SHORT", "TOO_LONG"}
            for issue in issues
            if isinstance(issue, dict)
        )
        else None
    )
    return {
        "recovery_id": (
            f"{task_id}:p{plan_revision}:t{task_revision}:{mode}:{sequence}"
        ),
        "task_id": task_id,
        "plan_revision": plan_revision,
        "task_revision": task_revision,
        "recovery_sequence": sequence,
        "evidence_queries": evidence_queries,
        "asset_actions": asset_actions,
        "length_target": length_target,
        "scope_constraints": scope_constraints,
    }


def _continuation_message(action: str, state: State) -> AIMessage | None:
    if action == WorkflowAction.NEXT.value:
        return AIMessage(
            content=json.dumps(
                {
                    "from": "DecisionPolicy",
                    "to": "Planner",
                    "type": "PROCEED",
                    "current_section": _current_task(state).get("task_name"),
                },
                ensure_ascii=False,
            )
        )
    if action == WorkflowAction.DONE.value:
        return AIMessage(
            content=json.dumps(
                {"from": "DecisionPolicy", "to": "Summarizer", "type": "SUMMARIZE"},
                ensure_ascii=False,
            )
        )
    return None


def decision_policy(
    state: State, config: Optional[RunnableConfig] = None, **kwargs
) -> dict[str, Any]:
    """Apply the pure policy and attach execution-only feedback for rework."""
    assessment = state.get("assessment") or {}
    update = decide_recovery_action(state, assessment)
    action = str(update.get("workflow_action") or "")
    if action == WorkflowAction.LENGTH_REWRITE.value:
        issues = list(assessment.get("issues") or [])
        target = parse_length_target(
            str(_current_task(state).get("task_description") or "")
        )
        limits = (
            f"目标字数范围：{target.get('min') or 0}～{target.get('max') or '不限'}字。"
            if target is not None
            else "严格修复确定性字数缺陷。"
        )
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "length_rewrite",
                "issues": issues,
                "instructions": f"{limits}\n{_issue_instructions(issues)}".strip(),
                "source_result": deepcopy(state.get("current_result") or {}),
            },
        )
    elif action == WorkflowAction.REWORK.value:
        issues = list(assessment.get("issues") or [])
        recovery_state = {**state, **update}
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "rework",
                "issues": issues,
                "instructions": _issue_instructions(issues),
                "recovery_plan": _build_recovery_plan(
                    recovery_state, issues, mode="rework"
                ),
            },
        )

    continuation = (
        str(update.get("continuation_action") or "")
        if action == WorkflowAction.ACCEPT_WITH_WARNING.value
        else action
    )
    message = _continuation_message(continuation, state)
    if message is not None:
        update["messages"] = [message]
        update["decision"] = continuation
    logger.info(
        "Workflow policy decision: source=system task=%s assessment_status=%s action=%s",
        _current_task_id(state),
        str(assessment.get("status") or "UNKNOWN").upper(),
        action or "UNSPECIFIED",
    )
    return update


def route_policy(state: State, **kwargs) -> str:
    action = str(state.get("workflow_action") or WorkflowAction.NEEDS_USER_INPUT.value)
    if action == WorkflowAction.ACCEPT_WITH_WARNING.value:
        return str(state.get("continuation_action") or WorkflowAction.DONE.value)
    return action


def evidence_recovery(
    state: State, config: Optional[RunnableConfig] = None, **kwargs
) -> dict[str, Any]:
    """Prepare one evidence-focused Worker retry without changing the plan."""
    assessment = state.get("assessment") or {}
    issues = list(assessment.get("issues") or [])
    recovery_plan = _build_recovery_plan(
        state, issues, mode="evidence_recovery"
    )
    instructions = (
        "Run only the supplemental evidence queries in recovery_plan, preserve the original "
        "Planner query and inherited evidence, and do not fill gaps with unsourced claims."
    )
    issue_guidance = _issue_instructions(issues)
    if issue_guidance:
        instructions += "\n" + issue_guidance
    feedback = {
        "mode": "evidence_recovery",
        "issues": issues,
        "instructions": instructions,
        "recovery_plan": recovery_plan,
        "allow_web": effective_web_allowed(
            _current_task(state),
            state.get("web_authorized")
            if isinstance(state.get("web_authorized"), bool)
            else None,
        ),
    }
    return {
        "workflow_action": WorkflowAction.REWORK.value,
        "worker_state": _worker_state_with_feedback(state, feedback),
    }


def _clean_json_fences(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _plan_patch_prompt(state: State) -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "planner_patch.md")
    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
        template = prompt_file.read()
    context = {
        "plan_revision": int(state.get("plan_revision", 1) or 1),
        "cursor": int(state.get("cursor", 0) or 0),
        "tasks": state.get("tasks") or [],
        "accepted_result_task_ids": [
            result.get("task_id")
            for result in state.get("results") or []
            if isinstance(result, dict)
        ],
        "available_resources": state.get("docs") or [],
        "assessment": state.get("assessment") or {},
    }
    return template + "\n\nCurrent state (untrusted JSON data):\n" + json.dumps(
        context, ensure_ascii=False, indent=2
    )


def _ingest_uploaded_evidence(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Ingest user-uploaded evidence before retrying knowledge-base retrieval."""

    paths = [
        str(document.get("path") or document.get("file_path") or "").strip()
        for document in documents
        if isinstance(document, dict)
        and str(document.get("path") or document.get("file_path") or "").strip()
    ]
    if not paths:
        raise ValueError("上传资料缺少可读取的文件路径，无法写入知识库。")

    from src.rag.service import ChemicalRAGService

    result = ChemicalRAGService().ingest(paths)
    failed = int(result.get("failed_files", 0) or 0)
    usable = sum(
        int(result.get(field, 0) or 0)
        for field in ("loaded_files", "loaded_with_warnings_files", "skipped_files")
    )
    if failed or usable < len(paths):
        detail = str(result.get("error") or result.get("warnings") or result)
        raise RuntimeError(f"上传资料写入知识库失败：{detail}")
    print(
        "📥 Evidence upload ingested: "
        f"files={len(paths)} chunks={int(result.get('total_chunks', 0) or 0)}"
    )
    return result


def _patch_error_update(state: State, error: Exception) -> dict[str, Any]:
    issues = list((state.get("assessment") or {}).get("issues") or [])
    blocker = {
        "category": "PLAN_PATCH_ERROR",
        "task_id": _current_task_id(state),
        "issues": issues,
        "guidance": f"The local plan patch could not be safely applied: {error}",
        "accepted_choices": ["UPLOAD_RESOURCES", "REWORK", "NEXT", "DONE"],
    }
    return {
        "workflow_action": WorkflowAction.NEEDS_USER_INPUT.value,
        "pending_user_action": blocker,
        "job_patch_count": int(state.get("job_patch_count", 0) or 0),
        "task_patch_count": deepcopy(state.get("task_patch_count") or {}),
    }


def plan_patcher(
    state: State, config: Optional[RunnableConfig] = None, **kwargs
) -> dict[str, Any]:
    """Generate, validate, and atomically apply one local plan patch."""
    try:
        model = get_llm(config, json_mode=True)
        response = model.invoke([HumanMessage(content=_plan_patch_prompt(state))])
        patch = json.loads(_clean_json_fences(str(response.content)))
        validate_plan_patch(state, patch)
        update = apply_plan_patch(state, patch)
    except Exception as exc:
        return _patch_error_update(state, exc)

    update["workflow_action"] = WorkflowAction.REWORK.value
    update["worker_state"] = _worker_state_with_feedback(
        state,
        {
            "mode": "plan_patch",
            "issues": list((state.get("assessment") or {}).get("issues") or []),
            "instructions": "Execute the locally patched task at the validated resume point.",
        },
    )
    return update


def needs_user_input(
    state: State, config: Optional[RunnableConfig] = None, **kwargs
) -> dict[str, Any]:
    """Interrupt with a concrete blocker and route the incremental resume safely."""
    pending = deepcopy(state.get("pending_user_action") or {})
    category = str(pending.get("category") or "EXTERNAL_BLOCKER")
    task_id = str(pending.get("task_id") or _current_task_id(state))
    issues = list(pending.get("issues") or (state.get("assessment") or {}).get("issues") or [])
    accepted_choices = list(
        pending.get("accepted_choices")
        or (
            _EVIDENCE_USER_CHOICES
            if category == "EVIDENCE_GAP"
            else ["UPLOAD_RESOURCES", "REWORK", "EVIDENCE_RECOVERY", "NEXT", "DONE"]
        )
    )
    if category == "EVIDENCE_GAP" and not public_web_runtime_available():
        accepted_choices = [
            choice for choice in accepted_choices if choice != "AUTHORIZE_WEB"
        ]
    evidence_descriptions = [
        str(issue.get("description") or "").strip()
        for issue in issues
        if isinstance(issue, dict)
        and str(issue.get("description") or "").strip()
    ]
    default_evidence_guidance = (
        "当前任务的自动证据恢复已达上限，仍存在以下证据缺口：\n"
        + "\n".join(f"- {description}" for description in evidence_descriptions)
        + "\n请在页面的阻塞处理区选择：上传补充资料、授权公开网络检索、"
        "调整任务要求，或接受仅报告现有证据及缺口。"
    )
    guidance = str(
        pending.get("guidance")
        or (
            default_evidence_guidance
            if category == "EVIDENCE_GAP"
            else "Provide the missing resource or concrete guidance for the affected task."
        )
    )
    blocker_id = str(
        pending.get("blocker_id")
        or f"{task_id}:p{int(state.get('plan_revision', 1) or 1)}:legacy"
    )
    payload = {
        "type": "needs_user_input",
        "guidance_text": guidance,
        "guidance": guidance,
        "category": category,
        "affected_task": task_id,
        "issues": issues,
        "accepted_choices": accepted_choices,
        "blocker_id": blocker_id,
        "blocker_status": "ACTIVE",
    }
    resumed = interrupt(payload)
    if isinstance(resumed, dict):
        text = str(resumed.get("text") or "").strip()
        resumed_docs = list(resumed.get("docs") or [])
    else:
        text = str(resumed or "").strip()
        resumed_docs = []
    requested_choice = _resume_choice(resumed)
    requested_action = _resume_action(resumed)
    accepted_choice_labels = {
        str(choice or "").strip().upper() for choice in accepted_choices
    }
    special_choice_accepted = (
        requested_choice in _SPECIAL_RESUME_CHOICES
        and requested_choice in accepted_choice_labels
    )
    if special_choice_accepted:
        if requested_choice == "UPLOAD_RESOURCES":
            requested_action = (
                WorkflowAction.EVIDENCE_RECOVERY.value
                if category == "EVIDENCE_GAP"
                else WorkflowAction.REWORK.value
            )
        elif requested_choice in {"AUTHORIZE_WEB", "ADJUST_REQUIREMENT"}:
            requested_action = WorkflowAction.REWORK.value
        elif requested_choice == "ACCEPT_EVIDENCE_GAP":
            requested_action = _continuation_action(state).value
        elif requested_choice == "ACCEPT_AS_DRAFT":
            requested_action = _continuation_action(state).value
    elif requested_choice in _SPECIAL_RESUME_CHOICES:
        requested_action = None

    if (
        special_choice_accepted
        and requested_choice == "ADJUST_REQUIREMENT"
        and not _adjusted_requirement_text(text)
    ):
        raise ValueError("请按“调整任务要求：<新要求>”提供具体调整内容。")

    should_ingest_upload = bool(resumed_docs) and category == "EVIDENCE_GAP"
    if should_ingest_upload:
        _ingest_uploaded_evidence(resumed_docs)

    routable_choices = {
        WorkflowAction.REWORK.value,
        WorkflowAction.EVIDENCE_RECOVERY.value,
        WorkflowAction.NEXT.value,
        WorkflowAction.DONE.value,
    }
    accepted_action_names = {
        action
        for choice in accepted_choices
        if (action := _canonical_resume_action(choice)) is not None
    }
    if category == "EVIDENCE_GAP":
        # Preserve safe resume commands accepted by older checkpoints/UI clients
        # without adding them to the new user-facing resolution choices.
        accepted_action_names.update(
            {
                WorkflowAction.REWORK.value,
                WorkflowAction.EVIDENCE_RECOVERY.value,
            }
        )
    if special_choice_accepted and requested_action in routable_choices:
        action = requested_action
    elif requested_action in routable_choices and requested_action in accepted_action_names:
        action = requested_action
    elif category == "EVIDENCE_GAP":
        action = WorkflowAction.EVIDENCE_RECOVERY.value
    else:
        action = WorkflowAction.REWORK.value

    policy_update: dict[str, Any] = {}
    accepted_gap_issues: list[dict[str, Any]] = []
    accepted_evidence_gaps = deepcopy(state.get("accepted_evidence_gaps") or {})
    if special_choice_accepted and requested_choice == "ACCEPT_EVIDENCE_GAP":
        accepted_gap_issues, remaining_issues = split_waivable_evidence_gaps(issues)
        accepted_evidence_gaps = record_evidence_gap_acceptance(
            state, accepted_gap_issues
        )
        issues = remaining_issues
        if remaining_issues:
            residual_assessment = {
                **(state.get("assessment") or {}),
                "status": "FAILED",
                "issues": remaining_issues,
            }
            policy_state = {
                **state,
                "accepted_evidence_gaps": accepted_evidence_gaps,
            }
            policy_update = decide_recovery_action(
                policy_state, residual_assessment
            )
            action = str(
                policy_update.get("workflow_action")
                or WorkflowAction.NEEDS_USER_INPUT.value
            )

    update: dict[str, Any] = {
        **policy_update,
        "workflow_action": action,
        "pending_user_action": policy_update.get("pending_user_action", {}),
        "docs": resumed_docs,
        "blocker_resolution": {
            "blocker_id": blocker_id,
            "blocker_status": "RESOLVED",
            "action": action,
            "choice": requested_choice or "UNSPECIFIED",
            "task_id": task_id,
        },
    }
    if special_choice_accepted and requested_choice == "ACCEPT_EVIDENCE_GAP":
        update["accepted_evidence_gaps"] = accepted_evidence_gaps
    uploaded_paths = [
        str(doc.get("path") or doc.get("file_path") or "").strip()
        for doc in resumed_docs
        if isinstance(doc, dict)
        and str(doc.get("path") or doc.get("file_path") or "").strip()
    ]
    if uploaded_paths:
        tasks = deepcopy(state.get("tasks") or [])
        cursor = int(state.get("cursor", 0) or 0)
        if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], dict):
            task = dict(tasks[cursor])
            if special_choice_accepted and requested_choice == "UPLOAD_RESOURCES":
                task["use_rag"] = True
            task["use_resources"] = list(
                dict.fromkeys(list(task.get("use_resources") or []) + uploaded_paths)
            )
            tasks[cursor] = task
            update["tasks"] = tasks
    if special_choice_accepted and requested_choice == "UPLOAD_RESOURCES" and resumed_docs:
        existing_docs = list(state.get("docs") or [])
        known = {
            str(doc.get("file_id") or doc.get("path") or doc.get("name") or "")
            for doc in existing_docs
            if isinstance(doc, dict)
        }
        update["docs"] = existing_docs + [
            doc
            for doc in resumed_docs
            if not isinstance(doc, dict)
            or str(doc.get("file_id") or doc.get("path") or doc.get("name") or "")
            not in known
        ]
    if special_choice_accepted and requested_choice == "AUTHORIZE_WEB":
        action = WorkflowAction.REWORK.value
        update["workflow_action"] = action
        update["web_authorized"] = True
        tasks = deepcopy(update.get("tasks") or state.get("tasks") or [])
        cursor = int(state.get("cursor", 0) or 0)
        if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], dict):
            tasks[cursor] = {**tasks[cursor], "use_web": True}
            update["tasks"] = tasks
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "user_web_authorization",
                "issues": issues,
                "instructions": (
                    text
                    or "用户已明确授权本任务使用公开网络补充当前证据缺口。"
                ),
                "recovery_plan": _build_recovery_plan(
                    state, issues, mode="evidence_recovery"
                ),
                "allow_web": True,
            },
        )
    elif special_choice_accepted and requested_choice == "ADJUST_REQUIREMENT" and text:
        action = WorkflowAction.REWORK.value
        update["workflow_action"] = action
        adjusted_text = _adjusted_requirement_text(text)
        tasks = deepcopy(update.get("tasks") or state.get("tasks") or [])
        cursor = int(state.get("cursor", 0) or 0)
        if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], dict):
            task = dict(tasks[cursor])
            description = str(task.get("task_description") or "").rstrip()
            task["task_description"] = (
                f"{description}\n\n用户明确调整的要求：{adjusted_text}".strip()
            )
            tasks[cursor] = task
            update["tasks"] = tasks
            revisions = deepcopy(state.get("task_revisions") or {})
            revisions[task_id] = int(revisions.get(task_id, 1) or 1) + 1
            update["task_revisions"] = revisions
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "user_requirement_adjustment",
                "issues": issues,
                "instructions": adjusted_text,
            },
        )
    accepted_current_result = action == WorkflowAction.NEXT.value or (
        action == WorkflowAction.DONE.value
        and special_choice_accepted
        and requested_choice in {"ACCEPT_EVIDENCE_GAP", "ACCEPT_AS_DRAFT"}
    )
    if accepted_current_result:
        update["results"] = commit_current_result(state)
        acceptance_status = (
            USER_ACCEPTED_GAP
            if requested_choice == "ACCEPT_EVIDENCE_GAP"
            else USER_ACCEPTED_WARNING
        )
        statuses = record_section_status(
            state,
            acceptance_status,
            accepted_by="user",
            issues=(
                accepted_gap_issues
                if requested_choice == "ACCEPT_EVIDENCE_GAP"
                else issues
            ),
        )
        update["section_status"] = statuses
        update["report_status"] = derive_report_status(
            state.get("tasks") or [], statuses
        )
    if (
        action in {WorkflowAction.REWORK.value, WorkflowAction.LENGTH_REWRITE.value}
        and "worker_state" not in update
    ):
        feedback_mode = (
            "length_rewrite"
            if action == WorkflowAction.LENGTH_REWRITE.value
            else "user_resume"
        )
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": feedback_mode,
                "issues": issues,
                "instructions": text or _issue_instructions(issues) or guidance,
                **(
                    {"source_result": deepcopy(state.get("current_result") or {})}
                    if feedback_mode == "length_rewrite"
                    else {
                        "recovery_plan": _build_recovery_plan(
                            {**state, **update}, issues, mode="rework"
                        )
                    }
                ),
            },
        )
    message = _continuation_message(action, state)
    if message is not None:
        update["messages"] = [message]
        update["decision"] = action
    logger.info(
        "User blocker decision: blocker=%s task=%s category=%s choice=%s action=%s uploaded_files=%s",
        blocker_id,
        task_id,
        category,
        requested_choice or "UNSPECIFIED",
        action,
        str(bool(resumed_docs)).lower(),
    )
    return update


def route_after_blocker(state: State, **kwargs) -> str:
    return str(state.get("workflow_action") or WorkflowAction.NEEDS_USER_INPUT.value)

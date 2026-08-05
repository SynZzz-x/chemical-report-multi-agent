"""Bounded recovery nodes for the automatic verification workflow."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.llm import get_llm
from src.nodes.planner import planner as planner_node
from src.recovery.plan_patch import apply_plan_patch, validate_plan_patch
from src.recovery.policy import (
    WorkflowAction,
    commit_current_result,
    decide_recovery_action,
)
from src.state import State


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


def _is_legacy_replan_message(message: Any) -> bool:
    try:
        payload = json.loads(str(getattr(message, "content", "") or "").strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("to") == "Planner"
        and str(payload.get("type") or "").strip().upper() == "REPLAN"
    )


def automatic_planner(
    state: State, config: RunnableConfig, **kwargs
) -> dict[str, Any]:
    """Run Planner on an auto-safe copy of legacy checkpoint state."""
    sanitized_state = dict(state)
    sanitized_state["decision"] = WorkflowAction.NEXT.value
    sanitized_state["messages"] = [
        message
        for message in state.get("messages", []) or []
        if not _is_legacy_replan_message(message)
    ]
    return planner_node(sanitized_state, config, **kwargs)


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


def decision_policy(state: State, config: RunnableConfig, **kwargs) -> dict[str, Any]:
    """Apply the pure policy and attach execution-only feedback for rework."""
    assessment = state.get("assessment") or {}
    update = decide_recovery_action(state, assessment)
    action = str(update.get("workflow_action") or "")
    if action == WorkflowAction.REWORK.value:
        issues = list(assessment.get("issues") or [])
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "rework",
                "issues": issues,
                "instructions": _issue_instructions(issues),
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
    return update


def route_policy(state: State, config: RunnableConfig | None = None, **kwargs) -> str:
    action = str(state.get("workflow_action") or WorkflowAction.NEEDS_USER_INPUT.value)
    if action == WorkflowAction.ACCEPT_WITH_WARNING.value:
        return str(state.get("continuation_action") or WorkflowAction.DONE.value)
    return action


def _web_allowed(task: dict[str, Any]) -> bool:
    visualization = task.get("visualization") or {}
    return bool(
        task.get("use_web")
        or task.get("allow_web_fallback")
        or visualization.get("allow_web_fallback")
    )


def evidence_recovery(state: State, config: RunnableConfig, **kwargs) -> dict[str, Any]:
    """Prepare one evidence-focused Worker retry without changing the plan."""
    assessment = state.get("assessment") or {}
    issues = list(assessment.get("issues") or [])
    missing = [
        str(item).strip()
        for item in assessment.get("requirements_missing") or []
        if str(item).strip()
    ]
    if not missing:
        missing = [
            str(issue.get("description") or issue.get("code") or "").strip()
            for issue in issues
            if isinstance(issue, dict)
            and str(issue.get("description") or issue.get("code") or "").strip()
        ]
    recovery_query = " ".join(missing)
    instructions = (
        "Rewrite the retrieval query around the missing requirements, broaden RAG coverage, "
        "preserve source provenance and evidence coverage, and do not fill gaps with unsourced claims."
    )
    issue_guidance = _issue_instructions(issues)
    if issue_guidance:
        instructions += "\n" + issue_guidance
    feedback = {
        "mode": "evidence_recovery",
        "issues": issues,
        "instructions": instructions,
        "recovery_query": recovery_query,
        "allow_web": _web_allowed(_current_task(state)),
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


def plan_patcher(state: State, config: RunnableConfig, **kwargs) -> dict[str, Any]:
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


def needs_user_input(state: State, config: RunnableConfig, **kwargs) -> dict[str, Any]:
    """Interrupt with a concrete blocker and route the incremental resume safely."""
    pending = deepcopy(state.get("pending_user_action") or {})
    category = str(pending.get("category") or "EXTERNAL_BLOCKER")
    task_id = str(pending.get("task_id") or _current_task_id(state))
    issues = list(pending.get("issues") or (state.get("assessment") or {}).get("issues") or [])
    accepted_choices = list(
        pending.get("accepted_choices")
        or ["UPLOAD_RESOURCES", "REWORK", "EVIDENCE_RECOVERY", "NEXT", "DONE"]
    )
    guidance = str(
        pending.get("guidance")
        or "Provide the missing resource or concrete guidance for the affected task."
    )
    payload = {
        "type": "needs_user_input",
        "guidance_text": guidance,
        "guidance": guidance,
        "category": category,
        "affected_task": task_id,
        "issues": issues,
        "accepted_choices": accepted_choices,
    }
    resumed = interrupt(payload)
    if isinstance(resumed, dict):
        text = str(resumed.get("text") or "").strip()
        resumed_docs = list(resumed.get("docs") or [])
    else:
        text = str(resumed or "").strip()
        resumed_docs = []
    requested_action = _resume_action(resumed)

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
    if requested_action in routable_choices and requested_action in accepted_action_names:
        action = requested_action
    elif category == "EVIDENCE_GAP":
        action = WorkflowAction.EVIDENCE_RECOVERY.value
    else:
        action = WorkflowAction.REWORK.value

    update: dict[str, Any] = {
        "workflow_action": action,
        "pending_user_action": {},
        "docs": resumed_docs,
    }
    if action in {WorkflowAction.NEXT.value, WorkflowAction.DONE.value}:
        update["results"] = commit_current_result(state)
    if action == WorkflowAction.REWORK.value:
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "user_resume",
                "issues": issues,
                "instructions": text or guidance,
            },
        )
    message = _continuation_message(action, state)
    if message is not None:
        update["messages"] = [message]
        update["decision"] = action
    return update


def route_after_blocker(
    state: State, config: RunnableConfig | None = None, **kwargs
) -> str:
    return str(state.get("workflow_action") or WorkflowAction.NEEDS_USER_INPUT.value)

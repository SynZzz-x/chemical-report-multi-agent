"""Bounded recovery nodes for the automatic verification workflow."""

from __future__ import annotations

import json
import os
import re
import hashlib
from copy import deepcopy
from datetime import datetime
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
from src.task_contract import task_allows_web
from src.quality.models import QualityDimensions, ReviewRecord
from src.workflow_records import ensure_task_records, set_task_status
from src.workflow_store import WorkflowRecordStore


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
    state: State, config: RunnableConfig, **kwargs
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


def _human_override_review(
    state: State,
    task_id: str,
    action: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact_id = str(
        (state.get("active_artifact_ids") or {}).get(task_id)
        or (state.get("current_result") or {}).get("artifact_id")
        or f"legacy-{task_id}"
    )
    stable = f"{task_id}|{artifact_id}|human_override|{action}"
    review_id = "review_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
    record = ReviewRecord(
        review_id=review_id,
        task_id=task_id,
        artifact_id=artifact_id,
        reviewer="human_override",
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        status="PASS",
        issues=[],
        quality_dimensions=QualityDimensions(
            completeness=5,
            evidence=5,
            logic=5,
            actionability=5,
            safety=5,
        ),
    ).model_dump(mode="json")
    records = list(state.get("review_records") or [])
    prior = next(
        (
            review
            for review in records
            if isinstance(review, dict) and review.get("review_id") == review_id
        ),
        None,
    )
    if prior is not None:
        record = dict(prior)
    else:
        records.append(record)
    return record, records


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
                    "to": "TaskController",
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
        responsible_handlers = list(
            dict.fromkeys(
                str(issue.get("responsible_handler") or "worker_agent")
                for issue in issues
                if isinstance(issue, dict)
            )
        )
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "rework",
                "issues": issues,
                "instructions": _issue_instructions(issues),
                "responsible_handlers": responsible_handlers,
            },
        )

    message = _continuation_message(action, state)
    if message is not None:
        update["messages"] = [message]
        update["decision"] = action
    return update


def route_policy(state: State, config: RunnableConfig | None = None, **kwargs) -> str:
    action = str(state.get("workflow_action") or WorkflowAction.NEEDS_USER_INPUT.value)
    return action


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
        "allow_web": task_allows_web(
            _current_task(state),
            job_authorized=state.get("web_authorized") is True,
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


def needs_user_input(
    state: State,
    config: RunnableConfig,
    store: Any = None,
    **kwargs,
) -> dict[str, Any]:
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
    records = ensure_task_records(state)
    if action in {WorkflowAction.NEXT.value, WorkflowAction.DONE.value}:
        update["results"] = commit_current_result(state)
        review_record, review_records = _human_override_review(
            state, task_id, action
        )
        update["review_record"] = review_record
        update["review_records"] = review_records
        if store is not None:
            WorkflowRecordStore(
                store,
                str(state.get("user_id") or ""),
                str(state.get("job_id") or ""),
            ).put_review(review_record)
        update["task_records"] = set_task_status(
            records,
            task_id,
            "PASSED",
            active_artifact_id=(state.get("active_artifact_ids") or {}).get(task_id)
            or (state.get("current_result") or {}).get("artifact_id"),
        )
    if action == WorkflowAction.REWORK.value:
        update["task_records"] = set_task_status(
            records, task_id, "REVISE_REQUIRED"
        )
        update["worker_state"] = _worker_state_with_feedback(
            state,
            {
                "mode": "user_resume",
                "issues": issues,
                "instructions": text or guidance,
            },
        )
    if action == WorkflowAction.EVIDENCE_RECOVERY.value:
        update["task_records"] = set_task_status(
            records, task_id, "EVIDENCE_REQUIRED"
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

"""Pure helpers for hiding graph-internal messages in every UI surface."""

from __future__ import annotations

import json
from typing import Any


INTERNAL_CONTROL_TYPES = frozenset(
    {
        "INTAKE_SUMMARY",
        "PLAN_RESULT",
        "PROCEED",
        "REPLAN",
        "FULL_REPLAN",
        "PLAN_PATCH",
        "EVIDENCE_RECOVERY",
        "NEEDS_USER_INPUT",
        "needs_user_input",
        "REWORK",
        "SUMMARIZE",
    }
)


BLOCKER_ACTION_SPECS: dict[str, dict[str, Any]] = {
    "AUTHORIZE_WEB": {
        "label": "授权公开网络检索",
        "button_label": "授权并继续",
        "default_text": "已授权公开网络检索，请继续。",
        "requires_text": False,
        "requires_documents": False,
    },
    "ACCEPT_EVIDENCE_GAP": {
        "label": "接受现有证据及缺口报告",
        "button_label": "接受并继续",
        "default_text": "接受现有证据及缺口报告，请继续。",
        "requires_text": False,
        "requires_documents": False,
    },
    "ADJUST_REQUIREMENT": {
        "label": "调整任务要求",
        "button_label": "提交新要求",
        "default_text": "",
        "requires_text": True,
        "requires_documents": False,
    },
    "UPLOAD_RESOURCES": {
        "label": "上传补充资料",
        "button_label": "上传并继续",
        "default_text": "我已上传补充资料，请结合附件继续处理当前任务。",
        "requires_text": False,
        "requires_documents": True,
    },
    "RETRY_INITIAL_PLAN": {
        "label": "重新生成初始规划",
        "button_label": "重新生成",
        "default_text": "请重新生成初始规划。",
        "requires_text": False,
        "requires_documents": False,
    },
    "RETRY_FULL_REPLAN": {
        "label": "重新生成替换规划",
        "button_label": "重新生成",
        "default_text": "请重新生成替换规划。",
        "requires_text": False,
        "requires_documents": False,
    },
    "RESUME_OLD_PLAN": {
        "label": "恢复原规划",
        "button_label": "恢复并继续",
        "default_text": "恢复原规划并继续。",
        "requires_text": False,
        "requires_documents": False,
    },
    "CANCEL": {
        "label": "取消当前任务",
        "button_label": "确认取消",
        "default_text": "取消当前任务。",
        "requires_text": False,
        "requires_documents": False,
    },
    "REWORK": {
        "label": "按反馈返工",
        "button_label": "返工并继续",
        "default_text": "请根据当前反馈返工。",
        "requires_text": False,
        "requires_documents": False,
    },
    "EVIDENCE_RECOVERY": {
        "label": "继续证据恢复",
        "button_label": "恢复并继续",
        "default_text": "请继续执行证据恢复。",
        "requires_text": False,
        "requires_documents": False,
    },
    "NEXT": {
        "label": "接受当前结果并继续",
        "button_label": "接受并继续",
        "default_text": "接受当前结果并继续下一任务。",
        "requires_text": False,
        "requires_documents": False,
    },
    "DONE": {
        "label": "结束工作流",
        "button_label": "确认结束",
        "default_text": "结束当前工作流。",
        "requires_text": False,
        "requires_documents": False,
    },
}


def is_internal_control_message(content: str) -> bool:
    """Return whether serialized content is graph coordination, not chat text."""
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return str(payload.get("type") or "") in INTERNAL_CONTROL_TYPES


def is_displayable_assistant_content(content: str) -> bool:
    """Return whether assistant content is safe to render as chat text."""
    normalized = str(content or "").strip()
    return bool(normalized) and not is_internal_control_message(normalized)


def is_displayable_assistant_message(role: str, content: str) -> bool:
    """Apply the role and internal-control rules used by every chat projection."""
    return str(role or "").lower() in {"ai", "assistant"} and is_displayable_assistant_content(content)


def blocker_guidance(payload: Any) -> str | None:
    """Return display-safe guidance for a user-input blocker, if present."""
    if not isinstance(payload, dict) or payload.get("type") != "needs_user_input":
        return None
    return str(
        payload.get("guidance_text") or "需要你的输入后才能继续当前任务。"
    )


def blocker_choices(payload: Any) -> list[str]:
    """Return the exact actions exposed by a user-input blocker."""

    if not isinstance(payload, dict) or payload.get("type") != "needs_user_input":
        return []
    choices = payload.get("accepted_choices")
    if not isinstance(choices, list):
        return []
    return [
        str(choice).strip()
        for choice in choices
        if isinstance(choice, str) and str(choice).strip()
    ]


def blocker_action_spec(action: str) -> dict[str, Any]:
    """Return a copy of the UI contract for one blocker action."""

    normalized = str(action or "").strip()
    if not normalized:
        return {}
    configured = BLOCKER_ACTION_SPECS.get(normalized)
    if configured is not None:
        return dict(configured)
    return {
        "label": normalized,
        "button_label": "确认并继续",
        "default_text": f"已选择处理方式：{normalized}。",
        "requires_text": False,
        "requires_documents": False,
    }


def validate_blocker_submission(
    action: str,
    text: str,
    document_count: int,
) -> str | None:
    """Return a user-facing validation error for an incomplete action."""

    spec = blocker_action_spec(action)
    if not spec:
        return "请选择有效的处理方式。"
    if spec.get("requires_text") and not str(text or "").strip():
        return "请输入调整后的任务要求。"
    if spec.get("requires_documents") and document_count <= 0:
        return "请先上传补充资料。"
    return None


def build_resume_payload(
    *,
    text: str,
    docs: list[dict[str, Any]],
    message_id: str,
    action: str | None = None,
) -> dict[str, Any]:
    """Build the graph resume payload without losing a selected blocker action."""

    payload: dict[str, Any] = {
        "text": str(text),
        "message_id": str(message_id),
        "docs": list(docs),
    }
    selected = str(action or "").strip()
    if selected:
        payload["action"] = selected
    return payload


def build_blocker_resume_payload(
    *,
    action: str,
    text: str,
    docs: list[dict[str, Any]],
    message_id: str,
) -> dict[str, Any]:
    """Validate and build a resume payload for an explicit blocker action."""

    selected = str(action or "").strip()
    documents = list(docs)
    spec = blocker_action_spec(selected)
    resolved_text = str(text or "").strip() or str(spec.get("default_text") or "")
    error = validate_blocker_submission(selected, resolved_text, len(documents))
    if error:
        raise ValueError(error)
    return build_resume_payload(
        text=resolved_text,
        docs=documents,
        message_id=message_id,
        action=selected,
    )

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

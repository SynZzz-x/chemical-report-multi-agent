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


def blocker_guidance(payload: Any) -> str | None:
    """Return display-safe guidance for a user-input blocker, if present."""
    if not isinstance(payload, dict) or payload.get("type") != "needs_user_input":
        return None
    return str(
        payload.get("guidance_text") or "需要你的输入后才能继续当前任务。"
    )

"""Deterministic JobStore projection from authoritative workflow state."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from hashlib import sha256
from typing import Any

from .failure_semantics import FailureAction, FailureClass


class JobWorkflowMarker(str, Enum):
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    FATAL_SYSTEM = "FATAL_SYSTEM"


_SAFE_FATAL_FIELDS = (
    "failure_id",
    "failure_class",
    "subtype",
    "origin",
    "component",
    "operation",
    "task_id",
    "diagnostic_code",
    "retryable",
)


def _has_pending_blocker(state: Mapping[str, Any]) -> bool:
    return any(
        isinstance(blocker, Mapping)
        and str(blocker.get("status") or "") == "pending"
        for blocker in state.get("pending_user_blockers") or []
    )


def _is_fatal(state: Mapping[str, Any]) -> bool:
    decision = state.get("failure_decision") or {}
    if isinstance(decision, Mapping) and (
        str(decision.get("failure_class") or "") == FailureClass.FATAL_SYSTEM.value
        or str(decision.get("action") or "") == FailureAction.FAIL_JOB.value
    ):
        return True
    if str(state.get("workflow_action") or "") in {
        JobWorkflowMarker.FATAL_SYSTEM.value,
        FailureAction.FAIL_JOB.value,
    }:
        return True
    return bool(state.get("fatal_system_error"))


def derive_job_status(
    state: Mapping[str, Any] | None,
    pending_interrupt: Any | None,
    *,
    graph_incomplete: bool = False,
) -> str:
    """Return the external job status, with user waiting taking precedence."""

    values = state or {}
    if (
        pending_interrupt is not None
        or str(values.get("workflow_action") or "")
        == JobWorkflowMarker.NEEDS_USER_INPUT.value
        or _has_pending_blocker(values)
    ):
        return "waiting"
    if _is_fatal(values) or graph_incomplete:
        return "failed"
    return "completed"


def _safe_fatal_error(state: Mapping[str, Any]) -> dict[str, Any]:
    raw = state.get("fatal_system_error") or {}
    if isinstance(raw, Mapping):
        projected = {
            field: deepcopy(raw[field])
            for field in _SAFE_FATAL_FIELDS
            if field in raw
        }
        metadata = raw.get("metadata") or {}
        if projected and isinstance(metadata, Mapping):
            projected["metadata"] = {
                str(key): value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
        if projected:
            return projected

    decision = state.get("failure_decision") or {}
    if not isinstance(decision, Mapping):
        decision = {}
    subtype = str(
        decision.get("subtype")
        or state.get("workflow_action")
        or FailureClass.FATAL_SYSTEM.value
    )
    task_id = str(decision.get("task_id") or "") or None
    identity = f"{subtype}|{task_id or ''}|terminal_projection"
    return {
        "failure_id": "fatal-" + sha256(identity.encode("utf-8")).hexdigest()[:24],
        "failure_class": FailureClass.FATAL_SYSTEM.value,
        "subtype": subtype,
        "origin": "graph",
        "component": "Workflow",
        "operation": "terminal_projection",
        "task_id": task_id,
        "diagnostic_code": subtype,
        "retryable": False,
        "metadata": {},
    }


def project_job_outcome(
    state: Mapping[str, Any] | None,
    pending_interrupt: Any | None,
    *,
    graph_incomplete: bool = False,
) -> dict[str, Any]:
    """Build the bounded JobStore fields shared by Streamlit and CLI."""

    values = state or {}
    status = derive_job_status(
        values,
        pending_interrupt,
        graph_incomplete=graph_incomplete,
    )
    projection: dict[str, Any] = {
        "status": status,
        "pending_interrupt": pending_interrupt if status == "waiting" else None,
        "fatal_system_error": None,
    }
    if status == "failed":
        fatal = _safe_fatal_error(values)
        if fatal:
            projection["fatal_system_error"] = fatal
    return projection

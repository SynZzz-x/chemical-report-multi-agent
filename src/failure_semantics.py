"""Serializable failure semantics shared by policy and workflow projections."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Literal, TypedDict


JSONValue = Any


class FailureClass(str, Enum):
    RETRYABLE_EXECUTION = "RETRYABLE_EXECUTION"
    REPAIRABLE_CONTRACT = "REPAIRABLE_CONTRACT"
    DEGRADABLE_QUALITY = "DEGRADABLE_QUALITY"
    USER_DECISION_REQUIRED = "USER_DECISION_REQUIRED"
    FATAL_SYSTEM = "FATAL_SYSTEM"


class FailureAction(str, Enum):
    COMMIT = "COMMIT"
    RETRY_VERIFIER = "RETRY_VERIFIER"
    RETRY_TASK = "RETRY_TASK"
    REPAIR_CONTRACT = "REPAIR_CONTRACT"
    RECOVER_EVIDENCE = "RECOVER_EVIDENCE"
    RECOVER_ASSET = "RECOVER_ASSET"
    PATCH_PLAN = "PATCH_PLAN"
    COMMIT_WITH_WARNING = "COMMIT_WITH_WARNING"
    REGISTER_BLOCKER = "REGISTER_BLOCKER"
    FAIL_JOB = "FAIL_JOB"


class RequirementProvenance(TypedDict):
    origin: Literal[
        "explicit_user",
        "intake_inferred",
        "legacy_reconstructed",
        "legacy_task_contract",
    ]
    source_message_id: str | None
    source_field: str
    source_index: int
    derivation: str


class RequirementRecord(TypedDict):
    requirement_id: str
    text: str
    severity: Literal["hard", "soft"]
    kind: str
    status: Literal["active", "modified", "withdrawn"]
    contract_revision: int
    provenance: RequirementProvenance


class RepairAttemptRecord(TypedDict):
    repair_type: str
    attempt: int
    budget: int
    outcome: str
    diagnostic_code: str | None


class FailureDecision(TypedDict):
    failure_class: str
    subtype: str
    reason: str
    task_id: str | None
    action: str
    retryable: bool
    repair_attempt: int
    repair_budget: int
    user_blocker: bool
    requirement_ids: list[str]
    hard_requirement_ids: list[str]
    metadata: dict[str, JSONValue]


class DegradedIssueRecord(TypedDict):
    issue_id: str
    task_id: str
    task_revision: int
    failure_class: str
    subtype: str
    reason: str
    affected_claims: list[str]
    affected_requirement_ids: list[str]
    attempted_repairs: list[RepairAttemptRecord]
    final_fallback: str
    status: Literal["active", "superseded"]
    metadata: dict[str, JSONValue]


class UserBlockerRecord(TypedDict):
    blocker_id: str
    status: Literal["pending", "retry_pending", "resolved", "cancelled"]
    task_id: str
    requirement_ids: list[str]
    affected_task_ids: list[str]
    reason: str
    required_user_action: str
    available_options: list[str]
    attempted_repairs: list[RepairAttemptRecord]
    metadata: dict[str, JSONValue]


class RequirementModification(TypedDict):
    requirement_id: str
    operation: Literal["update", "withdraw"]
    previous_text: str
    new_text: str | None
    previous_severity: str
    new_severity: str | None
    previous_contract_revision: int
    new_contract_revision: int


class BlockerResolutionRecord(TypedDict):
    resolution_id: str
    blocker_id: str
    action: Literal[
        "UPLOAD_RESOURCES",
        "MODIFY_REQUIREMENT",
        "APPROVE_EXCEPTION",
        "CANCEL_JOB",
    ]
    status: Literal["applied", "retry_pending", "rejected"]
    resource_ids: list[str]
    requirement_modification: RequirementModification | None
    affected_task_ids: list[str]
    contract_revision: int
    metadata: dict[str, JSONValue]


class TaskOutcomeRecord(TypedDict):
    task_id: str
    status: Literal[
        "pending",
        "running",
        "committed",
        "degraded",
        "blocked_dependency",
        "blocked_user",
    ]
    dependency_ids: list[str]
    blocker_ids: list[str]
    task_revision: int


class FatalSystemError(TypedDict):
    failure_id: str
    failure_class: str
    subtype: str
    origin: Literal["graph", "runner"]
    component: str
    operation: str
    task_id: str | None
    diagnostic_code: str
    retryable: bool
    metadata: dict[str, JSONValue]


_ALLOWED_ACTIONS = {
    FailureClass.RETRYABLE_EXECUTION: frozenset(
        {
            FailureAction.RETRY_VERIFIER,
            FailureAction.RETRY_TASK,
            FailureAction.RECOVER_EVIDENCE,
        }
    ),
    FailureClass.REPAIRABLE_CONTRACT: frozenset(
        {
            FailureAction.REPAIR_CONTRACT,
            FailureAction.RECOVER_ASSET,
            FailureAction.PATCH_PLAN,
        }
    ),
    FailureClass.DEGRADABLE_QUALITY: frozenset(
        {FailureAction.COMMIT_WITH_WARNING}
    ),
    FailureClass.USER_DECISION_REQUIRED: frozenset(
        {FailureAction.REGISTER_BLOCKER}
    ),
    FailureClass.FATAL_SYSTEM: frozenset({FailureAction.FAIL_JOB}),
}


def allowed_failure_actions(
    failure_class: FailureClass | str,
) -> set[FailureAction]:
    """Return a copy of the closed action set for one failure class."""

    normalized = FailureClass(failure_class)
    return set(_ALLOWED_ACTIONS[normalized])


def validate_failure_action_pair(
    failure_class: FailureClass | str,
    action: FailureAction | str,
) -> None:
    """Reject workflow actions that do not belong to the classified failure."""

    normalized_class = FailureClass(failure_class)
    normalized_action = FailureAction(action)
    if normalized_action not in _ALLOWED_ACTIONS[normalized_class]:
        raise ValueError(
            "invalid failure class/action pair: "
            f"{normalized_class.value}/{normalized_action.value}"
        )


FAILURE_STATE_DEFAULTS: dict[str, Any] = {
    "requirement_registry": [],
    "failure_decision": {},
    "degraded_issue_registry": [],
    "pending_user_blockers": [],
    "blocker_resolution_registry": [],
    "resolved_user_blocker_ids": [],
    "task_outcome_registry": {},
    "fatal_system_error": {},
}


def normalize_failure_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return safe vNext defaults without mutating a current or legacy State."""

    normalized = deepcopy(state)
    list_fields = {
        "requirement_registry",
        "degraded_issue_registry",
        "pending_user_blockers",
        "blocker_resolution_registry",
    }
    for field, default in FAILURE_STATE_DEFAULTS.items():
        value = state.get(field)
        if field in list_fields:
            normalized[field] = deepcopy(value) if isinstance(value, list) else []
        elif field == "resolved_user_blocker_ids":
            normalized[field] = list(
                dict.fromkeys(
                    item.strip()
                    for item in (value if isinstance(value, list) else [])
                    if isinstance(item, str) and item.strip()
                )
            )
        elif field == "task_outcome_registry":
            normalized[field] = deepcopy(value) if isinstance(value, dict) else {}
        else:
            normalized[field] = (
                deepcopy(value) if isinstance(value, dict) else deepcopy(default)
            )
    return normalized

from copy import deepcopy

import pytest

from src.failure_semantics import (
    FAILURE_STATE_DEFAULTS,
    FailureAction,
    FailureClass,
    allowed_failure_actions,
    normalize_failure_state,
    validate_failure_action_pair,
)
from src.state import State


def test_failure_taxonomy_contains_only_vnext_classes():
    assert {item.value for item in FailureClass} == {
        "RETRYABLE_EXECUTION",
        "REPAIRABLE_CONTRACT",
        "DEGRADABLE_QUALITY",
        "USER_DECISION_REQUIRED",
        "FATAL_SYSTEM",
    }


def test_failure_action_matrix_is_closed():
    assert allowed_failure_actions(FailureClass.DEGRADABLE_QUALITY) == {
        FailureAction.COMMIT_WITH_WARNING
    }
    assert allowed_failure_actions(FailureClass.USER_DECISION_REQUIRED) == {
        FailureAction.REGISTER_BLOCKER
    }
    assert allowed_failure_actions(FailureClass.FATAL_SYSTEM) == {
        FailureAction.FAIL_JOB
    }
    with pytest.raises(ValueError, match="invalid failure class/action pair"):
        validate_failure_action_pair(
            FailureClass.DEGRADABLE_QUALITY,
            FailureAction.REGISTER_BLOCKER,
        )


def test_legacy_state_gets_safe_empty_vnext_registries_without_mutation():
    legacy = {
        "tasks": [{"task_id": "T1"}],
        "accepted_evidence_gaps": {"T1": {"accepted": True}},
        "assessment": {
            "issues": [{"description": "必须补充资料", "severity": "major"}]
        },
    }
    before = deepcopy(legacy)

    normalized = normalize_failure_state(legacy)

    assert legacy == before
    assert normalized["requirement_registry"] == []
    assert normalized["pending_user_blockers"] == []
    assert normalized["degraded_issue_registry"] == []
    assert normalized["accepted_evidence_gaps"] == legacy["accepted_evidence_gaps"]
    assert all(
        value == normalized[key] for key, value in FAILURE_STATE_DEFAULTS.items()
    )


def test_normalization_copies_mutable_registry_values():
    state = {
        "requirement_registry": [{"requirement_id": "REQ-001"}],
        "resolved_user_blocker_ids": ["BLK-1", "BLK-1", "", 1],
        "task_outcome_registry": {"T1": {"status": "committed"}},
    }

    normalized = normalize_failure_state(state)
    normalized["requirement_registry"].append({"requirement_id": "REQ-002"})
    normalized["task_outcome_registry"]["T1"]["status"] = "degraded"

    assert state["requirement_registry"] == [{"requirement_id": "REQ-001"}]
    assert state["task_outcome_registry"]["T1"]["status"] == "committed"
    assert normalized["resolved_user_blocker_ids"] == ["BLK-1"]


def test_state_declares_every_vnext_failure_field():
    assert {
        "requirement_registry",
        "failure_decision",
        "degraded_issue_registry",
        "pending_user_blockers",
        "blocker_resolution_registry",
        "resolved_user_blocker_ids",
        "task_outcome_registry",
        "fatal_system_error",
    } <= set(State.__annotations__)

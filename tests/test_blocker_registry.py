from copy import deepcopy

import pytest

from src.blocker_registry import (
    affected_task_closure,
    apply_blocker_resolution,
    build_user_blocker,
    runnable_task_ids,
    upsert_user_blocker,
)
from src.nodes import recovery as recovery_module


def _tasks():
    return [
        {"task_id": "T1", "depends_on_task_ids": []},
        {"task_id": "T2", "depends_on_task_ids": ["T1"]},
        {"task_id": "T3", "depends_on_task_ids": []},
        {"task_id": "T4", "depends_on_task_ids": ["T2", "T3"]},
    ]


def _requirement(*, allow_exception: bool = False):
    return {
        "requirement_id": "REQ-001",
        "text": "必须提供用户文件",
        "severity": "hard",
        "kind": "human_approval" if allow_exception else "user_resource",
        "status": "active",
        "contract_revision": 1,
        "provenance": {
            "origin": "explicit_user",
            "source_message_id": "m1",
            "source_field": "constraints",
            "source_index": 0,
            "derivation": "explicit",
        },
    }


def _blocker(**overrides):
    metadata = overrides.pop(
        "metadata", {"task_revision": 1, "plan_revision": 2, "attempt": 1}
    )
    return build_user_blocker(
        job_scope="job-1",
        task_id="T1",
        subtype="MISSING_RESOURCE",
        requirement_ids=["REQ-001"],
        affected_task_ids=["T1", "T2", "T4"],
        missing_resource_id="feed.csv",
        reason="MISSING_RESOURCE",
        attempted_repairs=[],
        available_options=["UPLOAD_RESOURCES", "MODIFY_REQUIREMENT", "CANCEL_JOB"],
        metadata=metadata,
        **overrides,
    )


def _state(*, blocker=None, requirement=None):
    active = blocker or _blocker()
    return {
        "tasks": _tasks(),
        "results": [
            {"task_id": "T1", "text_output": "old T1"},
            {"task_id": "T3", "text_output": "keep T3"},
        ],
        "task_revisions": {"T1": 2, "T2": 1, "T3": 4, "T4": 1},
        "requirement_registry": [requirement or _requirement()],
        "pending_user_blockers": [active],
        "blocker_resolution_registry": [],
        "resolved_user_blocker_ids": [],
        "task_outcome_registry": {
            "T1": {"task_id": "T1", "status": "blocked_user", "dependency_ids": [], "blocker_ids": [active["blocker_id"]], "task_revision": 2},
            "T2": {"task_id": "T2", "status": "blocked_dependency", "dependency_ids": ["T1"], "blocker_ids": [active["blocker_id"]], "task_revision": 1},
            "T3": {"task_id": "T3", "status": "committed", "dependency_ids": [], "blocker_ids": [], "task_revision": 4},
            "T4": {"task_id": "T4", "status": "blocked_dependency", "dependency_ids": ["T2", "T3"], "blocker_ids": [active["blocker_id"]], "task_revision": 1},
        },
    }


def test_blocker_identity_excludes_revision_and_attempt_metadata():
    first = _blocker()
    second = _blocker(metadata={"task_revision": 99, "plan_revision": 50, "attempt": 8})

    assert first["blocker_id"] == second["blocker_id"]


def test_same_blocker_upserts_in_place_across_task_rewrite():
    first = _blocker()
    second = _blocker(metadata={"task_revision": 2})

    registry = upsert_user_blocker([first], second)

    assert registry == [second]


def test_affected_closure_uses_only_explicit_dependency_edges():
    tasks = _tasks()
    tasks[2]["requirement_ids"] = ["REQ-001"]

    assert affected_task_closure(tasks, "T1") == ["T1", "T2", "T4"]


def test_runnable_selection_ignores_shared_requirement_overlap():
    tasks = _tasks()
    tasks[2]["requirement_ids"] = ["REQ-001"]
    outcomes = _state()["task_outcome_registry"]

    assert runnable_task_ids(tasks, outcomes) == []
    outcomes = deepcopy(outcomes)
    outcomes["T3"]["status"] = "pending"
    assert runnable_task_ids(tasks, outcomes) == ["T3"]


def test_upload_resolution_invalidates_only_dependency_descendants():
    state = _state()
    blocker_id = state["pending_user_blockers"][0]["blocker_id"]

    update = apply_blocker_resolution(
        state,
        blocker_id=blocker_id,
        action="UPLOAD_RESOURCES",
        resource_ids=["resource-7"],
    )

    assert [result["task_id"] for result in update["results"]] == ["T3"]
    assert update["task_revisions"] == state["task_revisions"]
    assert update["pending_user_blockers"][0]["status"] == "retry_pending"
    assert update["task_outcome_registry"]["T1"]["status"] == "pending"
    assert update["task_outcome_registry"]["T2"]["status"] == "pending"
    assert update["task_outcome_registry"]["T4"]["status"] == "pending"
    assert update["resume_task_id"] == "T1"


def test_requirement_modification_retains_id_and_increments_contract_revision():
    state = _state()
    blocker_id = state["pending_user_blockers"][0]["blocker_id"]

    update = apply_blocker_resolution(
        state,
        blocker_id=blocker_id,
        action="MODIFY_REQUIREMENT",
        requirement_update={"requirement_id": "REQ-001", "new_text": "文件可选", "new_severity": "soft"},
    )

    requirement = update["requirement_registry"][0]
    assert requirement["requirement_id"] == "REQ-001"
    assert requirement["text"] == "文件可选"
    assert requirement["severity"] == "soft"
    assert requirement["status"] == "modified"
    assert requirement["contract_revision"] == 2
    modification = update["blocker_resolution_registry"][0]["requirement_modification"]
    assert modification["previous_contract_revision"] == 1
    assert modification["new_contract_revision"] == 2


def test_approval_requires_explicit_contract_authority():
    state = _state()
    state["pending_user_blockers"][0]["available_options"].insert(
        -1, "APPROVE_EXCEPTION"
    )
    blocker_id = state["pending_user_blockers"][0]["blocker_id"]

    with pytest.raises(ValueError, match="does not allow approval"):
        apply_blocker_resolution(state, blocker_id=blocker_id, action="APPROVE_EXCEPTION")

    allowed = _state(requirement=_requirement(allow_exception=True))
    allowed["pending_user_blockers"][0]["available_options"].insert(
        -1, "APPROVE_EXCEPTION"
    )
    update = apply_blocker_resolution(allowed, blocker_id=blocker_id, action="APPROVE_EXCEPTION")
    assert update["pending_user_blockers"][0]["status"] == "resolved"
    assert update["resolved_user_blocker_ids"] == [blocker_id]


def test_duplicate_resolution_is_idempotent_and_conflict_is_rejected():
    state = _state()
    blocker_id = state["pending_user_blockers"][0]["blocker_id"]
    first = apply_blocker_resolution(
        state,
        blocker_id=blocker_id,
        action="UPLOAD_RESOURCES",
        resource_ids=["resource-7"],
    )
    second = apply_blocker_resolution(
        {**state, **first},
        blocker_id=blocker_id,
        action="UPLOAD_RESOURCES",
        resource_ids=["resource-7"],
    )

    assert second == first
    with pytest.raises(ValueError, match="conflicting resolution"):
        apply_blocker_resolution(
            {**state, **first},
            blocker_id=blocker_id,
            action="MODIFY_REQUIREMENT",
            requirement_update={"requirement_id": "REQ-001", "new_text": "changed"},
        )


def test_resolution_cannot_escape_blocker_action_or_requirement_scope():
    state = _state()
    blocker_id = state["pending_user_blockers"][0]["blocker_id"]

    with pytest.raises(ValueError, match="not available"):
        apply_blocker_resolution(
            state, blocker_id=blocker_id, action="APPROVE_EXCEPTION"
        )
    with pytest.raises(ValueError, match="outside blocker scope"):
        apply_blocker_resolution(
            state,
            blocker_id=blocker_id,
            action="MODIFY_REQUIREMENT",
            requirement_update={
                "requirement_id": "REQ-OTHER",
                "new_text": "not allowed",
            },
        )


def test_consolidated_interrupt_applies_blocker_keyed_partial_submission(monkeypatch):
    state = _state()
    first = state["pending_user_blockers"][0]
    second = build_user_blocker(
        job_scope="job-1",
        task_id="T3",
        subtype="EXPLICIT_APPROVAL",
        requirement_ids=["REQ-001"],
        affected_task_ids=["T3", "T4"],
        missing_resource_id=None,
        reason="EXPLICIT_APPROVAL",
        attempted_repairs=[],
        available_options=["MODIFY_REQUIREMENT", "CANCEL_JOB"],
        metadata={},
    )
    state["pending_user_blockers"] = [first, second]
    state["pending_user_action"] = {
        "category": "CONSOLIDATED_BLOCKERS",
        "blockers": [first, second],
    }
    captured = {}

    def resume(payload):
        captured.update(payload)
        return {
            "resolutions": [
                {
                    "blocker_id": first["blocker_id"],
                    "action": "UPLOAD_RESOURCES",
                    "resource_ids": ["resource-7"],
                }
            ]
        }

    monkeypatch.setattr(recovery_module, "interrupt", resume)
    update = recovery_module.needs_user_input(state, {})

    assert captured["type"] == "needs_user_input"
    assert len(captured["blockers"]) == 2
    assert update["workflow_action"] == "REWORK"
    assert update["cursor"] == 0
    assert update["pending_user_blockers"][0]["status"] == "retry_pending"
    assert update["pending_user_blockers"][1]["status"] == "pending"

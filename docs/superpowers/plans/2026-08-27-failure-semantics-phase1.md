# Failure Semantics Phase 1 Implementation Plan

**Status:** Completed

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the unified failure taxonomy, closed action matrix, exact serializable registry schemas, and safe default normalization without changing recovery routing behavior.

**Architecture:** A new `src/failure_semantics.py` domain module owns enums, TypedDict contracts, action validation, and nonmutating compatibility defaults. `src/state.py` declares the corresponding optional LangGraph fields. Existing `IssueCategory`, `WorkflowAction`, graph edges, and recovery decisions remain unchanged in Phase 1; later phases consume the new contract incrementally.

**Tech Stack:** Python 3.11+, TypedDict, Enum, pytest, LangGraph TypedDict State

## Global Constraints

- Keep the existing LangGraph node topology.
- Do not modify SQLite schema or move workflow authority to `SqliteStore`.
- All new State values must be checkpoint-serializable plain dictionaries, lists, strings, integers, booleans, or null.
- Preserve existing deterministic citation/source validation, ConceptGraph and length-rewrite budgets, LLM observability, and legacy recovery fields.
- Do not change runtime routing behavior in Phase 1.
- Production code is written only after the corresponding test has failed for the expected missing-contract reason.

---

### Task 1: Unified enums and closed action matrix

**Files:**
- Create: `src/failure_semantics.py`
- Create: `tests/test_failure_semantics.py`

**Interfaces:**
- Consumes: no recovery-node implementation details.
- Produces: `FailureClass`, `FailureAction`, `allowed_failure_actions()`, and `validate_failure_action_pair()` for DecisionPolicy phases.

- [ ] **Step 1: Write the failing enum and matrix tests**

```python
import pytest

from src.failure_semantics import (
    FailureAction,
    FailureClass,
    allowed_failure_actions,
    validate_failure_action_pair,
)


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
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_failure_semantics.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.failure_semantics'`.

- [ ] **Step 3: Implement the enums and closed matrix**

```python
from __future__ import annotations

from enum import Enum


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


def allowed_failure_actions(failure_class: FailureClass | str) -> set[FailureAction]:
    normalized = FailureClass(failure_class)
    return set(_ALLOWED_ACTIONS[normalized])


def validate_failure_action_pair(
    failure_class: FailureClass | str,
    action: FailureAction | str,
) -> None:
    normalized_class = FailureClass(failure_class)
    normalized_action = FailureAction(action)
    if normalized_action not in _ALLOWED_ACTIONS[normalized_class]:
        raise ValueError(
            "invalid failure class/action pair: "
            f"{normalized_class.value}/{normalized_action.value}"
        )
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_failure_semantics.py -q`

Expected: `2 passed`.

---

### Task 2: Exact registry schemas and safe compatibility defaults

**Files:**
- Modify: `src/failure_semantics.py`
- Modify: `tests/test_failure_semantics.py`

**Interfaces:**
- Consumes: the Phase 1 enum values.
- Produces: TypedDict contracts and `normalize_failure_state(state)` for old/new checkpoints.

- [ ] **Step 1: Write failing normalization tests**

```python
from copy import deepcopy

from src.failure_semantics import FAILURE_STATE_DEFAULTS, normalize_failure_state


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
    assert all(value == normalized[key] for key, value in FAILURE_STATE_DEFAULTS.items())


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
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_failure_semantics.py -q`

Expected: import fails because `FAILURE_STATE_DEFAULTS` and `normalize_failure_state` do not exist.

- [ ] **Step 3: Add exact TypedDict schemas and normalization**

Implement the TypedDicts from Section 7 of the approved specification:

```python
from copy import deepcopy
from typing import Any, Literal, TypedDict


JSONValue = Any


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
```

Add these exact registry definitions:

```python
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
```

Then implement:

```python
FAILURE_STATE_DEFAULTS = {
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
    normalized = deepcopy(state)
    for field, default in FAILURE_STATE_DEFAULTS.items():
        value = state.get(field)
        if field in {
            "requirement_registry",
            "degraded_issue_registry",
            "pending_user_blockers",
            "blocker_resolution_registry",
        }:
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
            normalized[field] = deepcopy(value) if isinstance(value, dict) else deepcopy(default)
    return normalized
```

Phase 1 intentionally returns an empty requirement registry for legacy State;
authoritative reconstruction is Phase 2. It must never promote assessment prose
to a hard requirement.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_failure_semantics.py -q`

Expected: `4 passed`.

---

### Task 3: Declare checkpoint-serializable State fields

**Files:**
- Modify: `src/state.py`
- Modify: `tests/test_failure_semantics.py`

**Interfaces:**
- Consumes: exact TypedDict contracts from `src.failure_semantics`.
- Produces: optional LangGraph State annotations for all vNext registries.

- [ ] **Step 1: Write the failing State annotation test**

```python
from src.state import State


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
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_failure_semantics.py::test_state_declares_every_vnext_failure_field -q`

Expected: assertion fails with the new field names missing.

- [ ] **Step 3: Add the optional State annotations**

Import the TypedDict record types only for annotations and add:

```python
requirement_registry: List[RequirementRecord]
failure_decision: FailureDecision
degraded_issue_registry: List[DegradedIssueRecord]
pending_user_blockers: List[UserBlockerRecord]
blocker_resolution_registry: List[BlockerResolutionRecord]
resolved_user_blocker_ids: List[str]
task_outcome_registry: Dict[str, TaskOutcomeRecord]
fatal_system_error: FatalSystemError
```

State remains `total=False`, so old checkpoints do not require these keys.

- [ ] **Step 4: Run Phase 1 focused and compatibility tests**

Run: `.venv/bin/python -m pytest tests/test_failure_semantics.py tests/test_recovery_compatibility.py tests/test_graph_state_contract.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Run diff validation**

Run: `git diff --check`

Expected: no output and exit code 0.

---

### Task 4: Phase 1 checkpoint commit

**Files:**
- Modify: `docs/superpowers/specs/2026-08-27-failure-semantics-recovery-vnext-design.md`
- Create: `docs/superpowers/plans/2026-08-27-failure-semantics-phase1.md`
- Create: `src/failure_semantics.py`
- Modify: `src/state.py`
- Create: `tests/test_failure_semantics.py`

**Interfaces:**
- Consumes: Tasks 1–3 passing tests.
- Produces: one reviewable Phase 1 commit without runtime routing changes.

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -q`

Expected: the baseline 679 tests plus Phase 1 tests pass.

- [ ] **Step 2: Commit the Phase 1 boundary**

```bash
git add docs/superpowers/specs/2026-08-27-failure-semantics-recovery-vnext-design.md \
  docs/superpowers/plans/2026-08-27-failure-semantics-phase1.md \
  src/failure_semantics.py src/state.py tests/test_failure_semantics.py
git commit -m "feat: add unified failure semantics contract"
```

- [ ] **Step 3: Record the checkpoint**

Report the commit SHA, focused/full pytest results, and confirm that existing
recovery routes are unchanged. Continue to the Phase 2 requirement/dependency
plan only after this checkpoint is green.

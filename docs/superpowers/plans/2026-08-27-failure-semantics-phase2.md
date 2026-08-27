# Failure Semantics Phase 2 Implementation Plan

**Status:** Completed

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist canonical requirement provenance and require every new Planner task to declare requirement linkage and explicit task dependencies.

**Architecture:** A pure `src/requirements.py` module builds stable Intake requirement records and conservatively reconstructs legacy contracts. Intake writes the registry into both State and `INTAKE_SUMMARY`. Planner validates `requirement_ids` against that registry and validates `depends_on_task_ids` as a real acyclic graph; legacy task normalization preserves historical serial order.

**Tech Stack:** Python, TypedDict records from `src.failure_semantics`, LangGraph State, pytest

## Global Constraints

- Requirement overlap is never a dependency edge.
- Planner may link but may not invent requirements.
- Issue prose is never used to reconstruct a hard legacy requirement.
- New plans require `requirement_ids` and `depends_on_task_ids`; legacy checkpoints receive conservative normalization.
- Synthesis explicitly depends on every task result it consumes.
- No DecisionPolicy or recovery routing changes occur in Phase 2.

---

### Task 1: Canonical requirement registry and provenance

**Files:**
- Create: `src/requirements.py`
- Create: `tests/test_requirements.py`

**Interfaces:**
- Produces: `build_requirement_registry(parsed_request, raw_request, source_message_id)` and `reconstruct_legacy_requirements(state)`.

- [ ] **Step 1: Write failing tests**

Tests assert:

```python
def test_explicit_mandatory_constraint_is_hard_with_provenance():
    records = build_requirement_registry(
        {"constraints": ["必须使用用户提供的质量规程"]},
        "必须使用用户提供的质量规程",
        "msg-1",
    )
    assert records[0]["requirement_id"] == "REQ-001"
    assert records[0]["severity"] == "hard"
    assert records[0]["provenance"] == {
        "origin": "explicit_user",
        "source_message_id": "msg-1",
        "source_field": "constraints",
        "source_index": 0,
        "derivation": "deterministic_explicit_hard_marker",
    }


def test_inferred_quality_and_length_targets_default_soft():
    records = build_requirement_registry(
        {"constraints": ["报告表达清晰"], "doc_length": "约5000字"},
        "请写一份报告",
        "msg-2",
    )
    assert [record["severity"] for record in records] == ["soft", "soft"]
    assert [record["provenance"]["origin"] for record in records] == [
        "intake_inferred",
        "intake_inferred",
    ]


def test_legacy_reconstruction_never_uses_assessment_prose_as_hard_authority():
    records = reconstruct_legacy_requirements(
        {
            "tasks": [{"task_id": "T1"}],
            "assessment": {"issues": [{"description": "必须上传秘密文件"}]},
        }
    )
    assert records == []
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_requirements.py -q`

Expected: `ModuleNotFoundError: No module named 'src.requirements'`.

- [ ] **Step 3: Implement minimal deterministic builders**

The builder iterates canonical `constraints`, then non-`不限` `doc_length`, then
literal user-specified sections. It emits `REQ-001`, `REQ-002`, and so on;
deduplicates identical canonical text; sets `contract_revision=1` and
`status="active"`; marks hard only when an explicit hard marker occurs in the
authoritative raw clause; and records the exact provenance fields.

Legacy reconstruction reads only the latest persisted `INTAKE_SUMMARY` and
explicit structural task fields. It emits `LEGACY-REQ-*` or `LEGACY-TASK-*`
records and ignores assessment/verifier prose.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_requirements.py -q`

Expected: all requirement tests pass.

---

### Task 2: Intake persists the canonical requirement registry

**Files:**
- Modify: `src/nodes/intake.py`
- Modify: `tests/test_planner_contract.py`

**Interfaces:**
- Consumes: `build_requirement_registry`.
- Produces: `requirement_registry` in the Intake State update and `requirements` in `INTAKE_SUMMARY`.

- [ ] **Step 1: Write failing Intake tests**

Add tests that call `build_task_spec` with a supplied registry and assert the
same records are returned under the State field `requirement_registry` and the
serialized `INTAKE_SUMMARY.requirements`. Add a node test with a HumanMessage ID
and assert provenance uses that ID.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_planner_contract.py -k requirement_registry -q`

Expected: assertions fail because Intake does not emit the registry.

- [ ] **Step 3: Implement Intake integration**

Extend `build_task_spec(parsed_request, resources, metadata=None,
requirement_registry=None)` and include a copied registry in State and message
payload. In `intake`, derive the latest Human
message ID, build the registry from canonical parsed request and current raw
input, and pass it to `build_task_spec`. No extra LLM call is allowed.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_planner_contract.py -k 'intake or requirement' -q`

Expected: selected tests pass.

---

### Task 3: Planner task requirement and dependency contract

**Files:**
- Modify: `src/nodes/planner.py`
- Modify: `src/prompts/planner_to_worker.md`
- Modify: `src/prompts/planner_intake_replan.md`
- Modify: `src/prompts/planner_replan.md`
- Modify: `tests/test_planner_contract.py`
- Modify: `tests/test_recovery_compatibility.py`

**Interfaces:**
- Consumes: Intake `requirements` and task IDs.
- Produces: strict `requirement_ids`, strict `depends_on_task_ids`, `validate_task_dependencies(tasks)`, and `normalize_legacy_task_contracts(tasks)`.

- [ ] **Step 1: Write failing Planner contract tests**

Tests assert new generated tasks reject a missing field, unknown requirement
ID, missing dependency target, self dependency, forward dependency in initial
plans, and cycles. Tests assert a valid chain passes, independent tasks may use
`[]`, synthesis names every consumed prior task, and shared requirement IDs do
not create an implicit dependency.

Legacy normalization tests assert:

```python
assert normalize_legacy_task_contracts(
    [
        {"task_id": "T1", "task_type": "analysis"},
        {"task_id": "T2", "task_type": "analysis"},
        {"task_id": "T3", "task_type": "synthesis"},
    ]
) == [
    {"task_id": "T1", "task_type": "analysis", "requirement_ids": [], "depends_on_task_ids": []},
    {"task_id": "T2", "task_type": "analysis", "requirement_ids": [], "depends_on_task_ids": ["T1"]},
    {"task_id": "T3", "task_type": "synthesis", "requirement_ids": [], "depends_on_task_ids": ["T1", "T2"]},
]
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_planner_contract.py tests/test_recovery_compatibility.py -k 'depend or requirement_id or legacy_task_contract' -q`

Expected: tests fail because the fields and validators do not exist.

- [ ] **Step 3: Implement Planner validation and propagation**

Add both fields to generated/replacement contract sets. Validate both as unique
nonempty-string lists. Validate requirement IDs against the Intake registry
passed through `policy_context`. Validate dependency graph separately from
requirements. Preserve these fields during replacement normalization and
stable task-ID reassignment, remapping temporary dependency IDs to assigned
stable IDs.

Pass `requirements` into every initial/refinement/replan prompt value. Update
all three prompts to say each task has 14 exact fields, show both fields in the
JSON example, prohibit requirement invention, and require explicit dependency
edges. Update shared test task factories to include the two fields.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_planner_contract.py tests/test_recovery_compatibility.py -q`

Expected: all selected tests pass.

---

### Task 4: Phase 2 regression and checkpoint

**Files:** all Phase 2 files above plus this plan.

**Interfaces:**
- Produces: a behavior-preserving Planner/Intake contract checkpoint for Phase 3 policy consumption.

- [ ] **Step 1: Run focused compatibility suites**

Run: `.venv/bin/python -m pytest tests/test_requirements.py tests/test_planner_contract.py tests/test_recovery_compatibility.py tests/test_graph_state_contract.py -q`

- [ ] **Step 2: Run the full suite and diff check**

Run: `.venv/bin/python -m pytest -q`

Run: `git diff --check`

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-08-27-failure-semantics-phase2.md \
  src/requirements.py src/nodes/intake.py src/nodes/planner.py \
  src/prompts/planner_to_worker.md src/prompts/planner_intake_replan.md \
  src/prompts/planner_replan.md tests/test_requirements.py \
  tests/test_planner_contract.py tests/test_recovery_compatibility.py
git commit -m "feat: add requirement and task dependency contracts"
```

# Stable Report Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the existing multi-task report pipeline so every planned task advances through an auditable task ledger, versioned Worker artifacts, independent quality review, bounded local recovery, and truthful report delivery.

**Architecture:** Keep the existing LangGraph report workflow, but insert a deterministic `TaskController` as the sole owner of task progression and an `ArtifactCommit` boundary between Worker and quality review. Replace Verifier routing semantics with a structured `QualityReview -> DecisionPolicy` contract, then harden Summarizer/PDF output and expose the new state in Streamlit without turning the product into a full case-management platform.

**Tech Stack:** Python 3.13, LangGraph 1.0.10, LangChain 1.x, Pydantic 2.12, SQLite Checkpointer/Store, Streamlit 1.50, ReportLab 4.4, pytest 8.4.

## Global Constraints

- Preserve the current `codex/sqlite-checkpoint-store` branch and existing DeepSeek, hybrid RAG, evidence, concept-graph, and SQLite behavior.
- Keep the report workflow serial; do not introduce parallel DAG scheduling.
- Automatic execution must never trigger a full replan. Full replan remains a user-confirmed Planner operation.
- Keep validated local PlanPatch recovery; it may affect only declared current or unfinished tasks and must return to TaskController.
- No task may advance until its active Artifact has a PASS ReviewRecord.
- Retry exhaustion must enter HumanReview/`NEEDS_USER_INPUT`; never auto-accept defective content.
- Public web access must be explicitly authorized for the job and still explicitly enabled on the task.
- Without enterprise SOP or approved bounds, generated advice remains directional and never directly controls plant systems.
- Old SQLite checkpoints must restore without deleting either database.
- Use TDD for every task and commit only the files listed for that task.

---

## File Structure

New focused modules:

- `src/workflow_records.py`: pure task-ledger, Artifact, ReviewRecord, and compatibility helpers.
- `src/workflow_store.py`: LangGraph Store namespaces and record persistence.
- `src/nodes/task_controller.py`: deterministic task selection and attempt setup.
- `src/nodes/artifact_commit.py`: converts the legacy Worker result into a versioned Artifact.
- `src/quality/__init__.py`: quality package exports.
- `src/quality/models.py`: Pydantic contracts for issues, dimensions, and review records.
- `src/quality/validators.py`: deterministic Artifact/task validation.
- `src/nodes/quality_review.py`: LLM-assisted quality assessment with deterministic validation.
- `src/prompts/quality_review.md`: quality-review-only prompt.

Existing modules with scoped changes:

- `src/state.py`: add ledger, artifact, review, report, execution, and web-authorization fields.
- `src/nodes/planner.py`: stop advancing `cursor`; initialize ledgers; enforce job-level web authorization.
- `src/nodes/planner.py` and `src/nodes/recovery.py`: preserve user-only full replan and validated local patch semantics.
- `src/recovery/policy.py`: consume ReviewRecord categories, update task status, remove `ACCEPT_WITH_WARNING` progression.
- `src/nodes/verifier_manual.py`: emit manual ReviewRecords and return progression to TaskController.
- `src/graph.py`: install TaskController, ArtifactCommit, QualityReview, and new routes.
- `src/nodes/summarizer_v2.py`: gate on passed tasks, disable JSON mode, and return a truthful ReportManifest.
- `src/utils/md_to_pdf.py`: parse escaped pipes and constrain table width.
- `src/ui_projection.py`: provide task progress and report status view models.
- `src/job_store.py`: persist immutable per-job public-web authorization.
- `app.py`: render progress/reviews/report status and accept supported document/data uploads.

New tests:

- `tests/test_workflow_records.py`
- `tests/test_task_controller.py`
- `tests/test_artifact_commit.py`
- `tests/test_quality_review.py`
- `tests/test_report_generation.py`
- `tests/test_pdf_tables.py`
- `tests/test_pipeline_e2e.py`

Existing tests updated where behavior intentionally changes:

- `tests/test_recovery_policy.py`
- `tests/test_recovery_graph.py`
- `tests/test_auto_verifier_routing.py`
- `tests/test_verifier_routing.py`
- `tests/test_recovery_compatibility.py`
- `tests/test_planner_helpers.py`
- `tests/test_ui_projection.py`

---

### Task 1: Add the Workflow Record Contracts and Compatibility Migration

**Files:**
- Create: `src/workflow_records.py`
- Create: `src/workflow_store.py`
- Modify: `src/state.py:51-111`
- Test: `tests/test_workflow_records.py`

**Interfaces:**
- Produces: `TaskStatus`, `ensure_task_records(state)`, `first_runnable_task(tasks, task_records)`, `all_tasks_passed(tasks, task_records)`, `set_task_status(records, task_id, status, **changes)`, `make_execution_id(job_id, task_id, attempt_no)`, `build_artifact(state, current_result)`, and `WorkflowRecordStore`.
- Consumes: legacy `tasks`, `cursor`, `results`, and `current_result` dictionaries.

- [ ] **Step 1: Write failing record migration and idempotency tests**

```python
# tests/test_workflow_records.py
from src.workflow_records import (
    all_tasks_passed,
    build_artifact,
    ensure_task_records,
    first_runnable_task,
    make_execution_id,
)


def test_legacy_results_bootstrap_passed_task_records():
    state = {
        "tasks": [{"task_id": "T1"}, {"task_id": "T2"}],
        "results": [{"task_id": "T1", "text_output": "accepted"}],
    }
    records = ensure_task_records(state)
    assert records["T1"]["status"] == "PASSED"
    assert records["T2"]["status"] == "PENDING"
    assert first_runnable_task(state["tasks"], records) == (1, state["tasks"][1])
    assert all_tasks_passed(state["tasks"], records) is False


def test_artifact_identity_is_stable_for_a_replayed_attempt():
    state = {
        "job_id": "job-1",
        "current_execution_id": make_execution_id("job-1", "T2", 3),
        "task_records": {"T2": {"attempt_count": 3}},
        "active_artifact_ids": {"T2": "artifact-old"},
    }
    current = {"task_id": "T2", "text_output": "new", "citations": []}
    first = build_artifact(state, current)
    second = build_artifact(state, current)
    assert first["artifact_id"] == second["artifact_id"]
    assert first["attempt_no"] == 3
    assert first["supersedes"] == "artifact-old"
```

- [ ] **Step 2: Run the tests and verify the contracts do not exist**

Run: `pytest tests/test_workflow_records.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'src.workflow_records'`.

- [ ] **Step 3: Implement pure record helpers and Store persistence**

```python
# src/workflow_records.py
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

TaskStatus = Literal[
    "PENDING", "RUNNING", "REVISE_REQUIRED", "EVIDENCE_REQUIRED", "PASSED", "BLOCKED"
]


def ensure_task_records(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    existing = {
        str(key): dict(value)
        for key, value in (state.get("task_records") or {}).items()
        if isinstance(value, dict)
    }
    passed = {
        str(item.get("task_id"))
        for item in state.get("results") or []
        if isinstance(item, dict) and item.get("task_id") is not None
    }
    records: dict[str, dict[str, Any]] = {}
    for sequence, task in enumerate(state.get("tasks") or []):
        task_id = str(task.get("task_id") or f"T{sequence + 1}")
        prior = existing.get(task_id, {})
        records[task_id] = {
            **prior,
            "task_id": task_id,
            "sequence": sequence,
            "status": "PASSED" if task_id in passed else prior.get("status", "PENDING"),
            "attempt_count": int(prior.get("attempt_count", 0) or 0),
            "active_artifact_id": prior.get("active_artifact_id"),
            "dependencies": list(task.get("dependencies") or prior.get("dependencies") or []),
        }
    return records


def first_runnable_task(tasks, task_records):
    passed = {task_id for task_id, record in task_records.items() if record.get("status") == "PASSED"}
    for index, task in enumerate(tasks or []):
        task_id = str(task.get("task_id") or f"T{index + 1}")
        record = task_records[task_id]
        if record.get("status") == "PASSED":
            continue
        if all(str(dep) in passed for dep in record.get("dependencies") or []):
            return index, task
    return None


def all_tasks_passed(tasks, task_records) -> bool:
    return bool(tasks) and all(
        task_records.get(str(task.get("task_id")), {}).get("status") == "PASSED"
        for task in tasks
    )


def set_task_status(records, task_id, status: TaskStatus, **changes):
    updated = {key: dict(value) for key, value in records.items()}
    updated[task_id] = {**updated[task_id], "status": status, **changes}
    return updated


def make_execution_id(job_id: str, task_id: str, attempt_no: int) -> str:
    digest = hashlib.sha256(f"{job_id}|{task_id}|{attempt_no}".encode()).hexdigest()[:24]
    return f"execution_{digest}"


def build_artifact(state, current_result):
    task_id = str(current_result["task_id"])
    execution_id = str(state["current_execution_id"])
    artifact_id = "artifact_" + hashlib.sha256(execution_id.encode()).hexdigest()[:24]
    attempt_no = int(state["task_records"][task_id]["attempt_count"])
    return {
        **dict(current_result),
        "artifact_id": artifact_id,
        "task_id": task_id,
        "attempt_no": attempt_no,
        "artifact_type": "report_section",
        "producer": "worker",
        "content": current_result.get("text_output", ""),
        "evidence_refs": list(current_result.get("citations") or []),
        "source_scope": list(current_result.get("sources_used") or []),
        "created_at": current_result.get("generated_at") or datetime.now().astimezone().isoformat(),
        "supersedes": (state.get("active_artifact_ids") or {}).get(task_id),
        "execution_id": execution_id,
    }
```

```python
# src/workflow_store.py
from __future__ import annotations

from langgraph.store.base import BaseStore


class WorkflowRecordStore:
    def __init__(self, store: BaseStore, user_id: str, job_id: str):
        if not user_id or not job_id:
            raise ValueError("user_id and job_id are required")
        self.store = store
        self.root = (user_id, "report_jobs", job_id)

    def _put(self, record_type: str, key: str, value: dict) -> None:
        self.store.put((*self.root, record_type), key, value, index=False)

    def put_artifact(self, artifact: dict) -> None:
        self._put("artifacts", str(artifact["artifact_id"]), artifact)

    def put_review(self, review: dict) -> None:
        self._put("reviews", str(review["review_id"]), review)

    def put_report_manifest(self, manifest: dict) -> None:
        self._put("reports", str(manifest["report_id"]), manifest)
```

Add the following exact fields to `State`: `task_records: Dict[str, Dict[str, Any]]`, `artifacts: Dict[str, Dict[str, Any]]`, `active_artifact_ids: Dict[str, str]`, `review_record: Dict[str, Any]`, `review_records: List[Dict[str, Any]]`, `report_manifest: Dict[str, Any]`, `current_execution_id: str`, `controller_action: str`, `worker_retry_count: Dict[str, int]`, and `web_authorized: bool`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_workflow_records.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the record layer**

```bash
git add src/workflow_records.py src/workflow_store.py src/state.py tests/test_workflow_records.py
git commit -m "feat: add auditable workflow records"
```

### Task 2: Make TaskController the Sole Owner of Progression

**Files:**
- Create: `src/nodes/task_controller.py`
- Modify: `src/nodes/planner.py:720-805, 1025-1125`
- Modify: `src/graph.py:19-159`
- Test: `tests/test_task_controller.py`
- Update: `tests/test_planner_helpers.py`

**Interfaces:**
- Consumes: `ensure_task_records`, `first_runnable_task`, `all_tasks_passed`, and `make_execution_id` from Task 1.
- Produces: `task_controller(state, config=None) -> dict`, `route_task_controller(state, config=None) -> Literal["Worker", "Summarizer", "NeedsUserInput"]`.

- [ ] **Step 1: Write failing controller tests for nine-task progression and blocking**

```python
# tests/test_task_controller.py
from src.nodes.task_controller import route_task_controller, task_controller


def _state(count=9):
    return {
        "job_id": "job-1",
        "tasks": [{"task_id": f"T{i}", "task_name": f"Task {i}"} for i in range(1, count + 1)],
        "results": [],
        "task_records": {},
    }


def test_controller_selects_first_unpassed_task_without_planner_messages():
    state = _state()
    first = task_controller(state)
    assert first["current_task"]["task_id"] == "T1"
    assert first["cursor"] == 0
    assert first["task_records"]["T1"]["attempt_count"] == 1
    assert route_task_controller({**state, **first}) == "Worker"


def test_controller_never_summarizes_with_an_unpassed_task():
    state = _state(3)
    state["task_records"] = {
        "T1": {"task_id": "T1", "sequence": 0, "status": "PASSED", "attempt_count": 1},
        "T2": {"task_id": "T2", "sequence": 1, "status": "BLOCKED", "attempt_count": 2},
        "T3": {"task_id": "T3", "sequence": 2, "status": "PENDING", "attempt_count": 0},
    }
    update = task_controller(state)
    assert update["controller_action"] == "NEEDS_USER_INPUT"
    assert route_task_controller({**state, **update}) == "NeedsUserInput"
```

- [ ] **Step 2: Run the tests and verify TaskController is missing**

Run: `pytest tests/test_task_controller.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.nodes.task_controller'`.

- [ ] **Step 3: Implement deterministic selection and remove Planner progression**

```python
# src/nodes/task_controller.py
from __future__ import annotations

from typing import Any

from src.workflow_records import (
    all_tasks_passed,
    ensure_task_records,
    first_runnable_task,
    make_execution_id,
    set_task_status,
)


def task_controller(state, config=None, **kwargs) -> dict[str, Any]:
    tasks = list(state.get("tasks") or [])
    records = ensure_task_records(state)
    if all_tasks_passed(tasks, records):
        return {"task_records": records, "controller_action": "SUMMARIZE"}

    selected = first_runnable_task(tasks, records)
    if selected is None:
        return {"task_records": records, "controller_action": "NEEDS_USER_INPUT"}
    index, task = selected
    task_id = str(task["task_id"])
    if records[task_id].get("status") == "BLOCKED":
        return {"task_records": records, "current_task": task, "cursor": index, "controller_action": "NEEDS_USER_INPUT"}

    attempt_no = int(records[task_id].get("attempt_count", 0)) + 1
    execution_id = make_execution_id(str(state.get("job_id") or "job"), task_id, attempt_no)
    records = set_task_status(records, task_id, "RUNNING", attempt_count=attempt_no)
    return {
        "task_records": records,
        "current_task": task,
        "cursor": index,
        "current_execution_id": execution_id,
        "current_result": {},
        "controller_action": "DISPATCH",
    }


def route_task_controller(state, config=None, **kwargs) -> str:
    action = str(state.get("controller_action") or "NEEDS_USER_INPUT")
    return {"DISPATCH": "Worker", "SUMMARIZE": "Summarizer"}.get(action, "NeedsUserInput")
```

Change Planner so `PROCEED` never increments `cursor`; initial and confirmed plans initialize `task_records` through `ensure_task_records`. In `graph.py`, route Planner/PlannerConfirm to TaskController and route TaskController to Worker, Summarizer, or NeedsUserInput. Move the existing NeedsUserInput node registration outside the auto-only branch so it serves as the shared HumanReview gate in both review modes. Do not yet remove compatibility `PLAN_RESULT` messages because Worker still reads them.

- [ ] **Step 4: Run controller and Planner regression tests**

Run: `pytest tests/test_task_controller.py tests/test_planner_helpers.py tests/test_recovery_compatibility.py -q`

Expected: PASS; assertions that expected Planner to increment cursor are updated to assert TaskController progression instead.

- [ ] **Step 5: Commit deterministic progression**

```bash
git add src/nodes/task_controller.py src/nodes/planner.py src/graph.py tests/test_task_controller.py tests/test_planner_helpers.py tests/test_recovery_compatibility.py
git commit -m "feat: make task controller own progression"
```

### Task 3: Commit Versioned Worker Artifacts and Mirror Them to LangGraph Store

**Files:**
- Create: `src/nodes/artifact_commit.py`
- Modify: `src/graph.py`
- Test: `tests/test_artifact_commit.py`
- Update: `tests/test_report_evidence_integration.py`

**Interfaces:**
- Consumes: `build_artifact` and `WorkflowRecordStore` from Task 1, plus legacy Worker `current_result`.
- Produces: `artifact_commit(state, store=None) -> dict` with `current_result`, `artifacts`, `active_artifact_ids`, and updated `task_records`.

- [ ] **Step 1: Write failing Artifact commit tests**

```python
# tests/test_artifact_commit.py
from src.nodes.artifact_commit import artifact_commit


def test_artifact_commit_versions_without_overwriting_history():
    state = {
        "user_id": "u1",
        "job_id": "j1",
        "current_execution_id": "execution-2",
        "current_result": {"task_id": "T1", "text_output": "revision", "citations": []},
        "task_records": {"T1": {"task_id": "T1", "attempt_count": 2, "status": "RUNNING"}},
        "artifacts": {"artifact-old": {"artifact_id": "artifact-old", "task_id": "T1"}},
        "active_artifact_ids": {"T1": "artifact-old"},
    }
    update = artifact_commit(state)
    artifact = update["current_result"]
    assert artifact["supersedes"] == "artifact-old"
    assert update["active_artifact_ids"]["T1"] == artifact["artifact_id"]
    assert set(update["artifacts"]) == {"artifact-old", artifact["artifact_id"]}
    assert update["task_records"]["T1"]["active_artifact_id"] == artifact["artifact_id"]
```

- [ ] **Step 2: Run the test and verify the node is missing**

Run: `pytest tests/test_artifact_commit.py -q`

Expected: FAIL with missing `src.nodes.artifact_commit`.

- [ ] **Step 3: Implement ArtifactCommit and insert it after Worker**

```python
# src/nodes/artifact_commit.py
from __future__ import annotations

from langgraph.store.base import BaseStore

from src.workflow_records import build_artifact, set_task_status
from src.workflow_store import WorkflowRecordStore


def artifact_commit(state, store: BaseStore | None = None, **kwargs):
    current = state.get("current_result") or {}
    if not current or not current.get("task_id"):
        raise ValueError("Worker completed without a task-scoped current_result")
    artifact = build_artifact(state, current)
    task_id = str(artifact["task_id"])
    artifacts = {**(state.get("artifacts") or {}), artifact["artifact_id"]: artifact}
    active = {**(state.get("active_artifact_ids") or {}), task_id: artifact["artifact_id"]}
    records = set_task_status(
        state["task_records"], task_id, "RUNNING", active_artifact_id=artifact["artifact_id"]
    )
    if store is not None:
        WorkflowRecordStore(store, state["user_id"], state["job_id"]).put_artifact(artifact)
    return {
        "current_result": artifact,
        "artifacts": artifacts,
        "active_artifact_ids": active,
        "task_records": records,
    }
```

Change `Worker -> Verifier` to `Worker -> ArtifactCommit -> QualityReview/Verifier`. Keep `results` reserved for passed Artifacts only; `all_results` remains a compatibility trace until later cleanup.

- [ ] **Step 4: Run Artifact and evidence tests**

Run: `pytest tests/test_artifact_commit.py tests/test_report_evidence_integration.py tests/test_worker_tool_boundaries.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Artifact versioning**

```bash
git add src/nodes/artifact_commit.py src/graph.py tests/test_artifact_commit.py tests/test_report_evidence_integration.py
git commit -m "feat: version worker artifacts"
```

### Task 4: Replace Verifier Output with Independent QualityReview Records

**Files:**
- Create: `src/quality/__init__.py`
- Create: `src/quality/models.py`
- Create: `src/quality/validators.py`
- Create: `src/nodes/quality_review.py`
- Create: `src/prompts/quality_review.md`
- Modify: `src/nodes/verifier.py` to compatibility re-export only
- Modify: `src/graph.py`
- Test: `tests/test_quality_review.py`
- Update: `tests/test_auto_verifier_routing.py`

**Interfaces:**
- Consumes: current TaskRecord and active Artifact.
- Produces: `review_record`, append-only `review_records`, and legacy `assessment` projection.
- Produces Pydantic contracts `ReviewIssue`, `QualityDimensions`, `ReviewRecord`.

- [ ] **Step 1: Write failing deterministic and LLM review tests**

```python
# tests/test_quality_review.py
from src.quality.validators import validate_artifact


def test_deterministic_review_rejects_missing_required_table_and_bad_binding():
    task = {"task_id": "T2", "generate_table": True, "task_description": "生成指标表"}
    artifact = {"task_id": "T1", "artifact_id": "A1", "text_output": "正文", "tables": [], "citations": []}
    issues = validate_artifact(task, artifact, active_artifact_id="A2")
    assert {issue.code for issue in issues} == {"ARTIFACT_TASK_MISMATCH", "STALE_ARTIFACT", "MISSING_TABLE"}
    assert all(issue.responsible_handler for issue in issues)


def test_quality_review_service_failure_does_not_become_content_failure(monkeypatch):
    from src.nodes import quality_review as module

    class FailingModel:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("review API down")

    monkeypatch.setattr(module, "get_llm", lambda *args, **kwargs: FailingModel())
    update = module.quality_review(module_test_state(), {"configurable": {"use_llm": True}})
    assert update["review_record"]["issues"][0]["category"] == "REVIEW_FAILURE"
    assert update["review_record"]["artifact_id"] == "A1"
```

- [ ] **Step 2: Run tests and verify the quality package is missing**

Run: `pytest tests/test_quality_review.py -q`

Expected: FAIL during import.

- [ ] **Step 3: Implement structured review contracts and assessment-only node**

```python
# src/quality/models.py
from typing import Literal
from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    code: str
    category: Literal[
        "CONTENT_DEFECT", "EVIDENCE_GAP", "DATA_DEFECT", "VISUAL_DEFECT",
        "WORKER_FAILURE", "LOCAL_PLAN_DEFECT", "SAFETY_BOUNDARY",
        "REQUIREMENT_MISSING", "EXTERNAL_BLOCKER", "REVIEW_FAILURE"
    ]
    severity: Literal["minor", "major", "critical", "error"]
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    responsible_handler: str
    revision_instruction: str


class QualityDimensions(BaseModel):
    completeness: int = Field(ge=0, le=5)
    evidence: int = Field(ge=0, le=5)
    logic: int = Field(ge=0, le=5)
    actionability: int = Field(ge=0, le=5)
    safety: int = Field(ge=0, le=5)


class ReviewRecord(BaseModel):
    review_id: str
    task_id: str
    artifact_id: str
    status: Literal["PASS", "REVISE", "BLOCKED", "HUMAN_REVIEW"]
    issues: list[ReviewIssue]
    quality_dimensions: QualityDimensions
    reviewer: str
    created_at: str
```

`validate_artifact()` performs binding, Worker error/status, required table/figure, citation structure, and evidence-reference checks. `quality_review()` runs deterministic validation first, invokes `get_llm(config, json_mode=True)` with a prompt that explicitly requests a JSON object, validates the response through Pydantic, merges deterministic issues, writes the ReviewRecord to Store, and returns only review state. `src/nodes/verifier.py` re-exports `quality_review as verifier` for compatibility during migration.

- [ ] **Step 4: Run quality review and existing Verifier tests**

Run: `pytest tests/test_quality_review.py tests/test_auto_verifier_routing.py tests/test_ui_projection.py -q`

Expected: PASS; existing assertions use `review_record` first and verify the compatibility `assessment` projection contains no routing command.

- [ ] **Step 5: Commit independent quality review**

```bash
git add src/quality src/nodes/quality_review.py src/nodes/verifier.py src/prompts/quality_review.md src/graph.py tests/test_quality_review.py tests/test_auto_verifier_routing.py tests/test_ui_projection.py
git commit -m "feat: add independent quality review"
```

### Task 5: Route Review Categories Deterministically and Block Retry Exhaustion

**Files:**
- Modify: `src/recovery/policy.py`
- Modify: `src/nodes/recovery.py`
- Modify: `src/nodes/verifier_manual.py`
- Modify: `src/graph.py`
- Update: `tests/test_recovery_policy.py`
- Update: `tests/test_recovery_graph.py`
- Update: `tests/test_verifier_routing.py`
- Update: `tests/test_recovery_compatibility.py`

**Interfaces:**
- Consumes: `review_record`, `task_records`, active Artifact, and existing retry counters.
- Produces: `NEXT`, `REWORK`, `EVIDENCE_RECOVERY`, `PLAN_PATCH`, `NEEDS_USER_INPUT`, or `RETRY_VERIFIER`; all successful/local-retry paths return through TaskController.

- [ ] **Step 1: Change tests to require HumanReview instead of auto-accept**

```python
def test_content_retry_exhaustion_blocks_without_committing_result():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})
    state["task_records"] = {
        "T2": {"task_id": "T2", "sequence": 0, "status": "RUNNING", "attempt_count": 3}
    }
    decision = decide_recovery_action(
        state, assessment_with("TOO_SHORT", "CONTENT_DEFECT")
    )
    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert decision["task_records"]["T2"]["status"] == "BLOCKED"
    assert "results" not in decision


def test_pass_marks_active_artifact_and_returns_to_controller():
    state = recovery_state(task_id="T2")
    state.update(active_artifact_ids={"T2": "A2"}, task_records={
        "T2": {"task_id": "T2", "sequence": 0, "status": "RUNNING", "attempt_count": 1}
    })
    state["current_result"] = {"task_id": "T2", "artifact_id": "A2", "text_output": "ok"}
    decision = decide_recovery_action(state, {"status": "PASS", "issues": []})
    assert decision["workflow_action"] == WorkflowAction.NEXT
    assert decision["task_records"]["T2"]["status"] == "PASSED"
```

- [ ] **Step 2: Run policy tests and confirm old behavior fails**

Run: `pytest tests/test_recovery_policy.py tests/test_recovery_graph.py -q`

Expected: FAIL where current code returns `ACCEPT_WITH_WARNING`, derives completion from `cursor`, or routes `NEXT` to Planner.

- [ ] **Step 3: Implement category mapping, task status updates, and manual review parity**

Remove `ACCEPT_WITH_WARNING` from `WorkflowAction`. Add `DATA_DEFECT`, `VISUAL_DEFECT`, `WORKER_FAILURE`, `LOCAL_PLAN_DEFECT`, `SAFETY_BOUNDARY`, and `REVIEW_FAILURE` classification. Use `responsible_handler` from ReviewIssue in Worker feedback. On PASS, replace-or-append the passed Artifact in compatibility `results`, mark TaskRecord PASSED, and return `NEXT`; TaskController decides whether Summarizer is next.

Use this retry-exhaustion shape:

```python
def _block_task(update, records, task_id, category, assessment):
    update.update({
        "workflow_action": WorkflowAction.NEEDS_USER_INPUT.value,
        "task_records": set_task_status(records, task_id, "BLOCKED"),
        "pending_user_action": {
            "category": category.value,
            "task_id": task_id,
            "artifact_id": assessment.get("artifact_id"),
            "issues": list(assessment.get("issues") or []),
        },
    })
    return update
```

Manual Verifier creates a `ReviewRecord(reviewer="human")`. PASS marks PASSED and returns `NEXT`; REWORK marks `REVISE_REQUIRED` and returns to TaskController; FULL_REPLAN remains the only direct route to Planner and still requires confirmation. EvidenceRecovery and validated PlanPatcher return to TaskController so a new attempt number is allocated.

- [ ] **Step 4: Run all recovery and manual-review tests**

Run: `pytest tests/test_recovery_policy.py tests/test_recovery_graph.py tests/test_verifier_routing.py tests/test_recovery_compatibility.py -q`

Expected: PASS with no assertion referencing automatic `ACCEPT_WITH_WARNING` progression.

- [ ] **Step 5: Commit deterministic quality routing**

```bash
git add src/recovery/policy.py src/nodes/recovery.py src/nodes/verifier_manual.py src/graph.py tests/test_recovery_policy.py tests/test_recovery_graph.py tests/test_verifier_routing.py tests/test_recovery_compatibility.py
git commit -m "fix: block unapproved retry exhaustion"
```

### Task 6: Prove the Nine-Task Pipeline and SQLite Resume End to End

**Files:**
- Create: `tests/test_pipeline_e2e.py`
- Modify: `src/graph.py` only if the end-to-end graph exposes an integration defect

**Interfaces:**
- Consumes: actual TaskController, ArtifactCommit, DecisionPolicy, SQLitePersistence, and graph routes from Tasks 1-5.
- Produces: regression evidence that nine tasks cannot skip and restart does not duplicate an Artifact.

- [ ] **Step 1: Build a deterministic test graph with real control nodes**

```python
# tests/test_pipeline_e2e.py
from langgraph.graph import END, START, StateGraph

from src.nodes.artifact_commit import artifact_commit
from src.nodes.task_controller import route_task_controller, task_controller
from src.recovery.policy import decide_recovery_action
from src.state import State


def worker_stub(state):
    task_id = state["current_task"]["task_id"]
    return {"current_result": {"task_id": task_id, "status": "COMPLETED", "text_output": task_id, "citations": []}}


def review_stub(state):
    artifact = state["current_result"]
    return {"assessment": {"status": "PASS", "artifact_id": artifact["artifact_id"], "issues": []}}


def policy_node(state):
    return decide_recovery_action(state, state["assessment"])


def route_policy_for_test(state):
    return "TaskController" if state["workflow_action"] == "NEXT" else "END"


def summarizer_stub(state):
    return {"final_result": {"task_ids": [item["task_id"] for item in state["results"]]}}


def build_pipeline():
    graph = StateGraph(State)
    graph.add_node("TaskController", task_controller)
    graph.add_node("Worker", worker_stub)
    graph.add_node("ArtifactCommit", artifact_commit)
    graph.add_node("Review", review_stub)
    graph.add_node("Policy", policy_node)
    graph.add_node("Summarizer", summarizer_stub)
    graph.add_edge(START, "TaskController")
    graph.add_conditional_edges(
        "TaskController",
        route_task_controller,
        {"Worker": "Worker", "Summarizer": "Summarizer", "NeedsUserInput": END},
    )
    graph.add_edge("Worker", "ArtifactCommit")
    graph.add_edge("ArtifactCommit", "Review")
    graph.add_edge("Review", "Policy")
    graph.add_edge("Policy", "TaskController")
    graph.add_edge("Summarizer", END)
    return graph
```

Complete the graph with `START -> TaskController`, controller conditional routes to Worker or END, `Worker -> ArtifactCommit -> Review -> Policy`, and `Policy -> TaskController`. Record executed task IDs in a test-only reducer or artifact list.

- [ ] **Step 2: Add nine-task, local-rework, and SQLite restart assertions**

```python
def test_nine_tasks_execute_in_order_without_skips(tmp_path):
    app = build_pipeline().compile()
    tasks = [{"task_id": f"T{i}"} for i in range(1, 10)]
    final = app.invoke({"job_id": "j1", "user_id": "u1", "tasks": tasks, "results": []})
    assert [item["task_id"] for item in final["results"]] == [f"T{i}" for i in range(1, 10)]
    assert all(record["status"] == "PASSED" for record in final["task_records"].values())


def test_sqlite_resume_reuses_committed_artifact(tmp_path):
    from src.persistence import SQLitePersistence
    from src.runtime_config import execution_config

    config = execution_config({"configurable": {"thread_id": "resume-job"}})
    initial = {
        "user_id": "u1",
        "job_id": "resume-job",
        "tasks": [{"task_id": "T1"}, {"task_id": "T2"}],
        "results": [],
    }
    with SQLitePersistence.open(tmp_path) as persistence:
        paused = build_pipeline().compile(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            interrupt_after=["ArtifactCommit"],
        )
        paused.invoke(initial, config)
        snapshot = paused.get_state(config)
        first_artifact_id = snapshot.values["active_artifact_ids"]["T1"]

    with SQLitePersistence.open(tmp_path) as persistence:
        resumed = build_pipeline().compile(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
        )
        final = resumed.invoke(None, config)

    assert final["active_artifact_ids"]["T1"] == first_artifact_id
    assert len(final["artifacts"]) == 2
    assert len({item["execution_id"] for item in final["artifacts"].values()}) == 2
    assert [item["task_id"] for item in final["results"]] == ["T1", "T2"]
```

- [ ] **Step 3: Run end-to-end tests and fix only exposed graph wiring defects**

Run: `pytest tests/test_pipeline_e2e.py -q`

Expected: PASS; T1-T9 appear exactly once in the passed result order, while a deliberately failed T3 creates a second T3 Artifact without re-running T1/T2.

- [ ] **Step 4: Run the existing graph suite**

Run: `pytest tests/test_recovery_graph.py tests/test_auto_verifier_routing.py tests/test_verifier_routing.py tests/test_runtime_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit end-to-end progression coverage**

```bash
git add tests/test_pipeline_e2e.py src/graph.py
git commit -m "test: cover multi-task pipeline recovery"
```

### Task 7: Make Summarizer Gate Inputs and Publish a Truthful ReportManifest

**Files:**
- Modify: `src/nodes/summarizer_v2.py`
- Modify: `src/state.py`
- Test: `tests/test_report_generation.py`
- Update: `tests/test_report_evidence_integration.py`

**Interfaces:**
- Consumes: passed TaskRecords and active passed Artifacts.
- Produces: `report_manifest`, compatible `final_result`, successful attachment paths only.

- [ ] **Step 1: Write failing tests for JSON mode, completion gate, and partial file failure**

```python
# tests/test_report_generation.py
from pathlib import Path

import pytest
from src.nodes import summarizer_v2 as module


def test_report_llm_calls_disable_json_mode(monkeypatch):
    calls = []
    class Model:
        def invoke(self, *args, **kwargs):
            return type("Response", (), {"content": "正文"})()
    monkeypatch.setattr(module, "get_llm", lambda config, json_mode=True: calls.append(json_mode) or Model())
    module._generate_report_evaluation("report", {})
    assert calls == [False]


def test_summarizer_rejects_incomplete_task_ledger():
    state = {"tasks": [{"task_id": "T1"}], "task_records": {"T1": {"status": "RUNNING"}}}
    with pytest.raises(RuntimeError, match="not passed"):
        module.summarizer(state, {})


def test_failed_pdf_is_not_advertised_as_attachment(monkeypatch, tmp_path):
    state = passed_report_state(tmp_path)
    monkeypatch.setattr(module.md_to_pdf, "md_to_pdf", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("wide table")))
    monkeypatch.setattr(module.md_to_docx, "md_to_docx", lambda content, path: Path(path).write_bytes(b"docx") or path)
    result = module.summarizer(state, {})
    manifest = result["report_manifest"]
    assert manifest["pdf_status"] == "FAILED"
    assert manifest["docx_status"] == "SUCCEEDED"
    assert all(not path.endswith(".pdf") for path in result["final_result"]["attachments"])
```

- [ ] **Step 2: Run tests and confirm current false-success behavior**

Run: `pytest tests/test_report_generation.py -q`

Expected: FAIL because `get_llm` uses its default JSON mode, Summarizer has no ledger gate, and attachments include failed paths.

- [ ] **Step 3: Implement gated report assembly and per-format outcomes**

Use `get_llm(config, json_mode=False)` in `_generate_section_content()` and `_generate_report_evaluation()`. Before assembly, require `all_tasks_passed(state["tasks"], ensure_task_records(state))`. Select each task's active Artifact in plan order rather than trusting append order in legacy `results`.

Use this output helper:

```python
def _run_output(name, path, writer):
    try:
        writer()
        if not os.path.isfile(path):
            raise RuntimeError(f"{name} writer returned without creating a file")
        return {"status": "SUCCEEDED", "path": path, "error": None}
    except Exception as exc:
        logger.error("%s generation failed: %s", name, exc)
        return {"status": "FAILED", "path": None, "error": str(exc)}
```

Build `report_manifest` with included Artifact IDs, independent MD/rewritten-MD/DOCX/PDF outcomes, and `generation_errors`. Build compatibility `attachments` only from `SUCCEEDED` outcomes. Persist the manifest through `WorkflowRecordStore` when Store is injected.

- [ ] **Step 4: Run report tests**

Run: `pytest tests/test_report_generation.py tests/test_report_evidence_integration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit truthful report delivery**

```bash
git add src/nodes/summarizer_v2.py src/state.py tests/test_report_generation.py tests/test_report_evidence_integration.py
git commit -m "fix: report only successful outputs"
```

### Task 8: Parse Escaped Markdown Tables and Constrain PDF Width

**Files:**
- Modify: `src/utils/md_to_pdf.py:149-682`
- Create: `tests/test_pdf_tables.py`

**Interfaces:**
- Produces: `_split_markdown_table_row(line) -> list[str]`, `_normalize_table_rows(rows)`, and width-constrained `_flush_table(story, table_buffer, font_name, table_caption=None, caption_style=None, available_width=451.0)`.
- Consumes: Markdown generated by evidence reporting, including escaped `\|` characters.

- [ ] **Step 1: Write failing escaped-pipe and wide-table tests**

```python
# tests/test_pdf_tables.py
from src.utils.md_to_pdf import _normalize_table_rows, _split_markdown_table_row, md_to_pdf


def test_escaped_pipe_stays_inside_one_cell():
    row = r"| E1 | 条件 A \| 条件 B | 工艺说明 |"
    assert _split_markdown_table_row(row) == ["E1", "条件 A | 条件 B", "工艺说明"]


def test_rows_are_padded_to_one_column_count():
    assert _normalize_table_rows([["a", "b"], ["c"]]) == [["a", "b"], ["c", ""]]


def test_wide_table_generates_pdf_without_negative_available_width(tmp_path):
    header = "|" + "|".join(f"C{i}" for i in range(20)) + "|"
    divider = "|" + "|".join("---" for _ in range(20)) + "|"
    row = "|" + "|".join("value" for _ in range(20)) + "|"
    output = tmp_path / "wide.pdf"
    md_to_pdf("\n".join([header, divider, row]), str(output))
    assert output.is_file() and output.stat().st_size > 0
```

- [ ] **Step 2: Run tests and reproduce the split/width failure**

Run: `pytest tests/test_pdf_tables.py -q`

Expected: FAIL because the helpers do not exist and the current parser uses `line.split('|')`.

- [ ] **Step 3: Implement escape-aware parsing and fixed total table width**

```python
def _split_markdown_table_row(line: str) -> list[str]:
    cells, current = [], []
    text = line.strip()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _normalize_table_rows(rows):
    width = max((len(row) for row in rows), default=0)
    return [list(row) + [""] * (width - len(row)) for row in rows]
```

Use these helpers in `parse_markdown`. In `_flush_table`, normalize rows, select `fontSize = max(6, min(10, 60 / column_count))`, and construct `Table(data, colWidths=[available_width / column_count] * column_count, repeatRows=1)`. The sum of `colWidths` must remain exactly `available_width` for every column count.

- [ ] **Step 4: Run PDF tests**

Run: `pytest tests/test_pdf_tables.py -q`

Expected: PASS and the generated PDF is non-empty.

- [ ] **Step 5: Commit PDF table hardening**

```bash
git add src/utils/md_to_pdf.py tests/test_pdf_tables.py
git commit -m "fix: constrain markdown tables in pdf"
```

### Task 9: Expose Progress, Review History, Web Authorization, and Report Status in Streamlit

**Files:**
- Modify: `src/ui_projection.py`
- Modify: `src/job_store.py`
- Modify: `src/nodes/planner.py`
- Modify: `src/task_contract.py`
- Modify: `app.py`
- Update: `tests/test_ui_projection.py`
- Update: `tests/test_planner_helpers.py`
- Create: `tests/test_job_web_authorization.py`

**Interfaces:**
- Produces: `task_progress_view(state) -> list[dict]`, `report_status_view(state) -> dict`, and `job_allows_web(state) -> bool`.
- Consumes: TaskRecords, ReviewRecords, ReportManifest, and immutable job-level `web_authorized`.

- [ ] **Step 1: Write failing UI projection and double-authorization tests**

```python
def test_task_progress_view_uses_ledger_not_cursor():
    from src.ui_projection import task_progress_view
    state = {
        "tasks": [{"task_id": "T1", "task_name": "指标"}, {"task_id": "T2", "task_name": "参数"}],
        "cursor": 0,
        "task_records": {"T1": {"status": "PASSED", "attempt_count": 1}, "T2": {"status": "RUNNING", "attempt_count": 2}},
    }
    view = task_progress_view(state)
    assert [(item["task_id"], item["status"]) for item in view] == [("T1", "PASSED"), ("T2", "RUNNING")]
    assert view[1]["attempt_count"] == 2


def test_task_web_flag_cannot_bypass_job_authorization():
    from src.task_contract import task_allows_web
    assert task_allows_web({"use_web": True}, job_authorized=False) is False
    assert task_allows_web({"use_web": True}, job_authorized=True) is True
```

- [ ] **Step 2: Run focused tests and verify projections/authorization are missing**

Run: `pytest tests/test_ui_projection.py tests/test_job_web_authorization.py tests/test_planner_helpers.py -q`

Expected: FAIL because progress/report view models and job-level gating do not exist.

- [ ] **Step 3: Implement view models and explicit job authorization**

Change `task_allows_web(task, *, job_authorized=False)` to require both levels. Update all call sites to pass `state.get("web_authorized") is True`; compatibility tests that directly test task metadata pass `job_authorized=True` explicitly.

Add to `app.py`:

```python
if "web_authorized" not in st.session_state:
    st.session_state["web_authorized"] = False

web_authorized = st.checkbox(
    "允许本报告任务检索可信公开网络资料",
    value=st.session_state["web_authorized"],
    disabled=st.session_state.get("job_record_created", False),
)
st.session_state["web_authorized"] = web_authorized
```

Add the keyword-only parameter `web_authorized: bool = False` to `JobStore.create_job` and persist it in the job record; treat it as immutable for the job. Include it in new-job state and restored session state. Planner must force `use_web`, `allow_web_fallback`, and `visualization.allow_web_fallback` to `False` when job authorization is false, regardless of model output.

Render a top progress section from `task_progress_view(_snapshot_values())`, show current Artifact/review/attempt details in an expander, and render each ReportManifest format as `SUCCEEDED`, `FAILED`, or `NOT_ATTEMPTED` with errors. Keep `_report_paths_from_state()` restricted to successful manifest paths that exist under the job root.

Change upload types to:

```python
file_type=["pdf", "docx", "csv", "xlsx", "xls"]
```

- [ ] **Step 4: Run UI, authorization, security, and persistence tests**

Run: `pytest tests/test_ui_projection.py tests/test_job_web_authorization.py tests/test_planner_helpers.py tests/test_worker_tool_boundaries.py tests/test_security.py tests/test_recovery_compatibility.py -q`

Expected: PASS; no unauthorized task or recovery path enables web access.

- [ ] **Step 5: Commit the observable work area**

```bash
git add src/ui_projection.py src/job_store.py src/nodes/planner.py src/task_contract.py app.py tests/test_ui_projection.py tests/test_job_web_authorization.py tests/test_planner_helpers.py tests/test_worker_tool_boundaries.py tests/test_security.py tests/test_recovery_compatibility.py
git commit -m "feat: expose auditable report progress"
```

### Task 10: Run Full Regression, Document Server Verification, and Push

**Files:**
- Modify: `README.md`
- Modify: `.env.example` only if the existing variable names differ from the documented commands

**Interfaces:**
- Consumes: all implementation tasks.
- Produces: a server verification checklist and a clean pushed branch.

- [ ] **Step 1: Run static diff validation and the full test suite**

Run: `git diff --check`

Expected: no output.

Run: `pytest -q`

Expected: all tests pass; there are no failures, errors, or unexpected skips.

- [ ] **Step 2: Run targeted regression tests matching the observed server failures**

Run:

```bash
pytest tests/test_pipeline_e2e.py tests/test_report_generation.py tests/test_pdf_tables.py -q
```

Expected: all pass, including T1-T9 progression, no JSON mode for report prose, escaped pipe parsing, and truthful PDF failure state.

- [ ] **Step 3: Add exact server smoke-test instructions to README**

Document:

```bash
git fetch origin
git switch codex/sqlite-checkpoint-store
git pull --ff-only
python -m pytest -q
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The manual checklist must verify: create a nine-task report; authorize or deny public web explicitly; observe T1-T9 status; force one local task revision; confirm prior tasks do not rerun; download only successful report formats; restore the job after restarting Streamlit.

- [ ] **Step 4: Commit documentation after verification**

```bash
git add README.md .env.example
git commit -m "docs: add report pipeline verification"
```

If `.env.example` did not change, stage only `README.md`.

- [ ] **Step 5: Review branch history and push**

Run: `git status --short`

Expected: empty output.

Run: `git log --oneline -12`

Expected: the plan's focused commits appear above `50186ad`.

Run: `git push origin codex/sqlite-checkpoint-store`

Expected: push succeeds and the remote branch points at the final verified commit.

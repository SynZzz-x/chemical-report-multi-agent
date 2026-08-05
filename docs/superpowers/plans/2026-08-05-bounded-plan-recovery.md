# Bounded Plan Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace automatic full replanning with deterministic quality retry, evidence recovery, local plan patches, and user-input blocking while preserving unaffected task results and cursor progress.

**Architecture:** Keep the current serial LangGraph workflow, but separate LLM assessment from deterministic routing. Add pure recovery-policy and plan-patch modules, then integrate focused recovery nodes into the graph. The automatic graph must have no route that regenerates the full plan; user-requested full replanning remains available only through manual verification and Planner Confirm.

**Tech Stack:** Python 3.13+, LangGraph 1.0, LangChain Core 1.0, pytest 8.4, TypedDict state, Streamlit interrupts.

## Global Constraints

- Verifier reports facts and issue categories; it never directly chooses automatic full replanning.
- Content defects retry at most 2 times per task, then accept with a warning.
- Evidence gaps recover at most 1 time per task, then pause for user input.
- Local plan defects patch at most 1 time per task and 3 times per Job, then pause for user input.
- Automatic execution must never replace the whole task list or reset `cursor` to 0.
- Plan patches preserve unaffected accepted results and invalidate only declared affected results.
- Missing old checkpoint fields use safe defaults; existing SQLite data does not require deletion.
- `FULL_REPLAN` is user-triggered only and must pass through Planner Confirm.
- All new routing and patch behavior is covered with deterministic tests that do not call DeepSeek, RAG, or the network.

---

### Task 1: Deterministic Recovery Policy and State Contract

**Files:**
- Create: `src/recovery/__init__.py`
- Create: `src/recovery/policy.py`
- Modify: `src/state.py`
- Modify: `app.py`
- Test: `tests/test_recovery_policy.py`

**Interfaces:**
- Produces: `WorkflowAction`, `IssueCategory`, `classify_assessment(assessment, state)`, `decide_recovery_action(state, assessment)`, and `commit_current_result(state)`.
- `decide_recovery_action` returns a dictionary containing `workflow_action`, optional `continuation_action`, updated counter dictionaries, optional `verification_warning`, and `pending_user_action`.
- Counter dictionaries use stable string `task_id` keys. Reads must accept legacy integer cursor keys.

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_recovery_policy.py` with focused states and assertions for:

```python
def test_evidence_gap_recovers_once_then_requests_user_input():
    state = recovery_state(task_id="T2")
    assessment = assessment_with("EVIDENCE_GAP", "EVIDENCE_GAP")
    first = decide_recovery_action(state, assessment)
    assert first["workflow_action"] == "EVIDENCE_RECOVERY"
    assert first["evidence_recovery_count"] == {"T2": 1}

    second_state = {**state, **first}
    second = decide_recovery_action(second_state, assessment)
    assert second["workflow_action"] == "NEEDS_USER_INPUT"
    assert second["pending_user_action"]["category"] == "EVIDENCE_GAP"


def test_missing_resource_without_available_match_is_external_blocker():
    state = recovery_state(task_id="T2", docs=[])
    decision = decide_recovery_action(
        state,
        assessment_with("MISSING_RESOURCE", None, resource_name="missing.csv"),
    )
    assert decision["workflow_action"] == "NEEDS_USER_INPUT"


def test_available_but_unassigned_resource_is_local_plan_defect():
    state = recovery_state(
        task_id="T2",
        docs=[{"name": "data.csv", "path": "/job/data.csv"}],
        task_resources=[],
    )
    decision = decide_recovery_action(
        state,
        assessment_with("MISSING_RESOURCE", None, resource_name="data.csv"),
    )
    assert decision["workflow_action"] == "PLAN_PATCH"


def test_content_retry_limit_accepts_with_warning_and_commits_result():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})
    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )
    assert decision["workflow_action"] == "ACCEPT_WITH_WARNING"
    assert decision["results"][-1]["task_id"] == "T2"
```

- [ ] **Step 2: Run policy tests and verify RED**

Run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest tests/test_recovery_policy.py -q
```

Expected: import failure because `src.recovery.policy` does not exist.

- [ ] **Step 3: Implement the policy module**

Define exact string enums:

```python
class WorkflowAction(str, Enum):
    PASS = "PASS"
    NEXT = "NEXT"
    DONE = "DONE"
    REWORK = "REWORK"
    EVIDENCE_RECOVERY = "EVIDENCE_RECOVERY"
    PLAN_PATCH = "PLAN_PATCH"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    ACCEPT_WITH_WARNING = "ACCEPT_WITH_WARNING"


class IssueCategory(str, Enum):
    CONTENT_DEFECT = "CONTENT_DEFECT"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    LOCAL_PLAN_DEFECT = "LOCAL_PLAN_DEFECT"
    EXTERNAL_BLOCKER = "EXTERNAL_BLOCKER"
```

Use deterministic code maps and the following priority when multiple issue categories exist:

```python
EXTERNAL_BLOCKER > LOCAL_PLAN_DEFECT > EVIDENCE_GAP > CONTENT_DEFECT
```

For `MISSING_RESOURCE`, match `issue.resource_name` against resource `name`, `path`, basename, `file_id`, or `resource_id` in `state.docs`. A match that is not already in the current task's `use_resources` is `LOCAL_PLAN_DEFECT`; no match is `EXTERNAL_BLOCKER`. Generic evidence-related codes must never map to a plan defect.

When status is `PASS`, commit the current result once and return `NEXT` or `DONE`. For content retry overflow, commit once, append a structured warning, return `workflow_action=ACCEPT_WITH_WARNING`, and set `continuation_action` to `NEXT` or `DONE`. The graph router uses `continuation_action` without hiding how the result was accepted.

- [ ] **Step 4: Extend state and new-job defaults**

Add these optional State fields and initialize them for new jobs in `app.py`:

```python
workflow_action: str
plan_revision: int
task_revisions: Dict[str, int]
evidence_recovery_count: Dict[str, int]
task_patch_count: Dict[str, int]
job_patch_count: int
pending_user_action: Dict[str, Any]
plan_patch_history: List[Dict[str, Any]]
verification_warnings: List[Dict[str, Any]]
```

Defaults are empty containers, `plan_revision=1`, and `job_patch_count=0`. Keep `replan_count` for checkpoint compatibility but do not use it for automatic routing.

- [ ] **Step 5: Run policy and existing state tests**

Run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest tests/test_recovery_policy.py -q
```

Expected: all policy tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/recovery/__init__.py src/recovery/policy.py src/state.py app.py tests/test_recovery_policy.py
git commit -m "feat: add deterministic recovery policy"
```

---

### Task 2: Validated Local Plan Patch Engine

**Files:**
- Create: `src/recovery/plan_patch.py`
- Test: `tests/test_plan_patch.py`

**Interfaces:**
- Consumes: State fields defined in Task 1.
- Produces: `PatchValidationError`, `validate_plan_patch(state, patch)`, and `apply_plan_patch(state, patch)`.
- `apply_plan_patch` returns only top-level LangGraph state updates and performs no LLM call.

- [ ] **Step 1: Write failing patch tests**

Cover these behaviors in `tests/test_plan_patch.py`:

```python
def test_update_patch_preserves_unaffected_results_and_cursor_progress():
    state = patch_state(cursor=2, accepted_ids=["T1", "T2"])
    patch = update_patch(task_id="T3", resume_task_id="T3")
    update = apply_plan_patch(state, patch)
    assert [item["task_id"] for item in update["results"]] == ["T1", "T2"]
    assert update["cursor"] == 2
    assert update["plan_revision"] == 2


def test_patch_rejects_stale_base_revision_without_partial_write():
    state = patch_state(plan_revision=2)
    with pytest.raises(PatchValidationError, match="base_plan_revision"):
        apply_plan_patch(state, update_patch(base_plan_revision=1))


def test_patch_rejects_unknown_resource():
    state = patch_state(docs=[])
    patch = update_patch(changes={"use_resources": ["invented.csv"]})
    with pytest.raises(PatchValidationError, match="resource"):
        validate_plan_patch(state, patch)


def test_inserted_task_gets_unique_stable_id_and_resume_position():
    state = patch_state(cursor=2)
    patch = insert_before_patch(before_task_id="T3", inserted_task_id="T2A")
    update = apply_plan_patch(state, patch)
    assert [task["task_id"] for task in update["tasks"]] == ["T1", "T2", "T2A", "T3"]
    assert update["cursor"] == 2
```

- [ ] **Step 2: Run patch tests and verify RED**

Run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest tests/test_plan_patch.py -q
```

Expected: import failure because `src.recovery.plan_patch` does not exist.

- [ ] **Step 3: Implement validation**

Accept only these operations:

```python
{"op": "update_task", "task_id": "T3", "changes": {"query": "聚乙烯气相法 工艺条件"}}
{"op": "move_before", "task_id": "T4", "before_task_id": "T3"}
{"op": "insert_before", "before_task_id": "T3", "task": {"task_id": "T2A", "task_name": "补充证据检索", "task_description": "检索并整理 T3 所需证据", "task_type": "analysis", "use_rag": true, "use_web": true, "generate_table": false, "generate_figure": false, "query": "聚乙烯反应条件", "use_resources": []}}
```

`update_task.changes` may contain only:

```python
task_name, task_description, query, use_rag, use_web,
generate_table, generate_figure, use_resources,
tool_requirements, visualization
```

Validation must reject stale revisions, duplicate or unknown task IDs, unknown resources, operations outside `affected_task_ids`, moves of accepted tasks, undeclared accepted-result invalidation, missing/invalid `resume_task_id`, empty reasons, and unsupported operations.

- [ ] **Step 4: Implement atomic application**

Apply operations to copies only after full validation. Increment `plan_revision`, increment revisions for affected tasks, filter results only for affected task IDs, set cursor to the index of `resume_task_id`, increment task/job patch counters, append a history record, and clear `pending_user_action`.

- [ ] **Step 5: Run patch tests**

Run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest tests/test_plan_patch.py tests/test_recovery_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/recovery/plan_patch.py tests/test_plan_patch.py
git commit -m "feat: validate and apply local plan patches"
```

---

### Task 3: Recovery Nodes, Worker Feedback, and Graph Routing

**Files:**
- Create: `src/nodes/recovery.py`
- Create: `src/prompts/planner_patch.md`
- Modify: `src/nodes/verifier.py`
- Modify: `src/nodes/worker/agent/graph.py`
- Modify: `src/graph.py`
- Modify: `src/prompts/verifier.md`
- Modify: `tests/test_auto_verifier_routing.py`
- Create: `tests/test_recovery_graph.py`

**Interfaces:**
- Consumes: `decide_recovery_action`, `apply_plan_patch`, State recovery fields.
- Produces node functions `decision_policy`, `evidence_recovery`, `plan_patcher`, `needs_user_input`, plus routing functions `route_policy` and `route_after_blocker`.
- Worker consumes `worker_state.execution_feedback` containing `mode`, `issues`, `instructions`, and optional `recovery_query`/`allow_web`.

- [ ] **Step 1: Replace obsolete automatic-verifier tests with failing assessment-only tests**

Update `tests/test_auto_verifier_routing.py` so it asserts:

```python
update = verifier(state, config)
assert "decision" not in update
assert update["assessment"]["issues"][0]["category"] == "EVIDENCE_GAP"
```

Add `tests/test_recovery_graph.py` cases asserting:

```python
def test_t2_evidence_gap_never_routes_to_planner_or_resets_cursor():
    state = graph_state(cursor=1, accepted_ids=["T1"])
    update = decision_policy({**state, "assessment": evidence_gap_assessment()}, {})
    assert update["workflow_action"] == "EVIDENCE_RECOVERY"
    assert update.get("cursor", state["cursor"]) == 1
    assert [item["task_id"] for item in state["results"]] == ["T1"]


def test_auto_graph_has_no_replan_route_to_planner():
    source = inspect.getsource(WorkFlowBase._build)
    assert '"REPLAN": "Planner"' not in source
```

- [ ] **Step 2: Run graph tests and verify RED**

Run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest tests/test_auto_verifier_routing.py tests/test_recovery_graph.py -q
```

Expected: failures because Verifier still returns routing decisions and recovery nodes do not exist.

- [ ] **Step 3: Make Verifier assessment-only**

Keep LLM invocation, JSON sanitation, and logging. Require issue fields `code`, `category`, `description`, `suggestion`, optional `resource_name`, and `severity`. Remove retry/replan counters, result mutation, control-message generation, and `decision` from Verifier output. Log the sanitized assessment and current plan revision.

Update `src/prompts/verifier.md` so evidence insufficiency is `EVIDENCE_GAP`, an existing-but-unassigned resource is `LOCAL_PLAN_DEFECT`, an absent external dependency is `EXTERNAL_BLOCKER`, and output contains no route recommendation.

- [ ] **Step 4: Implement recovery nodes**

`decision_policy` calls the pure policy and creates `worker_state.execution_feedback` for `REWORK`. `evidence_recovery` creates an evidence-specific feedback payload and a recovery query from missing requirements; it enables web use only when the task has `use_web`, `allow_web_fallback`, or `visualization.allow_web_fallback` enabled.

`plan_patcher` calls the LLM with the exact JSON schema from Task 2, validates/applies the patch, and on any parse, model, or validation failure returns `workflow_action=NEEDS_USER_INPUT` with a concrete blocker payload. It must never fall back to whole-plan generation.

`needs_user_input` calls `interrupt` with `type=needs_user_input`, guidance, category, affected task, issues, and accepted choices. On resume it merges uploaded docs and returns `REWORK` for content/external resources or `EVIDENCE_RECOVERY` for evidence gaps, without changing cursor.

- [ ] **Step 5: Feed recovery instructions into Worker**

In `AutonomousToolNode.process`, copy the current task to an execution-only task, read `worker_state.execution_feedback`, append its instructions to the Worker human prompt, optionally use its recovery query, and enable the spider tool only when `allow_web` is true. Remove the consumed feedback from returned `worker_state` so it cannot leak into later tasks.

- [ ] **Step 6: Rewire automatic graph**

Build the automatic path as:

```text
Worker -> Verifier -> DecisionPolicy
DecisionPolicy NEXT -> Planner
DecisionPolicy DONE -> Summarizer
DecisionPolicy REWORK -> Worker
DecisionPolicy EVIDENCE_RECOVERY -> EvidenceRecovery -> Worker
DecisionPolicy PLAN_PATCH -> PlanPatcher
DecisionPolicy NEEDS_USER_INPUT -> NeedsUserInput
DecisionPolicy ACCEPT_WITH_WARNING -> continuation_action(NEXT or DONE)
PlanPatcher REWORK -> Worker
PlanPatcher NEEDS_USER_INPUT -> NeedsUserInput
NeedsUserInput REWORK -> Worker
NeedsUserInput EVIDENCE_RECOVERY -> EvidenceRecovery
NeedsUserInput NEXT -> Planner
NeedsUserInput DONE -> Summarizer
```

The manual workflow continues to use the manual verifier path. Remove the automatic `REPLAN -> Planner` edge entirely.

- [ ] **Step 7: Run recovery integration tests**

Run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest tests/test_auto_verifier_routing.py tests/test_recovery_graph.py tests/test_recovery_policy.py tests/test_plan_patch.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/nodes/recovery.py src/prompts/planner_patch.md src/nodes/verifier.py src/nodes/worker/agent/graph.py src/graph.py src/prompts/verifier.md tests/test_auto_verifier_routing.py tests/test_recovery_graph.py
git commit -m "feat: route automatic verification through bounded recovery"
```

---

### Task 4: User-Triggered Full Replan, Compatibility, and Final Regression

**Files:**
- Modify: `src/nodes/verifier_manual.py`
- Modify: `src/nodes/planner.py`
- Modify: `src/graph.py`
- Modify: `app.py`
- Modify: `tests/test_verifier_routing.py`
- Create: `tests/test_recovery_compatibility.py`

**Interfaces:**
- Manual verifier emits `FULL_REPLAN`, never automatic `REPLAN`.
- Planner recognizes only a user-originated `FULL_REPLAN` message for full task regeneration and routes it through `Planner_Confirm`.
- Old checkpoints missing recovery fields remain executable through policy defaults.

- [ ] **Step 1: Write failing manual/full-replan and compatibility tests**

Update the manual replan test to assert:

```python
assert verifier_update["decision"] == "FULL_REPLAN"
assert json.loads(verifier_update["messages"][-1].content)["type"] == "FULL_REPLAN"
```

Add tests that a full replan increments `plan_revision`, clears obsolete results only after the user confirms the replacement plan, and starts the confirmed new plan at cursor 0. Add a legacy-state test with no recovery fields proving policy defaults do not raise and do not choose full replan.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest tests/test_verifier_routing.py tests/test_recovery_compatibility.py -q
```

Expected: failures because manual verification still emits `REPLAN` and Planner does not recognize `FULL_REPLAN`.

- [ ] **Step 3: Implement manual full replan**

Normalize user requests for overall goal/outline changes to `FULL_REPLAN`. Route that decision from the manual verifier to Planner. Planner may call the existing whole-plan builder only for `FULL_REPLAN`, set `planner_action=FULL_REPLAN`, increment plan revision, and require `Planner_Confirm`. On confirmation, clear obsolete accepted/current results, reset counters for the new plan, initialize task revisions, and start at cursor 0.

Remove automatic use of `_build_tasks_from_replan_feedback`; retain it only as the implementation behind the explicitly user-triggered full-replan path, or rename it to make that restriction clear.

- [ ] **Step 4: Improve UI/control-message compatibility**

Treat `FULL_REPLAN`, `PLAN_PATCH`, `EVIDENCE_RECOVERY`, and `NEEDS_USER_INPUT` as internal control messages in `app.py`. Show generic `needs_user_input` interrupt guidance in the chat and persist it through the existing pending-interrupt mechanism. Do not require SQLite migration or deletion.

- [ ] **Step 5: Run complete verification**

Temporarily move the local `.venv` out of the repository so the security test does not scan third-party packages, then run:

```bash
/private/tmp/agent-master-test-venv/bin/python -m pytest -q
/private/tmp/agent-master-test-venv/bin/python -m compileall -q src app.py
git diff --check
```

Expected: zero failures and zero syntax/diff errors. Restore `.venv` immediately after tests.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/nodes/verifier_manual.py src/nodes/planner.py src/graph.py app.py tests/test_verifier_routing.py tests/test_recovery_compatibility.py
git commit -m "fix: reserve full replanning for user requests"
```

- [ ] **Step 7: Final review and push**

Run a whole-branch review against commit `ecf5b03`, fix all Critical and Important findings, rerun the complete verification commands, then push:

```bash
git push origin codex/sqlite-checkpoint-store
```

Confirm local `HEAD` equals `refs/remotes/origin/codex/sqlite-checkpoint-store` before reporting completion.

# Failure Semantics / Recovery vNext Semantic Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four remaining failure-semantics authority gaps with a minimal, backward-compatible diff.

**Architecture:** Preserve the current graph and persistence model. Make canonical blocker records, precise verifier linkage, canonical failure decisions, and checkpoint outcome projection the only authorities, while retaining legacy fields solely as derived compatibility data.

**Tech Stack:** Python 3.14, Pydantic v2, LangGraph, Streamlit, pytest, SQLite checkpoint/store.

## Global Constraints

- Do not add graph nodes or change SQLite saver/store layering.
- Do not rewrite persistence, RAG, rendering, scheduling, or the worker tool loop.
- Preserve old checkpoints by using additive defaults and one-way projections.
- Write each regression test before its production change and observe the expected failure.
- Do not modify or delete unrelated local files.

---

### Task 1: Canonical blocker actions

**Files:**
- Modify: `src/recovery/policy.py`
- Modify: `src/nodes/recovery.py`
- Modify: `src/graph.py`
- Test: `tests/test_failure_decision_policy.py`
- Test: `tests/test_recovery_graph.py`
- Test: `tests/test_blocker_registry.py`

**Interfaces:**
- Consumes: `UserBlockerRecord.available_options`, `apply_blocker_resolution()`.
- Produces: one-way `pending_user_action` projection and blocker-keyed resume validation.

- [x] Add a failing hard-evidence regression asserting single-blocker legacy choices equal or narrow canonical options and exclude acceptance/NEXT/DONE.
- [x] Add a failing forged `ACCEPT_EVIDENCE_GAP` resume test asserting rejection and unchanged blocker state.
- [x] Run the individual tests and confirm failures expose the legacy single-blocker path.
- [x] Make `_register_user_blocker()` always derive compatibility projection from active canonical blockers.
- [x] Make `needs_user_input()` use canonical blocker handling whenever a canonical blocker exists, including exactly one blocker.
- [x] Validate canonical actions before upload ingestion or task mutation side effects.
- [x] Run blocker and recovery targeted tests until green.

### Task 2: Authoritative runner outcome

**Files:**
- Create: `src/job_outcome.py`
- Modify: `app.py`
- Modify: `run.py`
- Test: `tests/test_job_store.py`
- Test: `tests/test_fatal_system.py`

**Interfaces:**
- Produces: `derive_job_status(state, pending_interrupt)` returning `waiting`, `failed`, or `completed`.
- Consumes: checkpoint `failure_decision`, `workflow_action`, and `fatal_system_error`.

- [x] Add failing tests for fatal normal return, DONE, pending blocker, and runner exception precedence.
- [x] Add source/integration assertions that both app and CLI use the shared helper for restore and completion.
- [x] Run the individual tests and confirm fatal normal return is incorrectly completed before implementation.
- [x] Implement the helper with waiting-before-fatal precedence and enum-backed comparisons.
- [x] Replace Streamlit and CLI duplicated completion inference with the helper.
- [x] Persist only bounded canonical fatal diagnostics already present in checkpoint state.
- [x] Explicitly clear stale fatal diagnostics when a later authoritative projection is nonfatal.
- [x] Run JobStore/fatal tests until green.

### Task 3: Exact issue requirement linkage

**Files:**
- Modify: `src/verifier_contract.py`
- Modify: `src/nodes/verifier.py`
- Modify: `src/prompts/verifier.md`
- Modify: `src/recovery/policy.py`
- Test: `tests/test_auto_verifier_routing.py`
- Test: `tests/test_failure_decision_policy.py`

**Interfaces:**
- Produces: `AssessmentIssue.requirement_ids: list[str]` with default `[]`.
- Consumes: active `requirement_registry` and task `requirement_ids`.

- [x] Add failing serialization and legacy-default tests.
- [x] Add failing deterministic linkage and unknown LLM-ID filtering tests.
- [x] Add failing mixed hard-resource/soft-length policy regression and legacy empty-linkage regression.
- [x] Run each test and confirm the expected linkage/fallback failures.
- [x] Add the schema field and LLM output contract.
- [x] Sanitize LLM IDs against the active legal scope and attach deterministic IDs by requirement kind.
- [x] Change `_requirement_scope()` so empty linkage remains empty.
- [x] Preserve report-wide active linkage for synthesis tasks, including legacy tasks with no local IDs.
- [x] Run verifier and decision-policy tests until green.

### Task 4: PlanPatcher failure closure

**Files:**
- Modify: `src/recovery/policy.py`
- Modify: `src/nodes/recovery.py`
- Test: `tests/test_failure_decision_policy.py`
- Test: `tests/test_recovery_graph.py`

**Interfaces:**
- Consumes: `decide_recovery_action(state, assessment)`.
- Produces: canonical `FailureDecision` for patch budget exhaustion and internal patch failure.

- [x] Replace the existing legacy expectation with failing tests for soft exhaustion, internal exception, and hard conflict.
- [x] Confirm failures show `PLAN_PATCH_ERROR` or noncanonical user input.
- [x] Add explicit policy handling for `PLAN_PATCH_INTERNAL_ERROR` and exhausted local-patch budgets.
- [x] Make `_patch_error_update()` sanitize exception metadata and call policy rather than creating a legacy action.
- [x] Replace the obsolete PlanPatcher `NEEDS_USER_INPUT` mapping with canonical `FATAL_SYSTEM -> Exit` routing.
- [x] Assert every patch failure has a valid closed-matrix `FailureDecision` and no legacy business blocker.
- [x] Run PlanPatcher and policy tests until green.

### Task 5: Full verification and semantic audit

**Files:**
- Verify all changed files; do not add unrelated changes.

**Interfaces:**
- Consumes: all preceding behavior.
- Produces: evidence for final report.

- [x] Run the combined targeted suite and record exact pass counts.
- [x] Run complete `pytest -q` and record exact pass/fail counts.
- [x] Run `python -m compileall` over `src`, `app.py`, and `run.py`.
- [x] Search for direct business `pending_user_action` construction outside canonical compatibility/legacy migration paths.
- [x] Search both runners for paths that can mark canonical fatal checkpoint state completed.
- [x] Run `git diff --check`, inspect the final diff, and record branch, HEAD, and status.

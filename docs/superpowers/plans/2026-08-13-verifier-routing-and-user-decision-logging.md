# Verifier Routing and User Decision Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route actionable deterministic/content issues ahead of Verifier contract failures and log explicit blocker-resolution decisions.

**Architecture:** Keep all issues for audit, but change `classify_assessment` to exclude `ASSESSMENT_CONTRACT_ERROR` from category competition whenever another issue exists; real LLM runtime/service failures keep their existing priority. Add structured INFO logs at the system policy boundary and at `needs_user_input` after the user choice has been normalized to its final workflow action.

**Tech Stack:** Python, LangGraph, pytest, standard-library logging.

## Global Constraints

- Do not modify Planner task granularity, `covers_sections`, or Task Contract.
- Do not change retry limits or explicit user-acceptance behavior.
- Do not log free-form user text or uploaded file paths.

---

### Task 1: Assessment category arbitration

**Files:**
- Modify: `src/recovery/policy.py`
- Test: `tests/test_recovery_policy.py`

**Interfaces:**
- Consumes: normalized assessment issue dictionaries and workflow state.
- Produces: `classify_assessment(assessment, state) -> IssueCategory` and unchanged `decide_recovery_action` output shape.

- [ ] Add a failing test where `ASSESSMENT_CONTRACT_ERROR` and `TOO_LONG` coexist and assert `CONTENT_DEFECT`, `REWORK`, and no Verifier retry increment.
- [ ] Run the focused test and confirm it fails because the current category is `VERIFIER_FAILURE`.
- [ ] Change category selection to ignore `ASSESSMENT_CONTRACT_ERROR` whenever at least one other issue exists, without changing runtime/service failure priority.
- [ ] Run the focused policy tests and confirm mixed issues route to Worker while contract-error-only assessments retain their current behavior.

### Task 2: User blocker-decision logging

**Files:**
- Modify: `src/nodes/recovery.py`
- Test: `tests/test_recovery_graph.py`

**Interfaces:**
- Consumes: normalized blocker choice and final `WorkflowAction` inside `needs_user_input`.
- Produces: one INFO log containing task, category, choice, action, and upload presence.

- [ ] Add a failing `caplog` test for `ACCEPT_AS_DRAFT` that checks the normalized decision fields.
- [ ] Add a failing `caplog` test that identifies automatic policy routing with `source=system`.
- [ ] Run the focused test and confirm no decision log currently exists.
- [ ] Add a module logger and emit the structured decision message after final action resolution.
- [ ] Run focused recovery graph tests and confirm explicit acceptance still advances and records `USER_ACCEPTED_WARNING`.

### Task 3: Regression verification and delivery

**Files:**
- Verify: `tests/test_recovery_policy.py`
- Verify: `tests/test_auto_verifier_routing.py`
- Verify: `tests/test_recovery_graph.py`

**Interfaces:**
- Consumes: completed implementation from Tasks 1 and 2.
- Produces: verified branch commit.

- [ ] Run the three related test modules.
- [ ] Run `git diff --check`.
- [ ] Review the diff for unrelated Planner changes.
- [ ] Commit and push `codex/sqlite-checkpoint-store`.

# Scoped Evidence Gap Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent evidence-gap acceptance from silently accepting unrelated section defects.

**Architecture:** Persist an evidence waiver scoped to a task revision, filter only waivable availability gaps during verification, and re-run existing recovery policy on all remaining issues. Preserve `USER_ACCEPTED_GAP` when the section later satisfies every non-waived check.

**Tech Stack:** Python, LangGraph state, pytest.

## Global Constraints

- Never waive citation integrity, unsupported-source, content, length, or asset issues.
- Keep existing retry limits and Planner contract unchanged.
- Keep accepted-gap sections out of `READY_FOR_FINAL`.

---

### Task 1: Lock mixed-issue resume behavior

**Files:**
- Modify: `tests/test_recovery_graph.py`
- Modify: `tests/test_recovery_policy.py`

**Interfaces:**
- Consumes: `needs_user_input`, `decide_recovery_action`.
- Produces: regression expectations for scoped acceptance and report status.

- [ ] Add a failing test for `EVIDENCE_GAP + TOO_LONG` followed by `ACCEPT_EVIDENCE_GAP`; require `REWORK`, no result commit, and a persisted waiver.
- [ ] Add a failing test proving a later PASS with a matching waiver records `USER_ACCEPTED_GAP`.
- [ ] Run both tests and confirm current behavior fails by returning `NEXT` or `VERIFIED_PASS`.

### Task 2: Persist and consume scoped waivers

**Files:**
- Modify: `src/state.py`
- Modify: `app.py`
- Modify: `src/recovery/policy.py`
- Modify: `src/nodes/recovery.py`
- Modify: `src/nodes/verifier.py`

**Interfaces:**
- Consumes: current task ID, plan revision, task revision, normalized issues.
- Produces: `accepted_evidence_gaps` state and filtered assessment routing.

- [ ] Add helpers for waiver scope matching and waivable evidence codes.
- [ ] On acceptance, persist only accepted availability-gap issues and route remaining issues through `decide_recovery_action`.
- [ ] Filter matching waived gaps after Verifier normalization and deterministic checks.
- [ ] On later PASS, preserve user-accepted-gap section status and audit issues.
- [ ] Run focused tests until green.

### Task 3: Regression verification and delivery

**Files:**
- Verify: `tests/test_recovery_graph.py`
- Verify: `tests/test_recovery_policy.py`
- Verify: `tests/test_auto_verifier_routing.py`

**Interfaces:**
- Consumes: completed scoped-waiver implementation.
- Produces: tested commit on `codex/sqlite-checkpoint-store`.

- [ ] Run related recovery and verifier tests.
- [ ] Run the complete pytest suite and `git diff --check`.
- [ ] Commit and push the current branch.

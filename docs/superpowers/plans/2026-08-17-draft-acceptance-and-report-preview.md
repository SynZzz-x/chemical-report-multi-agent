# Draft Acceptance and Report Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent citation-integrity defects from being accepted as draft, clarify local evidence waivers, hide Worker protocol messages, and render the final report from its safe Markdown artifact.

**Architecture:** Recovery policy remains the sole producer of allowed blocker actions, while the recovery node only validates and executes those actions. UI chat projection filters internal Worker protocol messages, and final report preview reads the authoritative Markdown artifact already admitted by the report-delivery gate.

**Tech Stack:** Python 3, LangGraph state, Streamlit, pytest.

## Global Constraints

- Do not change length, heading, or RAG behavior.
- Do not duplicate the full report body in checkpoint state.
- Citation-integrity evidence issues must disable `ACCEPT_AS_DRAFT` in every blocker category.
- Restored or incomplete blocker contracts must fail closed and must not synthesize acceptance actions.
- Production changes follow failing regression tests.

---

### Task 1: Draft-acceptance policy contract

**Files:**
- Modify: `tests/test_recovery_policy.py`
- Modify: `src/recovery/policy.py`
- Modify: `src/control_messages.py`

**Interfaces:**
- Consumes: `classify_issue(issue, state)` and `is_waivable_evidence_gap(issue)`.
- Produces: `_allows_accept_as_draft(state, issues) -> bool` and blocker `accepted_choices` guarded by that result.

- [ ] Add failing policy tests for the two integrity-blocked combinations and the positive waivable combination.
- [ ] Run the focused tests and confirm current unconditional blocker choices fail the new assertions.
- [ ] Widen read-only classification inputs to `Mapping`, add `_allows_accept_as_draft()`, and route every `ACCEPT_AS_DRAFT` insertion through it.
- [ ] Update `ACCEPT_EVIDENCE_GAP` label, button text, default text, and blocker guidance to describe a local waiver.
- [ ] Run the focused policy and recovery-node tests until green.
- [ ] Add restored-checkpoint tests for stale draft choices and missing blocker choices, then normalize both through the policy contract.

### Task 2: Internal protocol filtering and artifact preview

**Files:**
- Modify: `tests/test_recovery_compatibility.py`
- Create: `tests/test_report_preview.py`
- Modify: `src/control_messages.py`
- Create: `src/report_preview.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: `_report_paths_from_state() -> list[Path]`.
- Produces: `first_markdown_path(paths) -> Path | None`, `read_markdown_preview(path) -> str | None`, `_final_markdown_path() -> Path | None`, and `_render_report_preview() -> None`.

- [ ] Add failing tests proving `TASK_RESULT` and `FINAL_RESULT` are not displayable assistant content.
- [ ] Add failing tests for selecting/rendering the admitted Markdown artifact and tolerating no Markdown artifact.
- [ ] Run focused tests and confirm failures are caused by missing filtering/preview behavior.
- [ ] Add both protocol types to `INTERNAL_CONTROL_TYPES`.
- [ ] Implement Markdown selection and preview rendering, then call the preview between history and downloads.
- [ ] Treat top-level `BLOCKED` as authoritative and re-filter restored text history.
- [ ] Run focused tests until green.

### Task 3: Regression verification

**Files:**
- Verify all modified source, test, and documentation files.

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: a clean, regression-tested change set.

- [ ] Run recovery-policy, recovery-graph, control-message, and report-delivery tests.
- [ ] Run the full pytest suite.
- [ ] Run `git diff --check` and inspect the final diff for scope violations.

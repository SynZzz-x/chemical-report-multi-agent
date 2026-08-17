# Final Admission for User-Accepted Asset Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow final draft assembly to omit only those missing figure/table assets that the user explicitly accepted, without weakening integrity checks.

**Architecture:** Keep report-level acceptance in `report_acceptance.py`. Add a narrow issue-family matcher in `summarizer_v2.py`, use it only for absent requested assets, and preserve all existing blockers for missing results, revision mismatches, invalid assets, and missing files.

**Tech Stack:** Python, LangGraph state dictionaries, pytest.

## Global Constraints

- `ACCEPT_AS_DRAFT` must remain commit-only and must not call Worker or Verifier again.
- Only user-accepted missing-asset issues may be degraded.
- Newly discovered and integrity-related defects remain blocking.

---

### Task 1: Lock the admission behavior with tests

**Files:**
- Modify: `tests/test_summarizer_deterministic.py`

**Interfaces:**
- Consumes: `summarizer_v2.summarizer(state, config)`
- Produces: regression coverage for accepted and unaccepted asset absence

- [ ] Add a test where `USER_ACCEPTED_WARNING` contains `MISSING_FIGURE`, the task requests a figure, and the result has none; assert `DRAFT_WITH_GAPS` and no blocking result.
- [ ] Add the equivalent `MISSING_TABLE` test.
- [ ] Add a negative test where the accepted issue is `TOO_LONG` but the task requests a missing figure; assert `BLOCKED` with `MISSING_FIGURE_ASSET`.
- [ ] Run the new tests and confirm they fail because current `_ordered_sections()` blocks every missing requested asset.

### Task 2: Implement narrow accepted-defect coverage

**Files:**
- Modify: `src/nodes/summarizer_v2.py`

**Interfaces:**
- Consumes: one section status entry and one final asset issue code
- Produces: a boolean indicating whether that exact absence family was explicitly accepted by the user

- [ ] Add normalized figure/table missing-code families and a helper that requires `USER_ACCEPTED_WARNING`, `accepted_by=user`, and a matching accepted issue.
- [ ] In `_ordered_sections()`, degrade only `MISSING_FIGURE_ASSET` and `MISSING_TABLE_ASSET` when covered; keep invalid assets, missing files, corrupt images, revision mismatches, and missing results blocking.
- [ ] Log derived admission state, accepted degradations, and blocking sections.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Verify and deliver

**Files:**
- Verify: `src/nodes/summarizer_v2.py`
- Verify: `tests/test_summarizer_deterministic.py`

**Interfaces:**
- Consumes: repository test suite
- Produces: pushed commit on `codex/sqlite-checkpoint-store`

- [ ] Run `pytest tests/test_summarizer_deterministic.py -q`.
- [ ] Run the full pytest suite.
- [ ] Run `git diff --check` and inspect the final diff.
- [ ] Commit the implementation and push `codex/sqlite-checkpoint-store`.


# Workflow Acceptance and Evidence Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make terminal draft acceptance, cross-task citations, derived references, synthesis verification, and report rendering deterministic and internally consistent.

**Architecture:** Keep `section_status` and `report_status` as the only acceptance truth. Add task-scoped evidence keys at aggregation boundaries, freeze global display IDs only during deterministic report assembly, and derive reference output from admitted citations. Harden existing renderers without introducing GenericWorker, RunContext, EvidenceStore, or a Report AST.

**Tech Stack:** Python 3.13, LangGraph state updates, Pydantic evidence models, pytest, ReportLab, python-docx.

## Global Constraints

- Production changes require a failing regression test first.
- Cross-task evidence logic must never key by a bare local `E1`.
- Final display numbering follows admitted task order and first appearance order.
- `section_status` and `report_status` remain authoritative.
- GenericWorker, RunContext, persistent EvidenceStore, Report AST, and landscape layout are out of scope.

---

### Task 1: Terminal draft acceptance

**Files:**
- Modify: `src/nodes/recovery.py`
- Test: `tests/test_recovery_graph.py`

**Interfaces:**
- Consumes: `_continuation_action(state)`, `commit_current_result(state)`, and `record_section_status(state, status, accepted_by, issues)`.
- Produces: one accepted-state transition whose `workflow_action` is `DONE` for the terminal task and `NEXT` otherwise.

- [ ] Add a failing terminal-task regression test that resumes with `ACCEPT_AS_DRAFT` and asserts committed results, `USER_ACCEPTED_WARNING`, `DRAFT_WITH_GAPS`, and `DONE`.
- [ ] Run `pytest tests/test_recovery_graph.py -k "accept_content_warning_as_draft" -q` and confirm the terminal case fails because it returns `NEXT`.
- [ ] Replace the hard-coded special-choice route with `_continuation_action(state).value` while retaining the existing commit and status update.
- [ ] Re-run the focused tests and commit the behavior.

### Task 2: Task-scoped evidence identity and final display mapping

**Files:**
- Create: `src/evidence/identity.py`
- Modify: `src/nodes/synthesis.py`
- Modify: `src/nodes/summarizer_v2.py`
- Test: `tests/test_report_evidence_integration.py`
- Test: `tests/test_synthesis.py`
- Test: `tests/test_summarizer_deterministic.py`

**Interfaces:**
- Produces: `evidence_key(task_id: str, local_id: str) -> str`, `build_display_evidence_map(sections) -> dict[str, str]`, and section-scoped citation rewriting.
- Consumes: admitted sections with `task_id`, text, citations, tables, and figures.

- [ ] Add failing tests proving `T2:E1` and `T3:E1` remain distinct in synthesis context and receive deterministic global display IDs.
- [ ] Add a failing report test proving body markers, citation records, and figure evidence IDs use the same frozen global ID.
- [ ] Run the focused tests and verify failures are caused by bare-ID deduplication.
- [ ] Implement task-scoped keys and deterministic first-appearance mapping without mutating persisted Worker results.
- [ ] Re-run focused tests and commit.

### Task 3: Derived reference projection

**Files:**
- Modify: `src/report_outline.py`
- Modify: `src/nodes/summarizer_v2.py`
- Test: `tests/test_planner_contract.py`
- Test: `tests/test_summarizer_deterministic.py`

**Interfaces:**
- Consumes: outline system-generated section classification and globally normalized admitted citations.
- Produces: deterministic reference content with source, locator, supported section, and concise summary.

- [ ] Add failing classification tests for “知识库依据与参考文件说明” and equivalent source-list titles.
- [ ] Add a failing summarizer test proving the requested reference section is projected from citations and does not contain a Worker-authored empty-source claim.
- [ ] Run focused tests and confirm the titles are currently classified as content.
- [ ] Extend system-section classification and insert the deterministic reference projection at its outline position, with an end-of-report fallback for legacy outlines.
- [ ] Re-run focused tests and commit.

### Task 4: Synthesis verification provenance

**Files:**
- Modify: `src/nodes/synthesis.py`
- Modify: `src/nodes/verifier.py`
- Test: `tests/test_synthesis.py`
- Test: `tests/test_auto_verifier_routing.py`

**Interfaces:**
- Consumes: accepted sections, task-scoped evidence IDs, candidate text, and deterministic synthesis audit.
- Produces: verifier context that distinguishes accepted extractive claims from unsupported additions.

- [ ] Add failing tests showing an accepted sentence with task-scoped evidence is not rejected as a new synthesis fact.
- [ ] Run focused tests and confirm verifier context currently omits accepted-section lineage.
- [ ] Attach deterministic synthesis context/audit to the verifier prompt and sanitizer path; do not trust model-declared provenance as proof.
- [ ] Re-run focused tests and commit.

### Task 5: PDF and DOCX renderer hardening

**Files:**
- Modify: `src/utils/md_to_pdf.py`
- Modify: `src/utils/md_to_docx.py`
- Modify: `src/nodes/summarizer_v2.py`
- Test: `tests/test_summarizer_deterministic.py`
- Create: `tests/test_report_renderers.py`

**Interfaces:**
- Produces: bounded four-column evidence tables and heading numbering that never starts from zero.

- [ ] Add a failing PDF test rendering a long 34-row evidence table and a failing DOCX test whose Markdown begins with a level-three heading.
- [ ] Run the new renderer tests and confirm the PDF width error and `0.1` numbering.
- [ ] Add explicit four-column widths, repeated header rows, smaller padding, and presentation-summary truncation.
- [ ] Make report assembly emit valid parents and add a defensive DOCX parent counter for malformed Markdown.
- [ ] Re-run renderer tests and commit.

### Task 6: Length rewrite margin and final verification

**Files:**
- Modify: `src/nodes/worker/agent/graph.py`
- Test: `tests/test_worker_tool_boundaries.py`

**Interfaces:**
- Produces: a length-only rewrite instruction targeting 92% of the configured maximum while remaining tool-free.

- [ ] Add a failing test asserting a 2500-character maximum produces a 2300-character rewrite target.
- [ ] Run the focused test and confirm the current instruction targets the hard maximum.
- [ ] Implement the centralized 0.92 safety margin and preserve existing minimum-length behavior.
- [ ] Run focused tests, then run `pytest -q` and `git diff --check`.
- [ ] Commit all remaining changes and push `codex/sqlite-checkpoint-store`.

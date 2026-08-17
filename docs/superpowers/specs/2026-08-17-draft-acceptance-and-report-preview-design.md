# Draft Acceptance and Report Preview Design

## Goal

Close the first-stage UI and state-contract gaps without changing length handling, heading assembly, or RAG behavior.

## Safety contract

`src/recovery/policy.py` is the only layer that decides which recovery actions are allowed. A shared `_allows_accept_as_draft()` helper examines the complete assessment issue set. Draft acceptance remains available for ordinary content defects and explicitly waivable evidence-availability gaps, but is unavailable whenever any evidence-category issue is not accepted by `is_waivable_evidence_gap()`.

This rule applies to every blocker branch that may expose `ACCEPT_AS_DRAFT`, including content, evidence, and verifier-failure blockers. `src/nodes/recovery.py` continues to enforce membership in the blocker-provided `accepted_choices`; it does not duplicate the safety policy.

Restored SQLite checkpoints pass their persisted blocker choices through an idempotent policy sanitizer before display or execution. The sanitizer combines pending blocker issues with the current assessment, removes unsafe acceptance actions, and fails closed when an old blocker has no explicit `accepted_choices`; missing contracts never synthesize `NEXT` or draft acceptance.

`ACCEPT_EVIDENCE_GAP` remains a local waiver. Its UI copy must state that only evidence-availability gaps are accepted and that remaining length, format, content, and citation-integrity defects will continue through recovery.

## UI projection and report preview

Worker protocol messages with `type=TASK_RESULT` or `type=FINAL_RESULT` are internal and must not enter user chat history.

The final report preview is derived from the authoritative Markdown artifact selected through `_report_paths_from_state()`. That helper already rejects blocked/failed deliveries, stale authoritative fallbacks, paths outside the job root, unsupported suffixes, and missing files. The preview reads the first selected Markdown artifact, handles read failures without breaking the page, and renders before the existing download controls. The full report body is not duplicated into the SQLite checkpoint.

The top-level workflow `report_status=BLOCKED` is authoritative even when `final_result` is empty, so a replanned job cannot display a previous completed report. Restored text history is re-filtered at render time, while structured plan and verification projections remain visible.

## Type boundary

New read-only helpers accept `Mapping[str, Any]` and `Iterable[Mapping[str, Any]]`. Existing classification helpers used by the new API are widened to read-only mappings where their implementations do not mutate inputs.

## Verification

Permanent recovery-policy regressions cover:

- `TOO_LONG + INVALID_CITATION_ID` cannot expose `ACCEPT_AS_DRAFT`.
- `LLM_ERROR + INVALID_CITATION_ID` cannot expose `ACCEPT_AS_DRAFT`.
- `TOO_LONG + EVIDENCE_GAP` can expose `ACCEPT_AS_DRAFT` when the evidence issue is waivable.
- A crafted draft-acceptance resume cannot bypass `accepted_choices`.

UI regressions cover internal protocol filtering, Markdown preview selection/rendering, and the no-Markdown case.

## Out of scope

- Structured task length and retry counters.
- Canonical heading assembly.
- Claim-aware RAG coverage.
- Message visibility whitelist migration.

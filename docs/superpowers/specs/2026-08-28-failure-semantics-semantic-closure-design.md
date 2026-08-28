# Failure Semantics / Recovery vNext Semantic Closure Design

## Scope

Close exactly four remaining semantics gaps without changing SQLite persistence,
adding graph nodes, rewriting the worker loop, or changing unrelated behavior:

1. Canonical blockers own every user-visible and accepted action.
2. Final checkpoint failure state owns JobStore completion projection.
3. Verifier issues carry exact stable requirement IDs.
4. PlanPatcher failures return to the canonical decision policy.

## Canonical Blocker Projection

`FailureDecision(USER_DECISION_REQUIRED, REGISTER_BLOCKER)` creates a
`UserBlockerRecord`. Its `available_options` are the only authoritative actions.
Both single and multiple blocker interrupts use the same blocker-keyed payload.
`pending_user_action` remains a one-way compatibility projection derived from
active canonical blockers. Old checkpoints without canonical blockers retain a
restricted legacy resume path, but canonical blockers can never be overridden by
legacy choices. Every canonical response is validated by
`apply_blocker_resolution()`. Canonical registry records also own the interrupt
payload when compatibility copies coexist, and a batch of resolutions is fully
validated before any upload ingestion side effect runs.

## Job Outcome Projection

A small dependency-free helper in `src/job_outcome.py` derives `waiting`, `failed`, or
`completed` from pending interrupt plus checkpoint state. Waiting has precedence;
otherwise any canonical fatal marker (`failure_decision`, `workflow_action`, or
`fatal_system_error`) produces `failed`. Streamlit and CLI use this helper both
when restoring a job and after graph execution. Python exceptions keep their
existing runner-owned fatal path.

## Requirement Linkage

`AssessmentIssue.requirement_ids` is a checkpoint-safe list with default `[]`.
LLM-provided IDs are intersected with active registry IDs valid for the active
task (or report-wide synthesis scope). Deterministic checks attach only matching
requirement kinds, such as length, asset, resource, evidence, or citation. Empty
linkage stays empty; policy never expands it to every task requirement.

## PlanPatcher Failure Classification

PlanPatcher continues to execute one bounded patch only. Exceptions become a
sanitized `PLAN_PATCH_INTERNAL_ERROR` assessment and are classified by
`decide_recovery_action()`, producing a canonical fatal decision without a
business blocker. Exhausted local-patch budgets are classified by policy: an
explicitly linked hard contract creates a canonical blocker; a safe current
result may degrade; otherwise the internal inability to continue is fatal.

## Compatibility

The new issue field defaults to an empty list for old checkpoints. Existing state
keys and SQLite saver/store layers are unchanged. `pending_user_action` remains
present only as a compatibility projection. No graph node is added. The obsolete
PlanPatcher `NEEDS_USER_INPUT` terminal mapping is replaced in place by
`FATAL_SYSTEM -> Exit`, closing the canonical fatal route without expanding the
graph topology.

## Verification

Use failing regression tests first for each of the four gaps, then run the
targeted recovery/verifier/runner suite, the complete pytest suite, compileall,
and static searches for direct business `pending_user_action` construction and
runner-side unconditional completion.

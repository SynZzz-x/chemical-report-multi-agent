# Scoped Evidence Gap Acceptance Design

## Goal

Make `ACCEPT_EVIDENCE_GAP` waive only evidence-availability gaps for the current task contract revision. Other defects in the same assessment remain actionable.

## State and scope

Store accepted evidence gaps in execution state under `accepted_evidence_gaps`, keyed by stable task ID. Each record contains `plan_revision`, `task_revision`, and the evidence issues shown to the user. A record applies only while both revisions still match; task or plan changes invalidate it automatically.

The waiver covers evidence-availability codes such as `EVIDENCE_GAP`, `INSUFFICIENT_EVIDENCE`, `MISSING_EVIDENCE`, `RAG_COVERAGE_GAP`, and `RAG_INSUFFICIENT`. It does not cover citation integrity, unsupported sources, length, content, or asset defects.

## Resume behavior

When the user selects `ACCEPT_EVIDENCE_GAP`:

1. Persist the scoped waiver.
2. Remove only waivable evidence-gap issues from the current assessment.
3. If no issues remain, commit the result and continue with section status `USER_ACCEPTED_GAP`.
4. If issues remain, run the existing recovery policy against those issues. For example, `TOO_LONG` routes to `REWORK`; an exhausted content retry routes to another explicit blocker.

Worker feedback contains only the remaining issues, so an accepted gap is not presented as another repair requirement.

## Subsequent verification and report admission

Before routing a later assessment for the same task and revision, remove only evidence-availability issues covered by the waiver. If all remaining checks pass, the policy still records `USER_ACCEPTED_GAP`, not `VERIFIED_PASS`, preserving draft-report semantics and the accepted gap audit trail.

## Non-goals

- Do not change Planner task granularity or `covers_sections`.
- Do not implement causal-figure degradation in this change.
- Do not change length targets or retry limits.
- Do not allow evidence-gap acceptance to waive citation or source-integrity failures.

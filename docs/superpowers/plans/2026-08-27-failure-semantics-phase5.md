# Failure Semantics vNext — Phase 5 Implementation Plan

**Status:** Completed
**Scope:** PlanPatcher route narrowing.

## TDD sequence

1. Add a closed set of stable plan-defect subtypes.
2. Add failing policy tests proving only those subtypes select `PATCH_PLAN`.
3. Add failing PlanPatcher validation tests rejecting evidence, asset, length,
   citation, format, and transient reason codes before mutation.
4. Keep existing task/job patch budgets, atomic validation, revisions, and
   history unchanged.
5. Run focused policy/patch/recovery graph tests and the full suite.

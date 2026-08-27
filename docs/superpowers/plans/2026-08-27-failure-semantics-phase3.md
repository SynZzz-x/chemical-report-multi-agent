# Failure Semantics vNext — Phase 3 Implementation Plan

**Status:** In progress  
**Scope:** DecisionPolicy mapping and bounded terminal outcomes only.

## Goal

Make every DecisionPolicy route publish a schema-valid `FailureDecision`, then
enforce the approved terminal boundaries without adding blocker registries or
changing persistence schemas.

## TDD sequence

1. Add focused failing tests for pass, retry, repair, degradation, hard user
   decision, and fatal decisions.
2. Resolve hard/soft severity exclusively through the active task's
   `requirement_ids` and `requirement_registry`.
3. Keep existing bounded repair actions, but make exhausted soft length,
   evidence, and optional asset paths terminal `COMMIT_WITH_WARNING` outcomes.
4. Make semantic Verifier exhaustion `FATAL_SYSTEM / VERIFIER_UNAVAILABLE`
   with no `pending_user_action`.
5. Preserve legacy workflow actions as graph routing compatibility projections
   while treating `failure_decision` as the canonical policy result.
6. Run focused recovery/Verifier tests, update obsolete assertions that encode
   pre-vNext Human routing, then run the full suite.

## Non-goals

- degradation/blocker registry identity and upsert behavior (Phase 4/6);
- dependency scheduler and consolidated interrupt (Phase 6);
- runner-level persistence fatal handling (Phase 8);
- UI or Summarizer changes (Phase 7).

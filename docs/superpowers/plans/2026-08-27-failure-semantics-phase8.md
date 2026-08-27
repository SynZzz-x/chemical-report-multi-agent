# Failure Semantics vNext — Phase 8 Implementation Plan

**Status:** Completed
**Scope:** Fatal-system handling, observability, legacy compatibility, and final
documentation.

## TDD sequence

1. Add safe graph/runner fatal record helpers and schema tests; diagnostics
   expose exception type and bounded operation metadata, never exception text.
2. Update Streamlit/CLI escaped stream and persistence handlers to mark job
   failure when projection is available, without fabricating blocker resumes.
3. Emit one structured `FAILURE_POLICY_DECISION` log for canonical policy
   results with no assessment/prompt/response bodies.
4. Normalize all new State containers at policy entry; reconstruct only
   authoritative legacy requirements and persist them in the returned update.
5. Add old-checkpoint, log-redaction, runner fatal, and schema compatibility
   tests.
6. Update implementation status/documentation, run the full suite, review the
   complete diff, and integrate the branch.

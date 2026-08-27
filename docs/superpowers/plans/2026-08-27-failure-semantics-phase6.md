# Failure Semantics vNext — Phase 6 Implementation Plan

**Status:** In progress  
**Scope:** Consolidated blocker admission, canonical resolution, and explicit
dependency-based affected-task resume.

## TDD sequence

1. Add pure tests for blocker identity (excluding revisions/attempts), upsert,
   transitive dependency closure, and runnable selection.
2. Register hard-contract decisions in `pending_user_blockers` while retaining
   the legacy single-blocker projection during migration.
3. Continue independent tasks after registration; route to NeedsUserInput only
   when no runnable work remains or Summarizer admission finds unresolved
   blockers.
4. Add canonical resolution tests for upload, requirement modification,
   approval authorization, cancellation, duplicate idempotence, and conflicting
   rejection.
5. Invalidate only the blocker task and explicit dependency descendants; keep
   unrelated committed/degraded results and revisions.
6. Extend NeedsUserInput to accept blocker-keyed consolidated submissions while
   preserving legacy resume payloads.
7. Run focused graph/control/persistence compatibility tests and the full suite.

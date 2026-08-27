# Failure Semantics vNext — Phase 4 Implementation Plan

**Status:** Completed
**Scope:** Degradation registry and local Evidence/Asset recovery boundaries.

## TDD sequence

1. Add failing tests for degradation record schema, deterministic identity,
   retry/resume upsert, and task-revision supersession.
2. Add a pure registry helper; derive IDs only from task ID, task revision,
   subtype, and sorted affected requirement IDs.
3. Register terminal soft evidence, optional asset, and soft quality outcomes
   from DecisionPolicy while preserving supported results.
4. Add boundary tests proving EvidenceRecovery and AssetRecovery return local
   outcomes and never invoke `interrupt` or create blocker payloads.
5. Run focused recovery/asset/synthesis tests and the full suite.

## Non-goals

- consolidated blocker identity/resolution (Phase 6);
- PlanPatcher narrowing (Phase 5);
- final report and UI projections (Phase 7).

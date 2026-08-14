# Constrained Synthesis Implementation Plan

1. Extend Planner and plan-patch task contracts with `synthesis` plus deterministic no-tool constraints.
2. Add tests for Planner validation, Graph routing, admitted-context construction, consistency checks, retry, and extractive fallback.
3. Implement the dedicated synthesis node and route initial execution/rework to it.
4. Run focused and full regression tests, then commit and push `codex/sqlite-checkpoint-store`.

# Failure Semantics vNext — Phase 7 Implementation Plan

**Status:** Completed
**Scope:** Consolidated blocker UI and degradation projections.

## TDD sequence

1. Add pure control-message tests for consolidated blocker forms, canonical
   action labels, validation, and blocker-keyed resume payloads.
2. Render one always-visible radio group per blocker in Streamlit; retain the
   legacy single-blocker radio path.
3. Admit system `ACCEPT_WITH_WARNING` sections as `DRAFT_WITH_GAPS` and render
   deterministic system-degradation warnings without user-acceptance wording.
4. Treat active optional-asset degradation as authority to remove dangling
   references/captions while preserving text, tables, citations, and valid
   assets.
5. Add UI progress projection for degradation and fatal outcomes.
6. Run focused control/UI/summarizer tests and the full suite.

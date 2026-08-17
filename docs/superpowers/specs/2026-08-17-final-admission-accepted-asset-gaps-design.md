# Final Admission for User-Accepted Asset Gaps

## Goal

When a user explicitly chooses `ACCEPT_AS_DRAFT`, final report assembly must not block again on the same missing figure or table defect. The report must remain a clearly labelled `DRAFT_WITH_GAPS` and omit the unavailable asset.

## Safety boundary

- Only a `USER_ACCEPTED_WARNING` section accepted by `user` may waive an asset-absence check.
- `MISSING_FIGURE_ASSET` is covered only by an accepted missing-figure issue.
- `MISSING_TABLE_ASSET` is covered only by an accepted missing-table issue.
- `MISSING_RESULT`, `REVISION_MISMATCH`, missing files, corrupt/invalid assets, unsupported sources, and citation-integrity defects remain blocking.
- A newly discovered asset defect that is absent from the accepted issue list remains blocking.

## Data flow

`needs_user_input()` stores the complete accepted assessment issues in `section_status[task_id]`. `summarizer_v2._ordered_sections()` compares a final missing-asset condition with that section-specific acceptance record. A covered absence is omitted from the assembled section; an uncovered or integrity-related failure is returned through `_blocked_update()`.

## Observability

The Summarizer logs the derived report status and section statuses before admission, logs each accepted asset degradation, and logs the exact blocking sections when admission fails.

## Tests

- Accepted `MISSING_FIGURE` produces a draft without a figure.
- Accepted `MISSING_TABLE` produces a draft without a table.
- Accepting an unrelated issue does not waive a newly discovered missing figure.
- Missing, corrupt, and malformed assets remain blocking.


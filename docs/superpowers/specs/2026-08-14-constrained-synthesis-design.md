# Constrained Synthesis Design

## Goal

Prevent conclusion/summary chapters from inventing facts after ordinary sections have already been accepted.

## Contract

- Add `task_type="synthesis"` for chapters whose only input is accepted prior sections.
- A synthesis task cannot use RAG, Web, files, figures, tables, or a Planner query.
- The Graph routes synthesis tasks to a dedicated node, never to the ordinary Worker tool graph.
- The node receives accepted section text, accepted citation records, and user-accepted evidence gaps for the current plan/task revisions.
- It may compress and reorganize that material, but may not introduce new evidence IDs, numbers, technical identifiers, or claims of tool/data-analysis activity.

## Execution and failure handling

The synthesis node calls the configured model without tools. A deterministic consistency gate validates the response. It retries once with concrete gate findings. If the second response still fails, it emits an extractive fallback composed only from sentences already present in accepted sections. Verifier review remains mandatory.

Verifier rework for a synthesis task returns to the synthesis node. Ordinary task routing is unchanged.

## Scope

This change does not modify final report formatting, evidence appendix layout, or outline-driven rendering.

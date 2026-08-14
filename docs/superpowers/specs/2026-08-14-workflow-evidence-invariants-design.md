# Workflow Acceptance and Evidence Identity Design

## Objective

Stabilize final-task acceptance, cross-task evidence identity, derived reference sections, synthesis provenance, and report rendering without introducing a GenericWorker or a broad runtime refactor.

## Invariant 1: Accepted Results Commit Once and Cannot Be Overwritten

`section_status` remains the single source of truth for section acceptance, and `report_status` remains derived from it. No duplicate `acceptance`, `verification`, or `publishability` fields will be added to task results.

User acceptance must be applied as one state transition:

1. Commit the current result for the active `task_id`, `plan_revision`, and `task_revision`.
2. Record `USER_ACCEPTED_GAP` or `USER_ACCEPTED_WARNING` with `accepted_by=user`.
3. Derive the job-level report status.
4. Route through the shared continuation rule: `DONE` for the terminal task, otherwise `NEXT`.

The transition must be idempotent for the same task revision. A terminal synthesis accepted as a draft must execute once, remain user-accepted, and proceed directly to deterministic report assembly.

## Invariant 2: Display Evidence IDs Are Never Internal Identity

Worker-local evidence IDs such as `E1` remain valid inside one task. Any cross-task operation must use a serializable composite identity containing:

- `task_id`
- `local_evidence_id`

The persisted representation may use explicit fields or a canonical string such as `T3:E1`; raw `E1` must not be used as a cross-task key.

At final report assembly, evidence is frozen into deterministic global display IDs using accepted task order and first appearance order. Rewriting is section-scoped and covers:

- body citation markers;
- structured citation records;
- figure evidence IDs;
- graph evidence references;
- table citation markers;
- the evidence appendix.

Internal identity correctness precedes source deduplication. Records that point to the same source locator or chunk may share one final display entry, but their originating task identities remain traceable in audit state.

## Derived Reference Section

Reference and knowledge-base-source sections are report projections, not Worker writing tasks. Outline classification will recognize titles such as “知识库依据与参考文件说明” as system-generated.

The final assembler will derive the section from citations belonging to admitted task revisions. Source names, locators, supported sections, and concise evidence summaries are deterministic. An LLM must not decide which sources exist.

If the outline requests the reference section at a particular location, assembly preserves that placement. The existing evidence appendix may be reused or adapted, but the report must not contain contradictory source statements.

## Synthesis Provenance

Synthesis remains a dedicated, tool-free aggregation node. The system derives claim lineage from accepted sections and evidence rather than trusting model-declared provenance.

Synthesis verification receives:

- accepted source sections;
- task-scoped evidence identities;
- deterministic consistency findings;
- the candidate conclusion.

Copied or extractively compressed accepted claims are not treated as newly invented facts. Claims without accepted-section or evidence lineage remain unsupported.

## Rendering Hardening

### PDF

The evidence appendix uses a compact four-column schema: evidence number, source, locator, and summary. The renderer supplies explicit column widths within the A4 content width, repeats the header row, uses smaller padding, and truncates presentation summaries while retaining full evidence text in audit state. Landscape switching is deferred.

### DOCX and Outline

Report assembly must emit a valid parent-child heading hierarchy before rendering. The DOCX renderer also handles missing parents defensively and must never create headings numbered `0.x`. A full Report AST is out of scope.

### Length Rewrite

Length-only rewriting remains tool-free and preserves source content and assets. Its upper target uses a configurable safety margin, defaulting to 92% of the requested maximum, to avoid repeated boundary retries.

## Tests

Regression coverage must demonstrate:

- terminal `ACCEPT_AS_DRAFT` commits once and routes to `DONE`;
- the accepted terminal synthesis is not executed again or overwritten;
- `T2:E1` and `T3:E1` remain distinct internally;
- final citation numbering is deterministic and consistent across body, assets, and appendix;
- the reference section is generated from accepted citations without Worker execution;
- synthesis verification can recognize accepted source claims;
- long evidence appendices render to PDF without negative-width failures;
- missing parent headings do not produce DOCX numbering beginning with zero;
- length rewrite targets the configured safety margin.

## Out of Scope

- GenericWorker extraction;
- RunContext redesign;
- process-wide tool lifecycle refactoring;
- a new persistent EvidenceStore service;
- a full Report AST or publication layout engine.

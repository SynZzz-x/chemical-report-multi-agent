# Final Citation Integrity and Recovery Aggregation Design

**Date:** 2026-08-29

**Branch:** `codex/sqlite-checkpoint-store`

**Baseline:** `cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb`

## 1. Scope and invariants

This change fixes five bounded problems exposed by a real offline/online run:

1. final report citation display IDs can become ambiguous after synthesis and
   local-to-global remapping;
2. a degradable evidence gap can incorrectly carry an unresolved semantic issue
   into `COMMIT_WITH_WARNING`;
3. strong material factual assertions without an explicit citation marker can
   bypass claim-level semantic verification;
4. Verifier evidence is repeatedly serialized and the actual provider request
   controls have not been proven at the HTTP request boundary;
5. the grouped evidence appendix remains difficult to read in narrow PDF pages.

The following constraints are hard non-goals:

- no State or checkpoint schema change;
- no checkpoint migration;
- no workflow topology change;
- no new Recovery branch or action enum;
- no new semantic LLM call;
- no lexical or embedding evidence-support gate;
- no automatic online DeepSeek benchmark;
- no renderer rewrite;
- no basename-only citation identity;
- no presentation dedupe that hides a citation conflict;
- no `COMMIT_WITH_WARNING` while any unresolved repairable issue exists.

All new correctness identities and validation results are invocation-local. They
are not written into State or checkpoints.

## 2. Current failure paths

### 2.1 Citation remapping

The current report flow is:

```text
Worker local citation
  -> committed section citation
  -> optional synthesis citation
  -> normalize_sections_evidence()
  -> local/task evidence_key to final display E-ID
  -> Markdown appendix
  -> PDF/DOCX
```

`build_display_evidence_map()` and `project_used_citations()` currently create a
dictionary keyed only by the visible `evidence_id`. If two different citation
records use the same local or display ID, the later item can silently overwrite
the earlier item before the report sees the conflict. A validator consuming
`_deduplicate_citations()` or `project_used_citations()` would therefore be too
late: the lossy projection has already hidden the evidence.

### 2.2 Recovery aggregation

`classify_assessment()` already examines every issue and is category-order
independent, but the four semantic claim codes share the `EVIDENCE_GAP` category
with ordinary waivable evidence gaps. When evidence recovery is exhausted, the
category branch can commit a soft result without proving that every unresolved
issue is degradable.

### 2.3 Verifier request cost

The current semantic payload repeats the full bounded excerpt for every
claim/evidence pair. Internal logs call the configured output limit
`requested_max_completion_tokens`, but DeepSeek Chat Completions expects an
outgoing `max_tokens` field. The current implementation has not proven that the
internal completion budget, reasoning effort, and thinking mode reach the
serialized provider request as intended.

## 3. P0-1: final global citation integrity

### 3.1 One correctness identity authority

Introduce one pure internal helper, conceptually:

```python
canonical_citation_identity(citation: Mapping[str, Any]) -> str
```

The helper promotes and reuses the stable evidence identity semantics already
present in `src/evidence/normalizer.py::_dedupe_key()` instead of inventing a
second presentation-oriented identity. The existing normalizer dedupe delegates
to the public helper after extraction.

Identity components use this priority:

```text
stable file/source ID or canonical URL
  -> existing canonical source identity
  -> normalized full source path fallback
+ locator
+ stable chunk/evidence identity
+ hash of full raw supporting evidence when no stronger evidence ID exists
```

Rules:

- an absolute job/cache path is a fallback, not the preferred authority;
- two copies of the same stable source do not become different merely because
  their cache paths differ;
- two unrelated files with the same basename do not become equal merely because
  `canonical_source_identity()` has presentation-grouping semantics;
- `supporting_text_excerpt` and every presentation sanitizer are forbidden from
  correctness identity;
- when text is the final fallback, hash the complete raw supporting evidence,
  not a bounded or cleaned display excerpt;
- full paths and internal IDs remain internal and never become display labels.

`canonical_source_identity()` continues to own source grouping for report
presentation. It is not silently redefined to expose full paths or to become the
sole correctness authority.

### 3.2 Lossless pre-remap validation

Before building the global display map, inspect every raw citation entry without
turning the list into a dictionary keyed by display ID.

For each `(task_id, local_evidence_id)`:

- zero or one canonical identity is valid;
- repeated entries with exactly the same canonical identity may be
  deterministically deduplicated;
- two different canonical identities are a hard conflict;
- no later entry may overwrite an earlier entry.

Across different task IDs, the same local label such as `E8` remains distinct
through the existing task-scoped `evidence_key`. Two tasks may keep separate
final IDs even when they cite identical canonical evidence; merging identical
evidence across tasks is not required by this change.

### 3.3 Remap without silent overwrite

`build_display_evidence_map()` continues to allocate final IDs in stable first
appearance order and preserves the existing `evidence_key` contract. Its lookup
must consume the validated, lossless citation records. It must not reconstruct a
single citation with `dict[evidence_id]`.

The real regression fixture contains:

```text
task A local E8 -> polyethylene process document / section 5
task B local E8 -> heat-exchanger maintenance guide / section 2.1
```

The two records must receive different final display IDs. A conflicting reuse
within the same task-scoped key must fail closed rather than be renumbered
arbitrarily.

### 3.4 Lossless final registry and post-remap gate

The final registry is projected directly from every remapped
`sections[*].citations` entry. It must not be sourced from
`_deduplicate_citations()` or `project_used_citations()`, because both existing
paths can discard conflicts by display ID.

A pure validator, conceptually:

```python
validate_final_citation_integrity(
    normalized_sections,
    final_markdown,
    lossless_final_citations,
) -> FinalCitationValidation
```

checks:

1. each final display ID resolves to exactly one canonical citation identity;
2. a display ID never resolves to two source authorities;
3. a display ID never resolves to conflicting locator, chunk, or evidence
   identities;
4. exact duplicate entries count as one resolution;
5. every inline body marker resolves exactly once;
6. the body-used ID set equals the projected used-registry ID set;
7. the assembled Markdown cannot introduce or drop body citation IDs; appendix
   repetition cannot make a missing body binding appear valid;
8. task-local IDs cannot reappear as an accidental global alias;
9. the same input produces byte-identical Markdown on repeated assembly.

The validator returns structured deterministic errors for tests and logging. It
does not invoke a model and does not modify citations.

### 3.5 Summarizer integration and failure behavior

The integration point is:

```text
normalize sections
  -> assemble final Markdown in memory
  -> validate lossless final registry and body markers
  -> only on PASS calculate/create report paths
  -> write Markdown
  -> render PDF/DOCX
```

The gate runs before `os.makedirs()` and before any renderer or file write. On
failure, Summarizer reuses its existing deterministic blocked/admission result
with a report-level `FINAL_CITATION_INTEGRITY` blocking item. It does not add a
workflow action, State field, or Recovery branch. Tests assert that no report
directory, Markdown, PDF, or DOCX is produced.

## 4. P0-2: order-independent multi-issue recovery

### 4.1 Assessment profile

Add a pure assessment-level profile that preserves the complete issue list and
derives:

```text
issue_count
has_non_degradable_issue
all_unresolved_issues_degradable
selected_policy_issue_code
selected_policy_action/tier
```

The four codes below are non-degradable semantic issues:

- `CLAIM_UNSUPPORTED`
- `CLAIM_PARTIALLY_SUPPORTED`
- `CLAIM_EVIDENCE_MISMATCH`
- `UNLABELED_INFERENCE`

Selection is deterministic and independent of input order. It uses the current
category priority for compatibility and a stable code tie-break inside the same
tier. It never recursively calls `decide_recovery_action()` per issue and thus
never consumes counters more than once.

Conceptual tiers are:

```text
fatal or blocking
  > repair / retry / recover
  > degradable commit with warning
  > next
```

The selected issue only explains the chosen policy. The full assessment is
passed unchanged into status, recovery, blocker, and observability handling.

### 4.2 Degradation invariant

The policy invariant is executable, not merely documented:

```text
COMMIT_WITH_WARNING if and only if
all unresolved issues are degradable at that terminal point.
```

Every `_commit_degraded_result()` entry receives a shared defensive check so a
future caller cannot bypass the invariant. Degradability remains based on the
existing single-issue rules for evidence, content, asset, and plan outcomes;
the new semantic codes are never waivable. The guard must not broadly reclassify
existing soft content/asset/plan behavior.

### 4.3 Semantic recovery state machine

Single-issue initial behavior is preserved:

```text
semantic issue
  -> existing EVIDENCE_RECOVERY when authorized and budget remains
  -> re-verify
  -> semantic issue still unresolved
  -> existing REWORK and existing task retry counter
  -> re-verify
  -> rewrite/retry budget exhausted
  -> existing NEEDS_USER_INPUT / REGISTER_BLOCKER
```

No new action or branch is introduced. A soft requirement does not permit a
non-degradable semantic issue to commit after retry exhaustion.

For a mixed assessment such as:

```text
EVIDENCE_GAP (degradable)
+ CLAIM_PARTIALLY_SUPPORTED (non-degradable)
```

the result can never be `COMMIT_WITH_WARNING` while the semantic issue remains.
Reordering the two issues produces the same action.

### 4.4 Observability

The existing failure-policy log gains:

```text
issue_count
selected_policy_issue_code
selected_policy_action
has_non_degradable_issue
```

The current subtype is not reused as the selected issue because it may be
derived from input order. No issue is removed or flattened; category and code
remain separate.

## 5. P1-1: explicitly grounded material assertions

### 5.1 Worker contract

The Worker prompt requires an inline citation on every material factual claim,
including quantitative, causal, priority/superlative, and strong operational
claims. Citations attach to the exact statement; a following statement never
inherits markers from a preceding sentence or paragraph.

An explicitly labelled material inference is not treated as a direct source
fact, but it must still cite the evidence used as its premise. For example,
`据此可推测...` without a citation remains a Worker contract violation even
though it is not classified as an unlabelled factual assertion.

### 5.2 Conservative deterministic detector

Add an invocation-local detector in `src/evidence/claims.py`, reusing the current
statement splitting and evidence-gap/recommendation/inference wording helpers.
It answers only:

> Does this prose statement require an explicit citation marker but lack one?

It never answers whether a citation supports a claim.

The first version is deliberately high precision:

- skip statements that already contain a valid marker;
- skip headings, code, table separators, labels, and non-prose fragments;
- exclude explicit evidence-gap wording;
- exclude explicit recommendations from the factual rule;
- treat explicitly labelled material inference separately: it is allowed as
  inference wording only when it cites premise evidence;
- require a strong material pattern together with an assertive predicate or
  sentence form, rather than firing on an isolated word such as `最`, `核心`,
  `优先`, or `必须`;
- quantitative detection requires a number plus a unit/percentage and a
  comparison, threshold, or causal predicate.

High-confidence initial patterns cover constructions such as `最直接、最灵敏的
调节手段`, `核心控制变量`, `关键决定因素`, `主要由...决定`, `直接决定`,
`导致`, and `使...升高/降低`. The design prefers a false negative over a
recommendation or heading false positive.

An uncited material assertion returns deterministic
`UNCITED_MATERIAL_CLAIM` with bounded sentence context and category
`CONTENT_DEFECT`. It fast-fails before the semantic LLM and routes through the
existing `REWORK` path. It is non-degradable and cannot commit with warning.

The resulting verifier flow is:

```text
deterministic citation identity/binding preflight
  -> conservative uncited material gate
  -> failure: REWORK without semantic call
  -> derive cited claims and explicit evidence gaps
  -> resolve evidence
  -> one semantic Verifier call
```

## 6. P1-2: measure and safely reduce Verifier cost

### 6.1 Deterministic prompt contribution measurement

The offline benchmark records characters, not provider tokens, for:

- base instructions;
- task contract;
- Worker result;
- claim references;
- evidence catalog;
- Worker assets;
- other deterministic context;
- total serialized prompt.

Mock completion characters, call count, and any existing stable deterministic
token estimate remain separate metrics. Baseline measurements are captured from
the frozen baseline commit with the same fixture before optimized measurements;
tests do not copy the old production prompt builder.

### 6.2 Evidence catalog dedupe

Replace repeated embedded excerpts:

```text
C1 -> full E3
C2 -> full E3
C3 -> full E3
```

with one catalog and ID references:

```text
evidence_catalog: E3 -> semantic excerpt once
claims: C1 -> [E3], C2 -> [E3], C3 -> [E3]
```

This remains one AutoVerifier call with the same semantic authority. The current
semantic head/tail excerpt bound remains unchanged unless contribution
measurement plus real regression fixtures prove that a smaller bound retains the
critical E3/E6 support text. Presentation excerpts are never used for semantic
verification.

### 6.3 Provider-control audit

Before claiming any completion or reasoning control, trace:

```text
internal completion budget
  -> LangChain ChatOpenAI wrapper
  -> serialized Chat Completions request
  -> actual max_tokens, reasoning_effort, and thinking fields
```

DeepSeek Chat Completions defines the output budget field as `max_tokens` and
supports `reasoning_effort=low|high|max`. Internal log naming does not prove the
outgoing payload. Offline fake-client/request-serialization tests must prove:

- the Verifier assessment budget becomes provider `max_tokens`;
- an explicitly configured reasoning effort reaches `reasoning_effort`;
- absent Verifier-specific configuration preserves the existing payload and
  model;
- no unsupported `max_completion_tokens` field is sent to DeepSeek;
- thinking mode is only reported as controlled if the outgoing `thinking`
  mapping is explicitly proven.

Add optional backward-compatible configuration:

```text
VERIFIER_MODEL
VERIFIER_REASONING_EFFORT
```

`VERIFIER_REASONING_EFFORT` accepts only provider-supported values. It is passed
only when explicitly configured and only through a mechanism already supported
by the current `ChatOpenAI` wrapper (`reasoning_effort`, `model_kwargs`, or
`extra_body`, as established by the serialization test). No aggressive default
is introduced. A numeric reasoning-token budget is not invented.

Online latency, provider token reduction, and quality effects remain
`not re-measured online / requires real-run validation`.

Provider behavior in this section is based on the official DeepSeek Chat
Completions and Thinking Mode documentation:

- <https://api-docs.deepseek.com/api/create-chat-completion/>
- <https://api-docs.deepseek.com/guides/thinking_mode/>

## 7. P2: presentation-only appendix simplification

The presentation sanitizer removes repeated ASCII `|` and full-width `｜` table
noise without mutating `supporting_text` or provenance.

The grouped four-column table becomes stable per-evidence blocks:

```text
### Safe source title

**[E18] locator**
supporting section: ...
summary: ...
```

Source groups keep stable first-appearance order; entries use natural display-ID
order. Exact duplicate rows with the same display ID and canonical citation
identity may be shown once. A display ID with multiple identities is a P0 failure
and cannot be hidden by presentation dedupe. Existing Markdown-to-PDF and
Markdown-to-DOCX renderers remain unchanged.

## 8. Test strategy and checkpoints

All tests are offline and deterministic. Production behavior changes use
red-green TDD.

### P0-1 regressions

- different task-local `E8` records remap to different final IDs;
- same task-scoped ID with different identities fails before remap;
- missing body marker registry entry fails;
- one final ID with two identities fails;
- exact duplicate identity is allowed;
- the gate consumes lossless records and catches conflicts hidden by current
  lossy dictionary projections;
- failure invokes no renderer and creates no report directory;
- two assemblies produce byte-identical Markdown.

### P0-2 regressions

- one degradable gap commits with warning after its existing terminal policy;
- one semantic issue follows evidence recovery, then rework, then blocker;
- degradable gap plus semantic issue never commits with warning;
- two degradable gaps may commit with warning;
- issue permutation produces the same action and selected policy code;
- all issues and semantic claim context survive into recovery/blocker records;
- `_commit_degraded_result()` defensively rejects a non-degradable assessment.

### P1-1 regressions

- the two real uncited `最直接/最灵敏` and `核心控制变量` assertions fail;
- `建议优先检查氢气比` does not trigger the factual rule;
- explicit evidence-gap wording does not trigger;
- a cited causal assertion proceeds to the semantic verifier;
- an explicitly labelled material inference without premise evidence is rejected;
- headings, tables, code, and isolated trigger words do not create false issues;
- deterministic failure performs zero semantic calls.

### P1-2 regressions

- component character accounting sums to the serialized prompt total;
- repeated E3 is serialized once and referenced by multiple claims;
- semantic call count remains one;
- critical support text remains in semantic E3/E6 fixtures;
- fake request serialization proves `max_tokens` and optional
  `reasoning_effort` mapping;
- default configuration produces no Verifier-specific behavior change.

### P2 regressions

- ASCII and full-width separator noise is removed;
- appendix blocks are deterministic and readable without a four-column table;
- input citations remain deep-equal before and after rendering;
- exact duplicate rows collapse, while conflicting identities fail in P0.

Mandatory checkpoints:

1. P0-1 focused tests and commit;
2. P0-2 focused tests and commit;
3. full `pytest` checkpoint;
4. P1-1 commit;
5. P1-2 measurement, request mapping audit, and commit;
6. P2 commit;
7. full tests, compileall, `git diff --check`, and adversarial architecture
   review.

## 9. Commit structure

Suggested commits:

1. `report: validate final citation identity`
2. `recovery: aggregate unresolved issue severity`
3. `verifier: reject uncited material factual claims`
4. `verifier: deduplicate semantic evidence payload`
5. `report: simplify evidence appendix projection`

Correctness and presentation changes remain separated. Any provider-control
configuration plumbing belongs with the Verifier payload/measurement commit and
must not imply an online performance result.

## 10. Final acceptance

The implementation is acceptable only if it proves all of the following:

- final display IDs are globally unambiguous and every body marker resolves once;
- silent display-ID overwrite is impossible before and after remap;
- conflicting local IDs become distinct across tasks or fail closed within one
  task-scoped identity;
- multi-issue action selection is order independent;
- `COMMIT_WITH_WARNING` occurs if and only if every unresolved issue is
  degradable;
- semantic issues cannot hitchhike on degradable evidence gaps;
- strong uncited factual assertions and ungrounded material inferences are caught
  without treating recommendations as facts;
- deterministic preflight still prevents unnecessary semantic calls;
- the semantic Verifier remains exactly one call;
- request serialization, not log labels, proves any DeepSeek parameter control;
- old checkpoints resume without new fields;
- full provenance remains unchanged;
- appendix presentation never hides an identity conflict;
- no forbidden State, topology, Recovery, semantic-gate, renderer, or online
  benchmark change appears in the diff.

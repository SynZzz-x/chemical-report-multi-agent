# Claim Grounding and Pipeline Efficiency Design

## Scope

Incrementally improve claim-level evidence grounding, AutoVerifier efficiency,
LLM usage telemetry, Worker RAG prefetch reuse, Planner/Intake prompt size, and
the final evidence appendix on baseline commit
`3ba9fd3eb3ad84b193f699e72e15bc40bea40446`.

The stable workflow, persistence, DecisionPolicy, Recovery, and Summarizer
admission architecture remain authoritative.

### Explicit non-goals

- No State or checkpoint schema changes.
- No new LLM round trips.
- No lexical or embedding-based semantic support gate.
- No new Recovery branch.
- No persistence, workflow topology, or renderer rewrite.
- No online DeepSeek, embedding-service, or network calls in tests.

## 1. Claim-level evidence verification

### Responsibility boundary

The deterministic layer decides whether citation structure is valid. The
existing single AutoVerifier LLM call decides whether cited evidence
semantically supports a claim.

The execution order is fail-closed:

1. Read `current_result`.
2. Run deterministic citation identity and binding preflight.
3. If an evidence ID is invented, malformed, or structurally unbound, return a
   failed assessment without calling the verifier LLM.
4. Derive invocation-local claims.
5. Resolve only validated evidence IDs to their supporting evidence.
6. Send one compact semantic payload to the existing AutoVerifier call.
7. Parse and sanitize the existing `VerifierAssessment` contract.
8. Reapply deterministic validation as the final veto.
9. Continue through the existing DecisionPolicy and Recovery path.

### Invocation-local claim contract

Claims are derived data and are never written to State or a checkpoint:

```text
claim_id: stable within the invocation
text: bounded claim text
claim_type: factual | inference | recommendation | evidence_gap
evidence_ids: validated evidence IDs
evidence: [{evidence_id, title, locator, supporting_text_excerpt}]
```

Extraction is deliberately local rather than a full-document claim extractor:

- A statement containing `[E编号]` becomes a cited claim.
- Explicit evidence-gap wording becomes an `evidence_gap` claim even when it
  has no citation and therefore has `evidence_ids=[]` and `evidence=[]`.
- Other uncited prose is not force-extracted in this iteration.
- Explicit inference wording such as “据此可推测” and “综合来看” is classified
  as `inference`; explicit engineering advice such as “工程上可考虑” is
  classified as `recommendation`; remaining cited statements default to
  `factual`.
- The deterministic extractor does not identify `UNLABELED_INFERENCE`.
  A factual-looking claim that is actually an inference is detected by the
  semantic verifier.

### Semantic issue codes

The issue `category` remains `EVIDENCE_GAP`; the distinct `code` is preserved
through sanitization, logging, benchmarking, and Recovery input.

- `CLAIM_UNSUPPORTED`: evidence concerns the same subject but cannot establish
  the claim's core assertion. Example: evidence states that temperature affects
  molecular weight, while the claim asserts a specific chain-transfer
  mechanism and direction.
- `CLAIM_PARTIALLY_SUPPORTED`: a material portion of a compound claim is
  supported, but the claim adds scope, mechanism, sequence, causality, degree,
  priority, or threshold. Recovery guidance permits either narrowing the claim
  or obtaining additional evidence.
- `CLAIM_EVIDENCE_MISMATCH`: the cited evidence supports a different object,
  variable, relationship, or phenomenon. This means the citation is wrong, not
  merely weak.
- `UNLABELED_INFERENCE`: evidence can serve as a premise, but the new conclusion
  is presented as if directly stated by the source.

`requirement_ids` may contain only active legal requirement IDs already visible
to the task. Empty linkage remains empty. `retrieval_query` is optional because
some semantic issues should be repaired by rewriting or narrowing a claim rather
than retrieving more evidence.

### AutoVerifier output size

The existing assessment schema is unchanged. A passing response contains only
the required fields with empty collections. A failed response contains only
issues that Recovery can act on and does not summarize or rewrite the report.
Deterministic preflight continues to avoid the semantic LLM call when it already
has a decisive failure.

## 2. LLM usage observability

Provider usage is recorded faithfully with explicit names:

```text
requested_max_completion_tokens
provider_prompt_tokens
provider_completion_tokens
provider_reasoning_tokens  # only when supplied by the provider
provider_total_tokens
```

Compatibility fields remain available with the following mapping:

```text
input_tokens  -> provider_prompt_tokens
output_tokens -> provider_completion_tokens
total_tokens  -> provider_total_tokens
```

`provider_completion_tokens` retains the provider's definition and may include
provider-accounted reasoning usage. It is not described as visible response
text. If the provider exposes a separate reasoning-token field, it is logged
separately; the implementation does not fabricate or recompute it.

## 3. Worker RAG prefetch reuse

### One query identity authority

A single pure query-normalization helper is consumed by prefetch, the tool-loop
duplicate guard, and benchmark fixtures. It performs only deterministic textual
normalization. Fingerprints identify normalized queries for exact structural
deduplication; they are not semantic-equivalence judgments. No embeddings,
similarity thresholds, state, or network access are involved.

### First-iteration retrieval inventory

`_prefetch_rag()` continues to return the existing tool-call records. Before the
first Worker generation, these records are projected into a model-readable
completed-retrieval inventory containing:

```text
prefetched_queries
query_fingerprints
evidence: [{evidence_id, title, locator, supporting_text_excerpt}]
prefetch_queries_used
adaptive_queries_remaining
```

Evidence excerpts use the shared presentation sanitizer and a fixed bound so a
full RAG chunk is not duplicated into the prompt. The message tells the model
that prefetch is already complete, that sufficient evidence should lead directly
to final content, and that adaptive RAG may be used only for a concrete new
evidence gap. Existing duplicate guards remain the final deterministic defense.

Acceptance scenarios are:

- Sufficient Q1 prefetch: one Worker generation, one tool-loop iteration, no
  adaptive tool call, and no duplicate retrieval.
- Normalized-equivalent Q1 request: the duplicate guard rejects it without
  executing retrieval.
- Materially distinct Q2 request for a concrete gap: adaptive retrieval runs and
  is not rejected as a duplicate.

No prefetch metadata is added to persistent State.

## 4. Intake and Planner prompt compression

Intake and Planner retain their JSON contracts, validators, repair attempts, and
backward compatibility. Changes are restricted to removing repeated prompt
instructions and repeated serialized context. Planner remains responsible only
for the execution contract: task identity/type, requirements, resources, tool
policy, dependencies, and necessary retrieval query.

Prompt-size regressions compare serialized messages before and after the focused
changes. No contract field is removed and no validation standard is relaxed.

## 5. Deterministic offline benchmark

Tests use fixed fake model responses and never contact DeepSeek, an embedding
service, or the public network. Metrics remain distinct:

- LLM calls by Intake, Planner, Worker, and Verifier.
- Worker generations and tool-loop iterations.
- Duplicate and adaptive retrieval calls.
- Serialized prompt characters.
- Mock completion characters.
- Deterministic token estimate only if the repository already exposes a stable
  tokenizer suitable for this use; otherwise characters are reported as a
  proxy and never labelled provider tokens.
- Local execution time may detect algorithmic regressions but is not used to
  claim online model speedup.

The supplied online baseline remains reference-only:

| Metric | Online baseline | Optimized |
| --- | ---: | --- |
| Total reported provider tokens | 42,961 | Not re-measured online |
| Verifier reported provider tokens | 17,717 | Not re-measured online |
| Total LLM latency | 172.2 s | Requires real-run validation |
| Verifier latency | 84.7 s | Requires real-run validation |

The implementation may claim only deterministic structural improvements such
as fewer calls, smaller prompts/completions, fewer tool-loop iterations, and no
duplicate retrieval.

## 6. Evidence appendix projection

The final Markdown evidence appendix groups citations by canonical source
identity while preserving citation IDs and traceability:

```text
safe source display label
  E1 — locator — bounded summary
  E2 — locator — bounded summary
```

Source groups use stable first-appearance order. Evidence inside a group uses
natural display-ID order (`E1`, `E2`, through `E10`). Canonical source identity is
used only as the grouping key; the user-visible label prefers an existing safe
display title or filename and never exposes absolute cache paths, user IDs,
conversation IDs, or job IDs.

A presentation-only sanitizer collapses whitespace, removes repeated table
separator noise such as long `|` sequences, and applies a fixed excerpt bound.
It never mutates the citation registry or provenance objects. Safe locators such
as a section, page, or table remain visible; internal paths and chunk IDs do not.
Grouping follows the actual semantics of the existing
`canonical_source_identity()` helper and does not introduce a second basename
grouping rule. Regression tests document whether two same-basename paths are one
or two identities under that helper rather than silently changing its contract.

The body-text citation mechanism (`[E编号]`) is unchanged. Full provenance remains in result and
checkpoint data. Existing PDF and DOCX renderers continue to consume the same
assembled Markdown.

## 7. Initialization and warning audit

Directory creation, built-in tool registration, and immutable registry setup are
audited before any change. Caching is permitted only for objects proven immutable
and process-global. Job paths, task state, RAG results, LLM clients, and mutable
registries are never cached across jobs.

Repeated idempotent directory creation is preferable to unsafe cross-job state.
The `jieba/pkg_resources` warning is traced to its dependency. It is fixed only
when a dependency-safe change exists; otherwise it is recorded as known technical
debt and is not globally suppressed. A zero-production-code result is acceptable
for this phase.

## 8. Testing and delivery

Every behavior change follows red-green TDD with deterministic fake models and
tools. Each subsystem's focused tests run before the next stage. The final gate
includes the full pytest suite, checkpoint/store/recovery tests, citation
integrity tests, compileall, and `git diff --check`.

Implementation commits remain independently reviewable:

1. Claim extraction and semantic verifier regressions.
2. Compact verifier output and provider usage telemetry.
3. Worker prefetch inventory and duplicate-query handling.
4. Intake/Planner prompt compression and offline benchmark.
5. Grouped evidence appendix.
6. Optional low-risk initialization cleanup or an audit-only conclusion.

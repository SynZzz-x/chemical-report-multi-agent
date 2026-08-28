# Claim Grounding and Pipeline Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add claim-level semantic evidence verification while reducing avoidable Verifier/Worker/Planner cost and producing a compact, traceable evidence appendix.

**Architecture:** Keep State, checkpoints, graph topology, DecisionPolicy, Recovery, and Summarizer admission unchanged. Derive claims and prefetch inventories only within one invocation, keep deterministic citation checks as the final authority, and reuse the existing single AutoVerifier semantic call. Measure performance with fixed offline fixtures and preserve provider telemetry compatibility.

**Tech Stack:** Python 3.14, Pydantic v2, LangChain/LangGraph, pytest, existing SQLite persistence, mocked LLM/tool runnables.

## Global Constraints

- Baseline production commit is `3ba9fd3eb3ad84b193f699e72e15bc40bea40446`.
- Do not modify State/checkpoint schemas, persistence, workflow topology, DecisionPolicy branches, Recovery branches, or Summarizer admission.
- Do not add an LLM call or a lexical/embedding semantic-support gate.
- Keep deterministic citation validation fail-closed and execute it before semantic claim derivation.
- Tests must not contact DeepSeek, embedding services, or the public network.
- Keep Planner/Intake contracts, validators, and retry counts backward compatible.
- Do not mutate citation/provenance records for presentation.
- Use red-green TDD for every production behavior change and commit each reviewable stage separately.

---

### Task 1: Freeze deterministic performance baselines

**Files:**
- Create: `tests/benchmark_support.py`
- Create: `tests/test_offline_pipeline_benchmark.py`
- Create: `docs/benchmarks/2026-08-28-pipeline-baseline.json`

**Interfaces:**
- Produces: `measure_serialized_messages(messages) -> dict[str, int]` and fixed Scenario A/B fixtures reused by optimized benchmark tests.
- Produces: immutable baseline measurements for prompt characters, completion characters, calls, tool-loop iterations, and duplicate retrievals.

- [ ] **Step 1: Verify the production baseline is unchanged**

Run:

```bash
git diff --exit-code 3ba9fd3eb3ad84b193f699e72e15bc40bea40446 -- app.py run.py src
```

Expected: exit 0. The two design-document commits must be the only changes since the production baseline.

- [ ] **Step 2: Add fixed offline fixtures and measurement helpers**

Create `tests/benchmark_support.py` with a fake runnable that records each invocation without network access:

```python
from dataclasses import dataclass, field
import json
from typing import Any


def serialized_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(len(str(getattr(item, "content", item))) for item in value)
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


@dataclass
class BenchmarkRecorder:
    response: Any
    calls: list[Any] = field(default_factory=list)

    def invoke(self, value: Any, **kwargs: Any) -> Any:
        self.calls.append(value)
        return self.response

    @property
    def prompt_chars(self) -> int:
        return sum(serialized_chars(value) for value in self.calls)
```

Include fixed chemical-report inputs for:

- Scenario A: one Worker generation with no tool request.
- Scenario B: planner query `聚乙烯 质量异常 排查 建议 常见异常`, successful prefetch evidence, and a repeated first adaptive request.
- One PASS verifier response using the current required assessment fields.
- One FAILED semantic response containing `CLAIM_PARTIALLY_SUPPORTED`.

- [ ] **Step 3: Add a characterization benchmark test**

Create `tests/test_offline_pipeline_benchmark.py` that exercises current Intake, Planner prompt formatting, Worker tool loop, and Verifier prompt formatting through fakes. Assert only measurement integrity at this stage:

```python
def test_offline_benchmark_metrics_are_deterministic():
    first = collect_pipeline_metrics()
    second = collect_pipeline_metrics()
    assert first == second
    assert first["total_llm_calls"] >= 1
    assert first["serialized_prompt_chars"] > 0
    assert first["mock_completion_chars"] > 0
```

The metric dictionary must contain separate keys for Intake, Planner, Worker,
Verifier, Worker generations, tool-loop iterations, duplicate retrievals,
serialized prompt characters, and mock completion characters. Do not label
character counts as provider tokens.

- [ ] **Step 4: Run the baseline fixture twice and record the result**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_offline_pipeline_benchmark.py
.venv/bin/python -m pytest -q tests/test_offline_pipeline_benchmark.py
```

Expected: identical metrics on both runs. Write those exact measurements and the
baseline commit into `docs/benchmarks/2026-08-28-pipeline-baseline.json`. Do not
copy any old prompt builder or tool loop into tests.

- [ ] **Step 5: Commit the benchmark characterization**

```bash
git add tests/benchmark_support.py tests/test_offline_pipeline_benchmark.py docs/benchmarks/2026-08-28-pipeline-baseline.json
git commit -m "test: freeze offline pipeline baseline"
```

---

### Task 2: Derive invocation-local claims and evidence projections

**Files:**
- Create: `src/evidence/text_projection.py`
- Create: `src/evidence/claims.py`
- Create: `tests/test_claim_grounding.py`

**Interfaces:**
- Produces: `normalize_evidence_text(value: Any) -> str`.
- Produces: `semantic_evidence_excerpt(value: Any, limit: int = 4000) -> str` using a wide head/tail bound.
- Produces: `presentation_evidence_excerpt(value: Any, limit: int = 240) -> str` using a display-oriented bound.
- Produces: `derive_claims(content: str, citations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing text-projection and claim-extraction tests**

Add tests that require separate semantic and presentation bounds and prove no
mutation:

```python
def test_semantic_excerpt_preserves_late_support_while_presentation_stays_short():
    raw = "背景" * 2200 + "氢气乙烯比升高导致分子量下降。"
    assert "分子量下降" in semantic_evidence_excerpt(raw)
    assert len(presentation_evidence_excerpt(raw)) <= 240


def test_evidence_gap_without_citation_is_derived_locally():
    claims = derive_claims(
        "当前知识库未检出反应压力对 Mw/Mn 的直接证据，本报告不作推断。",
        [],
    )
    assert claims == [{
        "claim_id": "C1",
        "text": "当前知识库未检出反应压力对 Mw/Mn 的直接证据，本报告不作推断。",
        "claim_type": "evidence_gap",
        "evidence_ids": [],
        "evidence": [],
    }]
```

Also cover factual, explicit inference, recommendation, a compound cited claim,
multiple known evidence IDs, and an unknown ID never being resolved.

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py
```

Expected: import failure because `src.evidence.claims` and
`src.evidence.text_projection` do not exist.

- [ ] **Step 3: Implement the minimal pure helpers**

In `src/evidence/text_projection.py`, normalize whitespace and repeated table
separators without changing source objects. The semantic excerpt keeps both ends
when truncation is required; the presentation excerpt uses the bounded beginning.

In `src/evidence/claims.py`, split only at deterministic paragraph/sentence
boundaries, derive evidence-gap statements even without citations, use
`collect_used_evidence_ids()` for ordered validated IDs, and classify explicit
markers with this precedence:

```python
if evidence_gap_wording(statement):
    claim_type = "evidence_gap"
elif recommendation_wording(statement):
    claim_type = "recommendation"
elif inference_wording(statement):
    claim_type = "inference"
else:
    claim_type = "factual"
```

Do not implement semantic entailment or unlabeled-inference detection.

- [ ] **Step 4: Run focused evidence tests**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_report_evidence_integration.py
```

Expected: all pass.

- [ ] **Step 5: Commit the claim projection helpers**

```bash
git add src/evidence/claims.py src/evidence/text_projection.py tests/test_claim_grounding.py
git commit -m "verifier: derive local claim evidence pairs"
```

---

### Task 3: Integrate semantic claims into the existing AutoVerifier call

**Files:**
- Modify: `src/nodes/verifier.py`
- Modify: `src/prompts/verifier.md`
- Modify: `tests/test_claim_grounding.py`
- Modify: `tests/test_auto_verifier_routing.py`
- Modify: `tests/test_failure_decision_policy.py`

**Interfaces:**
- Consumes: `derive_claims()` from Task 2.
- Preserves: `VerifierAssessment`, `_sanitize_assessment()`, `_apply_citation_integrity()`, `_apply_deterministic_validation()`, and existing Recovery routing.
- Produces: one `claim_evidence_pairs` prompt variable in the existing assessment call.

- [ ] **Step 1: Write failing real-case semantic regressions**

Use fake verifier responses and capture the formatted prompt. Add the supplied
E6 cases:

```python
@pytest.mark.parametrize("claim", [
    "单变量验证，每次调整仅改变一个因素，用数据验证假设。[E6]",
    "先排除分析误差、公用工程波动等系统性因素。[E6]",
])
def test_existing_e6_does_not_make_unsupported_claim_pass(claim, fake_verifier):
    result = run_verifier_with_fake_semantic_issue(claim, E6_TREND_EVIDENCE)
    assert result["assessment"]["status"] == "FAILED"
    assert result["assessment"]["issues"][0]["code"] in {
        "CLAIM_UNSUPPORTED", "CLAIM_EVIDENCE_MISMATCH"
    }
```

Add separate tests for supported factual text, explicit inference, unlabeled
inference, evidence-gap disclosure, partially supported temperature mechanism,
and preservation of invented-ID preflight with zero fake-model calls.

- [ ] **Step 2: Run the regressions and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_auto_verifier_routing.py
```

Expected: semantic codes are rejected or missing from the formatted prompt.

- [ ] **Step 3: Extend verifier code/category mappings without changing Recovery**

Add the four claim codes to `_CATEGORY_BY_CODE` with category `EVIDENCE_GAP` and
to `_REQUIREMENT_KINDS_BY_CODE` with kinds `citation` and `evidence`. Keep the
original code during sanitization; do not flatten it to generic `EVIDENCE_GAP`.

- [ ] **Step 4: Add claim pairs after deterministic preflight**

Only after `_deterministic_preflight()` returns non-failed, call
`derive_claims(content, citations)` and serialize the result once as
`claim_evidence_pairs`. Replace the full duplicate citation payload in
`worker_assets` with bounded citation metadata (`evidence_id`, `title`,
`locator`); semantic excerpts live only in claim pairs.

- [ ] **Step 5: Tighten the verifier prompt mechanically**

Add the four exact issue definitions from the design. State that:

- citation existence alone never establishes support;
- compound claims with unsupported material clauses are
  `CLAIM_PARTIALLY_SUPPORTED`;
- a factual-looking derived conclusion is `UNLABELED_INFERENCE`;
- evidence-gap disclosure may have no citation;
- PASS uses only required fields and empty collections;
- FAILED returns only actionable issues and never rewrites the report.

Do not change `VerifierAssessment` fields or add a second call.

- [ ] **Step 6: Verify sanitize, deterministic veto, and Recovery compatibility**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_failure_decision_policy.py tests/test_recovery_policy.py
```

Expected: all pass; invented IDs still fast-fail without LLM; claim issue codes
survive sanitize and reach the existing evidence-gap policy.

- [ ] **Step 7: Commit semantic verification**

```bash
git add src/nodes/verifier.py src/prompts/verifier.md tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_failure_decision_policy.py
git commit -m "verifier: check claim-level evidence support"
```

---

### Task 4: Compact verifier responses and map provider usage faithfully

**Files:**
- Modify: `src/llm.py`
- Modify: `src/prompts/verifier.md`
- Modify: `tests/test_llm_observability.py`
- Modify: `tests/test_offline_pipeline_benchmark.py`

**Interfaces:**
- Produces: frozen `ProviderTokenUsage` with optional prompt, completion, reasoning, and total fields.
- Produces: `extract_provider_token_usage(response: Any) -> ProviderTokenUsage`.
- Preserves: `extract_token_usage(response) -> tuple[int | None, int | None, int | None]` as a compatibility adapter.

- [ ] **Step 1: Write failing provider-usage mapping tests**

Cover LangChain metadata and OpenAI-compatible metadata, including nested
reasoning details:

```python
def test_provider_usage_keeps_reasoning_separate_when_supplied():
    response = SimpleNamespace(response_metadata={"token_usage": {
        "prompt_tokens": 10,
        "completion_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 20},
        "total_tokens": 40,
    }})
    usage = extract_provider_token_usage(response)
    assert usage.provider_prompt_tokens == 10
    assert usage.provider_completion_tokens == 30
    assert usage.provider_reasoning_tokens == 20
    assert usage.provider_total_tokens == 40
```

Also assert missing reasoning remains `None`, legacy tuple output is unchanged,
and logs contain both the new names and compatibility aliases without prompt or
response contents.

- [ ] **Step 2: Run tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_llm_observability.py
```

Expected: `extract_provider_token_usage` is missing.

- [ ] **Step 3: Implement named provider usage**

Add:

```python
@dataclass(frozen=True)
class ProviderTokenUsage:
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_reasoning_tokens: int | None = None
    provider_total_tokens: int | None = None
```

Read provider values without recomputing completion or reasoning. Derive total
only under the existing compatibility rule when prompt and completion exist and
the provider omitted total. Keep `extract_token_usage()` delegating to the named
result.

- [ ] **Step 4: Extend logs compatibly**

In `LLM_CALL_START`, add `requested_max_completion_tokens` while retaining
`max_completion_tokens`. In `LLM_CALL_END`, log the four provider fields and
retain `input_tokens`, `output_tokens`, and `total_tokens` as aliases. Add a code
comment that provider completion usage is not guaranteed to equal visible text.

- [ ] **Step 5: Add compact PASS/FAILED benchmark assertions**

Use the same fixed verifier input and fake outputs. Assert one semantic call in
both cases, that PASS serializes to the minimum existing schema fields, and that
FAILED retains claim code, claim context, suggestion, and optional retrieval
query. Compare optimized prompt/completion characters with the frozen baseline;
do not assert provider tokens or online latency.

- [ ] **Step 6: Run P0 tests and full pytest checkpoint**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_llm_observability.py tests/test_failure_decision_policy.py tests/test_recovery_policy.py tests/test_job_store.py
.venv/bin/python -m pytest -q
```

Expected: all pass.

- [ ] **Step 7: Commit verifier compactness and telemetry**

```bash
git add src/llm.py src/prompts/verifier.md tests/test_llm_observability.py tests/test_offline_pipeline_benchmark.py
git commit -m "verifier: compact output and clarify token usage"
```

---

### Task 5: Expose completed RAG prefetch and use one query identity

**Files:**
- Create: `src/evidence/query_identity.py`
- Modify: `src/nodes/worker/agent/graph.py`
- Modify: `src/prompts/worker_system_template.md`
- Modify: `tests/test_worker_tool_boundaries.py`
- Modify: `tests/test_offline_pipeline_benchmark.py`

**Interfaces:**
- Produces: `normalize_query_identity(query: Any) -> str`.
- Produces: `query_fingerprint(query: Any) -> str`, hashing only the normalized textual identity.
- Consumes: `presentation_evidence_excerpt()` from Task 2.
- Preserves: current RAG limits and deterministic duplicate guard.

- [ ] **Step 1: Write failing single-authority query tests**

```python
def test_query_identity_is_textual_only():
    assert normalize_query_identity("  聚乙烯   质量异常 ") == "聚乙烯 质量异常"
    assert normalize_query_identity("聚乙烯质量问题") != normalize_query_identity(
        "聚乙烯 质量异常"
    )
```

Add source/AST assertions that prefetch and tool loop both call this helper and
tests do not implement their own lowercase/whitespace normalization.

- [ ] **Step 2: Write failing Worker integration scenarios**

Add exact Scenario A/B/Q2 tests:

- Sufficient prefetched evidence plus first fake final content yields one Worker
  LLM call, zero adaptive tool calls, zero duplicates, and one iteration.
- A same-normalized-fingerprint Q1 tool request is rejected and never invokes the
  knowledge tool.
- A textually distinct Q2 invokes the knowledge tool once and is recorded as an
  adaptive query.

Capture the first messages and assert they expose prefetched queries,
fingerprints, bounded evidence inventory, `prefetch_queries_used`, and
`adaptive_queries_remaining`.

- [ ] **Step 3: Run the Worker tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_worker_tool_boundaries.py -k "prefetch or duplicate or adaptive or first"
```

Expected: missing shared query helper and missing structured inventory fields.

- [ ] **Step 4: Implement the pure identity helper and replace both local normalizers**

Move current casefold/trim/whitespace behavior into
`normalize_query_identity()`. Do not add similarity, embeddings, synonym maps,
or network access. Use `query_fingerprint()` for inventory/log identity only.

- [ ] **Step 5: Project prefetch as completed retrieval inventory**

Update `_evidence_context_for_generation()` to serialize only:

```python
{
    "prefetched_queries": ["聚乙烯 质量异常 排查建议"],
    "query_fingerprints": ["8f04d3a18a21"],
    "evidence": [{
        "evidence_id": "E1",
        "title": "聚乙烯生产工艺与质量控制概述",
        "locator": "§5 关键工艺参数及其影响",
        "supporting_text_excerpt": "参数趋势、实验室分析与批次追踪可用于异常诊断。",
    }],
    "prefetch_queries_used": 1,
    "adaptive_queries_remaining": 2,
}
```

The surrounding instruction says retrieval is complete, sufficient evidence
should produce final content immediately, and adaptive retrieval is permitted
only for a concrete new gap. Keep the existing ToolMessage duplicate rejection.

- [ ] **Step 6: Verify Worker and benchmark behavior**

```bash
.venv/bin/python -m pytest -q tests/test_worker_tool_boundaries.py tests/test_report_evidence_integration.py tests/test_offline_pipeline_benchmark.py
```

Expected: all pass; Scenario B shows one fewer Worker generation/tool-loop
iteration and no actual duplicate retrieval.

- [ ] **Step 7: Commit Worker prefetch reuse**

```bash
git add src/evidence/query_identity.py src/nodes/worker/agent/graph.py src/prompts/worker_system_template.md tests/test_worker_tool_boundaries.py tests/test_offline_pipeline_benchmark.py
git commit -m "worker: expose completed RAG prefetch"
```

---

### Task 6: Compress Intake and Planner prompts without changing contracts

**Files:**
- Modify: `src/nodes/intake.py`
- Modify: `src/prompts/intake_to_planner.md`
- Modify: `src/prompts/planner_to_worker.md`
- Modify: `src/prompts/planner_replan.md`
- Modify: `src/prompts/planner_intake_replan.md`
- Modify: `tests/test_intake_helpers.py`
- Modify: `tests/test_planner_contract.py`
- Modify: `tests/test_offline_pipeline_benchmark.py`
- Modify: `docs/benchmarks/2026-08-28-pipeline-baseline.json`

**Interfaces:**
- Preserves: canonical Intake JSON fields, 14-field Planner task contract, validators, web authorization, requirements, and two-attempt plan repair.
- Produces: smaller serialized Intake/Planner messages for identical fixed fixtures.

- [ ] **Step 1: Add failing contract-preserving prompt-size tests**

Capture current formatted messages for fixed Intake and Planner fixtures. Assert
the optimized version must remain below the frozen baseline character counts
while retaining every contract field and these semantic anchors:

- stable `requirement_ids` only;
- `depends_on_task_ids` is execution dependency only;
- RAG means new evidence retrieval;
- synthesis cannot add facts;
- web requires explicit authorization;
- CSV-only quantitative analysis;
- causal is the only concept-graph kind.

- [ ] **Step 2: Run tests and confirm RED on size limits**

```bash
.venv/bin/python -m pytest -q tests/test_intake_helpers.py tests/test_planner_contract.py tests/test_offline_pipeline_benchmark.py
```

Expected: prompt sizes equal the baseline and violate the optimized thresholds.

- [ ] **Step 3: Remove duplicated Intake prompt material**

Keep the schema and rules once. Remove the long task/chat examples and the
duplicated “Current Task / 请生成 JSON” tail. Format the prompt as system
instructions plus one human message containing only `raw_request`. Do not change
`_normalize_canonical_intake()` or its required/default fields.

- [ ] **Step 4: Remove duplicated Planner prose while preserving anchors**

Delete redundant output examples and repeated explanations already enforced by
`_validate_generated_task_schema()` and `_validate_generated_task_semantics()`.
Do not remove contract fields, evidence-gap semantics, source authorization,
dependency rules, or validation/repair messages. Apply the same concise wording
to initial, full-replan, and intake-replan prompts rather than letting them drift.

- [ ] **Step 5: Run Planner/Intake and checkpoint tests**

```bash
.venv/bin/python -m pytest -q tests/test_intake_helpers.py tests/test_planner_helpers.py tests/test_planner_contract.py tests/test_graph_state_contract.py tests/test_recovery_compatibility.py tests/test_offline_pipeline_benchmark.py
.venv/bin/python -m pytest -q
```

Expected: all pass; prompt characters fall below the baseline without contract
or retry changes.

- [ ] **Step 6: Record optimized offline measurements**

Add an `optimized` object beside the immutable `baseline` object in the benchmark
artifact. Include call counts, Worker generations, tool-loop iterations,
duplicate retrievals, serialized prompt chars, mock completion chars, and local
execution time labelled offline-only. Keep provider-token and online-latency
optimized values `null` with `requires_real_run=true`.

- [ ] **Step 7: Commit prompt compression and benchmark results**

```bash
git add src/nodes/intake.py src/prompts/intake_to_planner.md src/prompts/planner_to_worker.md src/prompts/planner_replan.md src/prompts/planner_intake_replan.md tests/test_intake_helpers.py tests/test_planner_contract.py tests/test_offline_pipeline_benchmark.py docs/benchmarks/2026-08-28-pipeline-baseline.json
git commit -m "planner: reduce redundant generation context"
```

---

### Task 7: Render a grouped, compact evidence appendix

**Files:**
- Modify: `src/evidence/reporting.py`
- Modify: `src/nodes/summarizer_v2.py`
- Modify: `tests/test_summarizer_deterministic.py`
- Modify: `tests/test_report_renderers.py`

**Interfaces:**
- Produces: `format_grouped_evidence_appendix(citations, *, heading_level, heading_title) -> str`.
- Consumes: `presentation_evidence_excerpt()` and existing `canonical_source_identity()`.
- Preserves: full citation dictionaries, final display IDs, locators,正文 citation markers, and current PDF/DOCX renderers.

- [ ] **Step 1: Write failing grouped-appendix regressions**

Cover:

- E1/E2/E3 from one document produce one source group.
- Source groups retain first-appearance order and evidence uses natural ID order.
- Safe title/filename is displayed instead of absolute paths.
- Section/page/table locators remain visible; chunk IDs and cache paths do not.
- Repeated `|` noise is removed and summaries are bounded.
- Two consecutive renders are byte-identical.
- Input citations are deep-equal before and after rendering.
- Same-basename behavior matches the actual existing
  `canonical_source_identity()` contract and is not overridden by another
  basename grouping rule.

- [ ] **Step 2: Run report tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_summarizer_deterministic.py tests/test_report_renderers.py
```

Expected: current flat evidence table does not satisfy grouped layout assertions.

- [ ] **Step 3: Implement stable grouped projection**

Group by canonical identity in insertion order, retain a separate safe display
label, and sort evidence with a natural key extracted from `E<number>`. Render a
small per-source table containing evidence ID, locator, supporting section, and
`presentation_evidence_excerpt()`. Never write sanitized text back into citation
objects.

- [ ] **Step 4: Use grouped projection only in the final reference appendix**

Update `_assemble_markdown()` to call the grouped formatter for the evidence
index/reference section. Keep the file-level summary when explicitly requested
and keep existing Markdown-to-PDF/DOCX calls unchanged.

- [ ] **Step 5: Run report, citation, and admission tests**

```bash
.venv/bin/python -m pytest -q tests/test_summarizer_deterministic.py tests/test_report_renderers.py tests/test_report_evidence_integration.py tests/test_report_acceptance.py tests/test_synthesis.py
```

Expected: all pass.

- [ ] **Step 6: Commit the grouped appendix**

```bash
git add src/evidence/reporting.py src/nodes/summarizer_v2.py tests/test_summarizer_deterministic.py tests/test_report_renderers.py
git commit -m "report: group and compact evidence appendix"
```

---

### Task 8: Audit initialization noise without unsafe caching

**Files:**
- Create: `docs/benchmarks/2026-08-28-initialization-audit.md`
- Modify only if proven safe: immutable process-global registration code identified by the audit.
- Test only if code changes: the focused test file owning that registration behavior.

**Interfaces:**
- Produces: an evidence-backed audit conclusion.
- Must not cache job paths, task state, RAG results, mutable registries, or LLM clients.

- [ ] **Step 1: Trace repeated initialization and warnings**

Run:

```bash
rg -n "makedirs|mkdir|register|registry|pkg_resources|jieba" src tests
.venv/bin/python -W default -c "import jieba"
```

Record the exact source, scope, mutability, frequency, and safety assessment for
each candidate. Do not suppress warnings globally.

- [ ] **Step 2: Apply the exit criterion**

If no object is proven immutable and process-global, make no production change.
Record that repeated idempotent directory creation with `exist_ok=True` is intentionally retained over
cross-job caching. If one registration is proven safe, first add a failing test
showing repeated construction performs duplicate registration, then implement
idempotence and rerun that focused test.

- [ ] **Step 3: Commit the audit conclusion**

```bash
git add docs/benchmarks/2026-08-28-initialization-audit.md
git commit -m "chore: document initialization noise audit"
```

If a tested production fix exists, include only its exact source and test files
in the same commit.

---

### Task 9: Final verification and diff review

**Files:**
- Verify every changed file; do not add unrelated cleanup.

**Interfaces:**
- Produces: final evidence for correctness, compatibility, performance claims, and commit history.

- [ ] **Step 1: Run focused compatibility suites**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_failure_decision_policy.py tests/test_recovery_policy.py tests/test_recovery_compatibility.py tests/test_graph_state_contract.py tests/test_job_store.py
.venv/bin/python -m pytest -q tests/test_worker_tool_boundaries.py tests/test_llm_observability.py tests/test_offline_pipeline_benchmark.py
.venv/bin/python -m pytest -q tests/test_summarizer_deterministic.py tests/test_report_renderers.py tests/test_report_evidence_integration.py tests/test_report_acceptance.py
```

Expected: all pass without network access.

- [ ] **Step 2: Run the complete suite and compilation**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src app.py run.py
git diff --check
```

Expected: zero failures and zero compile/diff errors.

- [ ] **Step 3: Audit forbidden scope changes**

```bash
git diff 3ba9fd3eb3ad84b193f699e72e15bc40bea40446 -- src/state.py src/persistence.py src/graph.py src/recovery src/job_store.py
rg -n "invoke_llm\(" src/nodes/verifier.py
git status --short
git log --oneline 3ba9fd3eb3ad84b193f699e72e15bc40bea40446..HEAD
git diff --stat 3ba9fd3eb3ad84b193f699e72e15bc40bea40446..HEAD
```

Expected: no State/persistence/graph/Recovery/JobStore changes; the Verifier has
the same semantic-call count; only the unrelated pre-existing zip remains
untracked.

- [ ] **Step 4: Review fail-closed and compatibility invariants**

Confirm from code and tests:

- deterministic citation preflight still prevents the LLM call;
- semantic codes survive sanitize and use existing evidence-gap Recovery;
- old checkpoints require no new fields;
- Worker query identity is textual only and Q2 remains available;
- full provenance is unmodified;
- benchmark claims do not present mock characters or local time as provider
  tokens or online latency.

- [ ] **Step 5: Prepare the final report**

Report changed files and reasons, added tests, exact pytest counts, State/resume
compatibility, claim classification behavior, duplicate-RAG behavior, the
baseline/optimized deterministic benchmark table, online metrics marked “not
re-measured online / requires real-run validation,” remaining risks, diff stat,
and the complete commit list.

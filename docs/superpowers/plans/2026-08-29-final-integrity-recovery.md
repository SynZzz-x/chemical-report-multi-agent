# Final Integrity and Recovery Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make final report citations globally unambiguous, prevent mixed semantic failures from degrading into warning delivery, reject conservatively detected uncited material assertions, prove and reduce Verifier request cost, and make the evidence appendix readable.

**Architecture:** Preserve State, checkpoints, graph topology, model-call count, and existing Recovery actions. Add invocation-local lossless citation validation around the existing remap, derive one order-independent recovery profile from the complete assessment, extend deterministic citation-required checks without judging evidence support, and prove DeepSeek request serialization offline before changing provider controls.

**Tech Stack:** Python 3.14, Pydantic v2, LangChain/LangGraph, `langchain-openai==1.0.0`, `openai==2.54.0`, pytest, httpx MockTransport, and existing Markdown/PDF/DOCX renderers.

## Global Constraints

- Production baseline is `cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb`; commits `ccf0b7d` and `eb8978c` are design documentation only.
- Do not modify State/checkpoint schemas, persistence formats, graph topology, action enums, or Recovery branches.
- Do not add an LLM call or a lexical/embedding evidence-support gate.
- Keep the semantic AutoVerifier at exactly one call after deterministic preflight.
- Keep deterministic citation checks fail-closed and run them before any report file or renderer call.
- `COMMIT_WITH_WARNING` is allowed only when every unresolved issue is degradable and existing terminal policy independently permits warning delivery.
- `CLAIM_UNSUPPORTED`, `CLAIM_PARTIALLY_SUPPORTED`, `CLAIM_EVIDENCE_MISMATCH`, `UNLABELED_INFERENCE`, and `UNCITED_MATERIAL_CLAIM` are non-degradable.
- Do not use basename-only identity or presentation excerpts as correctness identity.
- Do not mutate raw citations or provenance for presentation.
- Do not access DeepSeek, embedding services, or public network in tests.
- Do not claim provider-token or online-latency improvement from character benchmarks.
- Use red-green TDD for every production behavior change and preserve the untracked `agent-master-clean-20260828.zip`.

## File and interface map

- `src/evidence/identity.py`: canonical citation identity and compatible remap.
- `src/evidence/normalizer.py`: existing evidence dedupe delegates to that identity.
- `src/evidence/integrity.py`: pure pre-remap/final validation and lossless registry.
- `src/nodes/summarizer_v2.py`: final gate before paths, writes, and renderers.
- `src/recovery/policy.py`: assessment profile, degradation guard, and existing-action state machine.
- `src/evidence/claims.py`, `src/nodes/verifier.py`: uncited gate and deduplicated semantic payload.
- `src/config.py`, `src/llm.py`: proven provider request mapping and optional Verifier controls.
- `src/evidence/text_projection.py`, `src/evidence/reporting.py`: presentation-only appendix changes.

---

### Task 1: Establish one lossless citation identity and protect pre-remap input

**Files:**
- Modify: `src/evidence/identity.py:1-157`
- Modify: `src/evidence/normalizer.py:1-41`
- Create: `src/evidence/integrity.py`
- Create: `tests/test_citation_integrity.py`
- Modify: `tests/test_report_evidence_integration.py:115-166`
- Test: `tests/test_evidence_pipeline.py`
- Test: `tests/test_synthesis.py`

**Interfaces:**
- Produces: `canonical_citation_identity(citation: Mapping[str, Any]) -> str`.
- Produces: `CitationIntegrityIssue`, `CitationIntegrityValidation`, and `validate_pre_remap_citation_integrity(sections)`.
- Preserves: `build_display_evidence_map()` and `normalize_sections_evidence()` signatures.
- Consumed by: Task 2 final registry validation.

- [ ] **Step 1: Verify production files still match the baseline**

Run:

```bash
git diff --exit-code cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb -- app.py run.py src tests
```

Expected: exit 0. Only approved design documentation differs.

- [ ] **Step 2: Write RED identity-authority tests**

Create `tests/test_citation_integrity.py`:

```python
from copy import deepcopy

from src.evidence.identity import canonical_citation_identity
from src.evidence.integrity import validate_pre_remap_citation_integrity


def citation(path: str, *, evidence_id: str = "E8", file_id: str = "") -> dict:
    return {
        "evidence_id": evidence_id,
        "source_type": "rag",
        "file_id": file_id,
        "file_path": path,
        "locator": "section 5",
        "chunk_ids": ["chunk-5"],
        "supporting_text": "温度影响分子量。",
        "supporting_text_excerpt": "展示摘要",
    }


def test_identity_prefers_stable_source_id_over_cache_path():
    left = citation("/job-a/cache/process.docx", file_id="doc-42")
    right = citation("/job-b/cache/process.docx", file_id="doc-42")
    assert canonical_citation_identity(left) == canonical_citation_identity(right)


def test_identity_does_not_merge_distinct_same_basename_files():
    assert canonical_citation_identity(citation("/a/process.docx")) != canonical_citation_identity(citation("/b/process.docx"))


def test_identity_ignores_presentation_excerpt():
    left = citation("/a/process.docx")
    right = deepcopy(left)
    right["supporting_text_excerpt"] = "另一展示摘要"
    assert canonical_citation_identity(left) == canonical_citation_identity(right)


def test_identity_uses_full_raw_text_as_last_evidence_fallback():
    left = citation("/a/process.docx")
    right = deepcopy(left)
    left["chunk_ids"] = []
    right["chunk_ids"] = []
    right["supporting_text"] = "温度不影响分子量。"
    assert canonical_citation_identity(left) != canonical_citation_identity(right)
```

Add:

```python
def test_identity_canonicalizes_url_host_and_fragment_only():
    left = {**citation(""), "source_type": "web", "file_path": "", "url": "HTTPS://Example.COM/report?id=7#page-1"}
    right = {**left, "url": "https://example.com/report?id=7#page-9"}
    other = {**left, "url": "https://example.com/other?id=7"}
    assert canonical_citation_identity(left) == canonical_citation_identity(right)
    assert canonical_citation_identity(left) != canonical_citation_identity(other)
```

- [ ] **Step 3: Write RED pre-remap conflict tests**

Add:

```python
def test_pre_remap_rejects_same_task_local_id_with_two_identities():
    sections = [{
        "task_id": "T1",
        "text_output": "工艺结论 [E8]。",
        "citations": [
            citation("/docs/polyethylene-process.docx"),
            {**citation("/docs/heat-exchanger-guide.docx"), "locator": "section 2.1", "chunk_ids": ["maintenance-2"], "supporting_text": "换热器需要清洗。"},
        ],
    }]
    result = validate_pre_remap_citation_integrity(sections)
    assert result.is_valid is False
    assert result.issues[0].code == "LOCAL_CITATION_IDENTITY_CONFLICT"
    assert result.issues[0].task_id == "T1"
    assert result.issues[0].evidence_id == "E8"


def test_pre_remap_allows_exact_duplicate_records():
    item = citation("/docs/process.docx")
    result = validate_pre_remap_citation_integrity([{"task_id": "T1", "text_output": "工艺 [E8]。", "citations": [item, deepcopy(item)]}])
    assert result.is_valid is True


def test_pre_remap_scopes_same_local_id_by_task():
    sections = [
        {"task_id": "T1", "text_output": "工艺 [E8]。", "citations": [citation("/docs/process.docx")]},
        {"task_id": "T2", "text_output": "维护 [E8]。", "citations": [citation("/docs/maintenance.docx")]},
    ]
    assert validate_pre_remap_citation_integrity(sections).is_valid is True
```

Extend the existing cross-task remap test to assert both raw entries survive and `T1:E8`/`T2:E8` receive different final IDs.

- [ ] **Step 4: Run and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_citation_integrity.py tests/test_report_evidence_integration.py
```

Expected: collection fails because the new identity and integrity module do not exist.

- [ ] **Step 5: Implement the identity authority**

In `src/evidence/identity.py`, build one SHA-256 identity from stable source ID, else canonical URL, else a sufficiently strong explicit canonical source field, else normalized full path, else title fallback; then locator and chunk IDs, or full raw supporting-text hash when chunk IDs are absent. Use fields `file_id`, `resource_id`, `source_id`, `document_id`, and `doc_id` as stable source IDs. A basename returned by presentation-only `canonical_source_identity()` is never sufficiently strong while a full path exists. Exclude display/local evidence IDs and `supporting_text_excerpt`.

```python
def canonical_citation_identity(citation: Mapping[str, Any]) -> str:
    stable_id = next((str(citation.get(field) or "").strip() for field in _STABLE_SOURCE_ID_FIELDS if str(citation.get(field) or "").strip()), "")
    canonical_url = _canonical_url(citation.get("url"))
    full_path = _normalized_source_path(citation.get("file_path"))
    explicit_canonical = str(citation.get("canonical_source_id") or citation.get("source_identity") or "").strip().casefold()
    title_fallback = str(citation.get("title") or "").strip().casefold()
    authority = f"stable:{stable_id}" if stable_id else f"url:{canonical_url}" if canonical_url else f"canonical:{explicit_canonical}" if explicit_canonical else f"path:{full_path}" if full_path else f"fallback:{title_fallback}"
    chunk_ids = tuple(str(value) for value in citation.get("chunk_ids") or ())
    evidence_identity = {"chunk_ids": chunk_ids} if chunk_ids else {"supporting_text_sha256": hashlib.sha256(str(citation.get("supporting_text") or "").encode("utf-8")).hexdigest()}
    payload = {"source_type": str(citation.get("source_type") or "").casefold(), "authority": authority, "locator": str(citation.get("locator") or "").strip(), **evidence_identity}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
```

Make `src/evidence/normalizer.py::_dedupe_key()` call `canonical_citation_identity(record.model_dump(mode="json"))` so the repository has one correctness authority.

- [ ] **Step 6: Implement frozen validation results**

Create `src/evidence/integrity.py`:

```python
from dataclasses import asdict, dataclass
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .identity import canonical_citation_identity


@dataclass(frozen=True)
class CitationIntegrityIssue:
    code: str
    description: str
    task_id: str = ""
    evidence_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CitationIntegrityValidation:
    issues: tuple[CitationIntegrityIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues
```

`validate_pre_remap_citation_integrity()` groups every raw list entry by `(task_id, local_evidence_id)` and fails when its canonical identity set contains more than one value. Refactor `build_display_evidence_map()` to iterate validated raw records in stable list order; remove its `citations_by_id` overwrite.

- [ ] **Step 7: Run GREEN and compatibility tests**

```bash
.venv/bin/python -m pytest -q tests/test_citation_integrity.py tests/test_report_evidence_integration.py tests/test_evidence_pipeline.py tests/test_synthesis.py
python -m compileall -q src/evidence
git diff --check
```

Expected: all selected tests pass; compileall and diff check are silent.

- [ ] **Step 8: Commit**

```bash
git add src/evidence/identity.py src/evidence/integrity.py src/evidence/normalizer.py tests/test_citation_integrity.py tests/test_report_evidence_integration.py
git commit -m "evidence: preserve lossless citation identity"
```

---

### Task 2: Add the final citation gate before every report artifact

**Files:**
- Modify: `src/evidence/integrity.py`
- Modify: `src/nodes/summarizer_v2.py:584-815`
- Modify: `tests/test_citation_integrity.py`
- Modify: `tests/test_summarizer_deterministic.py`
- Test: `tests/test_report_renderers.py`

**Interfaces:**
- Consumes: Task 1 canonical identity and validation types.
- Produces: `project_lossless_used_citations(sections) -> list[dict[str, Any]]`.
- Produces: `validate_final_citation_integrity(normalized_sections, final_markdown, lossless_final_citations) -> CitationIntegrityValidation`.
- Preserves: Summarizer result schema and renderer inputs.

- [ ] **Step 1: Write RED final-registry tests**

Add to `tests/test_citation_integrity.py`:

```python
from src.evidence.integrity import project_lossless_used_citations, validate_final_citation_integrity


def test_final_gate_rejects_one_display_id_with_two_identities():
    sections = [
        {"task_id": "T1", "text_output": "工艺 [E1]。", "citations": [citation("/docs/process.docx", evidence_id="E1")]},
        {"task_id": "T2", "text_output": "维护 [E1]。", "citations": [citation("/docs/maintenance.docx", evidence_id="E1")]},
    ]
    registry = [item for section in sections for item in section["citations"]]
    result = validate_final_citation_integrity(sections, "工艺 [E1]。\n维护 [E1]。", registry)
    assert result.is_valid is False
    assert {issue.code for issue in result.issues} == {"FINAL_DISPLAY_IDENTITY_CONFLICT"}


def test_final_gate_rejects_unbound_body_marker():
    sections = [{"task_id": "T1", "text_output": "工艺 [E9]。", "citations": []}]
    result = validate_final_citation_integrity(sections, "工艺 [E9]。", [])
    assert result.is_valid is False
    assert "FINAL_CITATION_BINDING_MISSING" in {issue.code for issue in result.issues}


def test_final_gate_allows_exact_duplicate_resolution():
    item = citation("/docs/process.docx", evidence_id="E1")
    sections = [{"task_id": "T1", "text_output": "工艺 [E1]。", "citations": [item, deepcopy(item)]}]
    result = validate_final_citation_integrity(sections, "工艺 [E1]。\n\n[E1]", [item, deepcopy(item)])
    assert result.is_valid is True
```

Add these named assertions:

```python
def test_appendix_marker_cannot_mask_missing_body_binding():
    sections = [{"task_id": "T1", "text_output": "正文 [E9]。", "citations": []}]
    result = validate_final_citation_integrity(sections, "正文 [E9]。\n\n证据附录 [E9]", [])
    assert "FINAL_CITATION_BINDING_MISSING" in {issue.code for issue in result.issues}


def test_lossless_projection_retains_conflicting_entries():
    left = citation("/docs/a.docx", evidence_id="E1")
    right = citation("/docs/b.docx", evidence_id="E1")
    projected = project_lossless_used_citations([{"task_id": "T1", "text_output": "正文 [E1]。", "citations": [left, right]}])
    assert projected == [left, right]
```

Name the remaining cases `test_final_gate_rejects_registry_id_unused_by_body`, `test_final_gate_rejects_task_local_alias_after_remap`, and `test_final_gate_accepts_grouped_markers`; assert their issue codes or `is_valid` explicitly.

- [ ] **Step 2: Write RED Summarizer fail-closed tests**

Add to `tests/test_summarizer_deterministic.py`:

```python
def test_final_citation_conflict_blocks_before_delivery_paths(monkeypatch):
    state = _state(
        statuses={"T1": _status("VERIFIED_PASS"), "T2": _status("VERIFIED_PASS")},
        results=[{
            "task_id": "T1",
            "text_output": "冲突正文 [E1]。",
            "plan_revision": 1,
            "task_revision": 1,
            "citations": [
                {"evidence_id": "E1", "file_path": "/docs/a.docx", "locator": "1", "supporting_text": "甲"},
                {"evidence_id": "E1", "file_path": "/docs/b.docx", "locator": "2", "supporting_text": "乙"},
            ],
        }, {
            "task_id": "T2",
            "text_output": "正常正文 [E2]。",
            "plan_revision": 1,
            "task_revision": 1,
            "citations": [{"evidence_id": "E2", "file_path": "/docs/c.docx", "locator": "3", "supporting_text": "丙"}],
        }],
    )
    monkeypatch.setattr(summarizer_v2, "get_session_cache_dir", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("gate must precede path resolution")))
    update = summarizer_v2.summarizer(state, {})
    assert update["report_status"] == "BLOCKED"
    assert update["final_result"]["attachments"] == []
    assert update["final_result"]["blocking_sections"][0]["issues"][0]["code"] == "FINAL_CITATION_INTEGRITY"


def test_markdown_assembly_is_byte_deterministic():
    state = _state(statuses={"T1": _status("VERIFIED_PASS"), "T2": _status("VERIFIED_PASS")})
    sections, _ = summarizer_v2.normalize_sections_evidence(state["results"])
    first = summarizer_v2._assemble_markdown(state, sections, "READY_FOR_FINAL")
    second = summarizer_v2._assemble_markdown(state, sections, "READY_FOR_FINAL")
    assert first == second
```

The determinism test calls assembly twice; production must not.

- [ ] **Step 3: Run and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_citation_integrity.py tests/test_summarizer_deterministic.py -k "citation or markdown_assembly"
```

Expected: the final projection/validator is absent and Summarizer still reaches path creation.

- [ ] **Step 4: Implement lossless projection and final validation**

`project_lossless_used_citations()` must scan every citation list entry whose ID occurs in the normalized body; it must not construct `dict[evidence_id]`. The validator separately derives body IDs, registry IDs, and final Markdown IDs; it groups registry entries into `display_id -> set[canonical_identity]`, permits one canonical identity with exact duplicates, and emits stable sorted issues for conflicts, missing bindings, unused registry bindings, and remap aliases.

```python
def project_lossless_used_citations(sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for section in sections:
        body = str(section.get("text") or section.get("content") or section.get("text_output") or "")
        used_ids = set(extract_inline_evidence_ids(body))
        for citation in section.get("citations") or ():
            if not isinstance(citation, Mapping):
                continue
            evidence_id = str(citation.get("evidence_id") or "").strip().upper()
            if evidence_id in used_ids:
                projected.append(dict(citation))
    return projected
```

Do not source the gate from `_deduplicate_citations()` or `project_used_citations()` because both are lossy by display ID.

- [ ] **Step 5: Integrate before path resolution**

Use this order in `summarizer_v2.summarizer()`:

```python
preflight = validate_pre_remap_citation_integrity(sections)
if not preflight.is_valid:
    return _citation_integrity_blocked_update(preflight)

sections, evidence_display_map = normalize_sections_evidence(sections)
final_markdown = _assemble_markdown(state, sections, report_status)
final_citations = project_lossless_used_citations(sections)
final_validation = validate_final_citation_integrity(sections, final_markdown, final_citations)
if not final_validation.is_valid:
    return _citation_integrity_blocked_update(final_validation)

report_dir = os.path.join(get_session_cache_dir(state, config), "report")
```

The local adapter converts validation issues into one report-level `_blocked_update()` item with code `FINAL_CITATION_INTEGRITY`. It adds no State field or graph route.

- [ ] **Step 6: Run the complete P0-1 checkpoint**

```bash
.venv/bin/python -m pytest -q tests/test_citation_integrity.py tests/test_report_evidence_integration.py tests/test_summarizer_deterministic.py tests/test_synthesis.py tests/test_evidence_pipeline.py tests/test_report_renderers.py
python -m compileall -q src tests
git diff --check
```

Expected: all selected tests pass; compileall and diff check are silent.

- [ ] **Step 7: Commit**

```bash
git add src/evidence/integrity.py src/nodes/summarizer_v2.py tests/test_citation_integrity.py tests/test_summarizer_deterministic.py
git commit -m "report: validate final citation identity"
```

---

### Task 3: Aggregate unresolved recovery issues without warning hitchhiking

**Files:**
- Modify: `src/recovery/policy.py:48-1657`
- Modify: `tests/test_failure_decision_policy.py`
- Modify: `tests/test_recovery_policy.py`
- Test: `tests/test_recovery_compatibility.py`
- Test: `tests/test_recovery_graph.py`
- Test: `tests/test_graph_state_contract.py`

**Interfaces:**
- Produces: private `_AssessmentPolicyProfile` and `_profile_assessment(assessment, state)`.
- Preserves: `decide_recovery_action()`, counters, action enums, blocker schema, and routes.
- Consumed by: Task 4 relies on `UNCITED_MATERIAL_CLAIM` being non-degradable.

- [ ] **Step 1: Write RED mixed-assessment tests**

Add to `tests/test_failure_decision_policy.py`:

```python
from itertools import permutations


def mixed_assessment() -> dict:
    return {"status": "FAILED", "issues": [
        {"code": "EVIDENCE_GAP", "category": "EVIDENCE_GAP", "description": "缺少次要来源", "suggestion": "保留缺口", "severity": "minor", "requirement_ids": []},
        {"code": "CLAIM_PARTIALLY_SUPPORTED", "category": "EVIDENCE_GAP", "description": "优先级没有证据", "suggestion": "收缩论断", "severity": "major", "requirement_ids": [], "affected_claims": [{"claim_id": "C2", "text": "氢气是第一优先项"}]},
    ]}


def test_mixed_gap_and_semantic_issue_never_commits_warning():
    state = _state(code="EVIDENCE_GAP", requirement_severity="soft")
    state["evidence_recovery_count"] = {"T1": 1}
    update = decide_recovery_action(state, mixed_assessment())
    assert update["workflow_action"] == "REWORK"
    assert update.get("results", []) == []
    assert update["failure_decision"]["action"] == "RETRY_TASK"


def test_multi_issue_policy_is_order_independent():
    outcomes = []
    for order in permutations(mixed_assessment()["issues"]):
        state = _state(code="EVIDENCE_GAP", requirement_severity="soft")
        state["evidence_recovery_count"] = {"T1": 1}
        update = decide_recovery_action(state, {"status": "FAILED", "issues": list(order)})
        outcomes.append((update["workflow_action"], update["failure_decision"]["action"], update["failure_decision"]["subtype"], update["task_retry_count"]))
    assert len(set(map(repr, outcomes))) == 1
```

Add:

```python
def test_two_degradable_gaps_keep_existing_warning_delivery():
    state = _state(code="EVIDENCE_GAP", requirement_severity="soft")
    state["evidence_recovery_count"] = {"T1": 1}
    issues = [
        {"code": "EVIDENCE_GAP", "category": "EVIDENCE_GAP", "description": "gap one", "suggestion": "disclose", "severity": "minor", "requirement_ids": []},
        {"code": "MISSING_EVIDENCE", "category": "EVIDENCE_GAP", "description": "gap two", "suggestion": "disclose", "severity": "minor", "requirement_ids": []},
    ]
    update = decide_recovery_action(state, {"status": "FAILED", "issues": issues})
    assert update["failure_decision"]["action"] == "COMMIT_WITH_WARNING"
    assert update["results"]
    assert update["section_status"]["T1"]["status"] == "ACCEPT_WITH_WARNING"
```

- [ ] **Step 2: Write RED full-state-machine and logging tests**

Implement `test_semantic_issue_recovers_reworks_then_blocks_without_warning` by calling `decide_recovery_action()` four times. Copy `evidence_recovery_count` from call one into the state for call two, copy `task_retry_count` from calls two and three, and assert the exact action sequence:

```python
assert actions == ["EVIDENCE_RECOVERY", "REWORK", "REWORK", "NEEDS_USER_INPUT"]
assert final["failure_decision"]["action"] == "REGISTER_BLOCKER"
assert final.get("results", []) == []
assert {issue["code"] for issue in final["pending_user_blockers"][0]["issues"]} == {"EVIDENCE_GAP", "CLAIM_PARTIALLY_SUPPORTED"}
```

Use `caplog` to require `issue_count=2`, `selected_policy_issue_code=CLAIM_PARTIALLY_SUPPORTED`, `selected_policy_action=RETRY_TASK`, and `has_non_degradable_issue=true`; repeat with reversed issue order.

- [ ] **Step 3: Write RED defensive and uncited-code tests**

Add:

```python
def test_uncited_material_claim_is_non_degradable_after_rework_exhaustion():
    state = _state(code="UNCITED_MATERIAL_CLAIM", requirement_severity="soft")
    state["task_retry_count"] = {"T1": 2}
    assessment = {"status": "FAILED", "issues": [{"code": "UNCITED_MATERIAL_CLAIM", "category": "CONTENT_DEFECT", "description": "强断言缺少显式引用", "suggestion": "补引用或收缩断言", "severity": "major", "requirement_ids": []}]}
    update = decide_recovery_action(state, assessment)
    assert update["workflow_action"] == "NEEDS_USER_INPUT"
    assert update["failure_decision"]["action"] == "REGISTER_BLOCKER"
    assert update.get("results", []) == []
```

Directly test the shared warning gate with one semantic assessment and with two ordinary waivable gaps.

- [ ] **Step 4: Run and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_failure_decision_policy.py tests/test_recovery_policy.py
```

Expected: the mixed assessment degrades or selected subtype remains order dependent.

- [ ] **Step 5: Implement the pure profile**

Add:

```python
from dataclasses import dataclass

NON_DEGRADABLE_ISSUE_CODES = frozenset({"CLAIM_UNSUPPORTED", "CLAIM_PARTIALLY_SUPPORTED", "CLAIM_EVIDENCE_MISMATCH", "UNLABELED_INFERENCE", "UNCITED_MATERIAL_CLAIM"})


@dataclass(frozen=True)
class _AssessmentPolicyProfile:
    issue_count: int
    has_non_degradable_issue: bool
    all_unresolved_issues_degradable: bool
    selected_policy_issue_code: str
    selected_policy_category: IssueCategory
    selected_policy_tier: str
```

`_profile_assessment()` classifies every issue once and selects by verifier/fatal, blocking, non-degradable repair, ordinary repair/degradation, then next; use existing `_CATEGORY_PRIORITY` and normalized-code tie-break inside a tier. Keep the complete assessment unchanged and do not add semantic codes to `_EVIDENCE_CODES`.

- [ ] **Step 6: Implement the existing-action state machine and warning guard**

Preserve initial authorized `EVIDENCE_RECOVERY`. When evidence recovery is exhausted and semantic issues remain, consume `task_retry_count` with `REWORK`; after `MAX_CONTENT_RETRIES`, reuse blocker construction and `REGISTER_BLOCKER`. Synthesis remains on `SYNTHESIS_REWRITE`.

Add:

```python
def _can_commit_with_warning(profile: _AssessmentPolicyProfile) -> bool:
    return profile.issue_count > 0 and profile.all_unresolved_issues_degradable and not profile.has_non_degradable_issue
```

Use it before every `_commit_degraded_result()` call and defensively inside the helper. A rejected warning path returns existing rework/blocker behavior; it does not raise or create a route.

- [ ] **Step 7: Make decisions and logs order independent**

Pass the selected code into `_failure_decision()`/`_set_decision()` instead of using `_assessment_subtype()` first-issue semantics. Log the four approved fields; `selected_policy_action` is the canonical `FailureAction`.

- [ ] **Step 8: Run focused recovery GREEN tests**

```bash
.venv/bin/python -m pytest -q tests/test_failure_decision_policy.py tests/test_recovery_policy.py tests/test_recovery_compatibility.py tests/test_recovery_graph.py tests/test_graph_state_contract.py tests/test_degradation_registry.py tests/test_blocker_registry.py
python -m compileall -q src app.py run.py
git diff --check
```

Expected: focused suites pass; compileall and diff check are silent.

- [ ] **Step 9: Commit**

```bash
git add src/recovery/policy.py tests/test_failure_decision_policy.py tests/test_recovery_policy.py
git commit -m "recovery: aggregate unresolved issue severity"
```

- [ ] **Step 10: Run the mandatory post-P0 full checkpoint**

```bash
.venv/bin/python -m pytest -q
```

Expected: the full suite passes before any P1 work starts.

---

### Task 4: Reject conservatively detected uncited material assertions

**Files:**
- Modify: `src/evidence/claims.py:12-132`
- Modify: `src/nodes/verifier.py:42-855`
- Modify: `src/prompts/worker_system_template.md`
- Modify: `src/nodes/worker/agent/graph.py:2064-2120`
- Modify: `tests/test_claim_grounding.py`
- Modify: `tests/test_auto_verifier_routing.py`
- Test: `tests/test_worker_tool_boundaries.py`

**Interfaces:**
- Produces: `find_uncited_material_claims(content: str) -> list[dict[str, str]]`.
- Produces: deterministic `UNCITED_MATERIAL_CLAIM / CONTENT_DEFECT` before semantic invocation.
- Preserves: cited claim derivation, semantic authority, assessment schema, and one-call behavior.

- [ ] **Step 1: Write RED detector tests**

Add to `tests/test_claim_grounding.py`:

```python
from src.evidence.claims import find_uncited_material_claims


@pytest.mark.parametrize("text", [
    "氢气/乙烯比是控制MFR档位最直接、最灵敏的调节手段。",
    "共聚单体/乙烯比是密度档位的核心控制变量。",
])
def test_uncited_material_factual_assertion_is_detected(text):
    assert find_uncited_material_claims(text) == [{"text": text, "claim_type": "factual"}]


@pytest.mark.parametrize("text", [
    "建议优先检查氢气比。",
    "现有证据不足，因此不作判断。",
    "### 核心控制变量",
    "| 核心 | 变量 |",
    "必须",
])
def test_non_factual_or_non_prose_material_words_do_not_trigger(text):
    assert find_uncited_material_claims(text) == []


def test_material_inference_requires_premise_citation():
    assert find_uncited_material_claims("据此可推测反应压力升高会显著扩大MWD。") == [{"text": "据此可推测反应压力升高会显著扩大MWD。", "claim_type": "inference"}]
```

Add:

```python
def test_material_sentence_does_not_inherit_previous_citation():
    text = "氢气影响熔指。[E1] 共聚单体/乙烯比是密度档位的核心控制变量。"
    assert find_uncited_material_claims(text) == [{"text": "共聚单体/乙烯比是密度档位的核心控制变量。", "claim_type": "factual"}]
```

- [ ] **Step 2: Write RED Verifier fast-fail tests**

Add `test_uncited_material_preflight_skips_semantic_model` and `test_cited_material_claim_reaches_one_semantic_call`. The first records `before = len(fake_verifier.calls)` and asserts the count is unchanged after FAILED preflight; the second supplies E1 supporting text and asserts the count increases by one. Add `test_uncited_issue_filters_invalid_requirement_ids` using one active and one withdrawn requirement and assert only the active ID remains.

- [ ] **Step 3: Run and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_auto_verifier_routing.py
```

Expected: the detector is absent and uncited material statements do not fast-fail.

- [ ] **Step 4: Implement the conservative detector**

Reuse `_statements()`, `evidence_gap_wording()`, `recommendation_wording()`, and `inference_wording()`. Existing valid markers short-circuit. Skip Markdown headings, fenced code, table rows, labels, recommendations, and evidence gaps. Require a high-precision material construction plus assertive predicate. Explicit inference is returned only when material and uncited.

```python
def find_uncited_material_claims(content: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for statement in _statements(content):
        if extract_inline_evidence_ids(statement):
            continue
        if _non_prose_statement(statement) or evidence_gap_wording(statement):
            continue
        claim_type = "inference" if inference_wording(statement) else "factual"
        if claim_type == "factual" and recommendation_wording(statement):
            continue
        if _requires_material_citation(statement):
            findings.append({"text": semantic_evidence_excerpt(statement, limit=_CLAIM_TEXT_LIMIT), "claim_type": claim_type})
    return findings
```

Patterns cover the approved high-confidence phrases and quantitative number-plus-unit assertions. They determine citation requirement only, never support.

- [ ] **Step 5: Integrate after citation preflight and before claim derivation**

Add `UNCITED_MATERIAL_CLAIM` to `_CATEGORY_BY_CODE` as `CONTENT_DEFECT` and to `_REQUIREMENT_KINDS_BY_CODE` as citation/evidence. Extend `_deterministic_preflight()` after `_apply_citation_integrity()` to return the existing deterministic FAILED assessment with suggestion to add an adjacent validated citation or narrow/remove the assertion. Reuse the current sanitizer and do not modify `VerifierAssessment`.

```python
uncited = find_uncited_material_claims(content)
if uncited:
    return _failed_preflight_assessment(
        code="UNCITED_MATERIAL_CLAIM",
        category="CONTENT_DEFECT",
        description="Material assertion lacks an explicit inline evidence marker.",
        suggestion="Add an adjacent validated citation or narrow/remove the assertion.",
        affected_claims=uncited,
    )
```

- [ ] **Step 6: Strengthen Worker evidence instructions**

Add the same contract to `worker_system_template.md`, `_evidence_context_for_generation()`, and `_bind_claims_to_evidence()`:

```text
Every material quantitative, causal, priority/superlative, or strong operational
assertion must carry an adjacent validated [E#] marker. Citations do not inherit
across sentences or paragraphs. A material inference must cite the evidence used
as its premise even when inference wording is explicit.
```

Do not add a binding call; update existing prompt text only.

- [ ] **Step 7: Run GREEN and compatibility tests**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_worker_tool_boundaries.py
python -m compileall -q src/evidence src/nodes/verifier.py
git diff --check
```

Expected: cited claims make one semantic call; uncited deterministic failures make zero.

- [ ] **Step 8: Commit**

```bash
git add src/evidence/claims.py src/nodes/verifier.py src/prompts/worker_system_template.md src/nodes/worker/agent/graph.py tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_worker_tool_boundaries.py
git commit -m "verifier: reject uncited material factual claims"
```

---

### Task 5: Freeze Verifier component and actual request-control baselines

**Files:**
- Modify: `tests/benchmark_support.py`
- Modify: `tests/test_offline_pipeline_benchmark.py`
- Modify: `tests/test_llm_observability.py`
- Create: `tests/measure_verifier_controls.py`
- Create: `docs/benchmarks/2026-08-29-verifier-controls-baseline.json`

**Interfaces:**
- Produces: `measure_template_contributions(template, values, groups) -> dict[str, int]`.
- Produces: an offline child-process capture of real `ChatOpenAI` HTTP JSON.
- Produces: frozen pre-optimization prompt components and current request mapping.
- Preserves: no production behavior change in this task.

- [ ] **Step 1: Add component measurement**

In `tests/benchmark_support.py` add:

```python
from collections.abc import Mapping, Sequence


def measure_template_contributions(template: str, values: Mapping[str, str], groups: Mapping[str, Sequence[str]]) -> dict[str, int]:
    blank = {key: "" for key in values}
    base_chars = len(template.format(**blank))
    metrics = {"base_instructions_chars": base_chars}
    for group_name, fields in groups.items():
        isolated = dict(blank)
        for field in fields:
            isolated[field] = values[field]
        metrics[f"{group_name}_chars"] = len(template.format(**isolated)) - base_chars
    metrics["verifier_prompt_total_chars"] = len(template.format(**values))
    component_total = sum(value for key, value in metrics.items() if key != "verifier_prompt_total_chars")
    assert component_total == metrics["verifier_prompt_total_chars"]
    return metrics
```

Groups are task contract, Worker result, claim payload, Worker assets, and other deterministic context. Keep chars distinct from tokens.

- [ ] **Step 2: Add real-wrapper offline request capture**

Create `tests/measure_verifier_controls.py`. Use the installed real `ChatOpenAI` with `httpx.MockTransport`, capture outgoing JSON, and return a minimal valid OpenAI chat response. Call production `get_llm(config, json_mode=True, purpose="assessment")`, apply `with_completion_budget(model, "assessment")`, and invoke one `HumanMessage`. Print JSON fields: model, `max_tokens`, `max_completion_tokens`, `reasoning_effort`, and whether `thinking` exists.

Use these exact imports in the probe module:

```python
import functools
import json
import os
import subprocess
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI as RealChatOpenAI

from src.llm import get_llm, with_completion_budget
from tests.test_offline_pipeline_benchmark import collect_verifier_pass_metrics


def capture_request(env: dict[str, str]) -> dict[str, object]:
    captured: dict[str, object] = {}
    probe_env = {
        "DEEPSEEK_API_KEY": "offline-test-key",
        "DEEPSEEK_BASE_URL": "https://offline.invalid/v1",
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
        **env,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 0, "model": captured["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    with patch.dict(os.environ, probe_env, clear=False), patch("src.llm.ChatOpenAI", functools.partial(RealChatOpenAI, http_client=client)):
        model = get_llm({}, json_mode=True, purpose="assessment")
        runnable, _ = with_completion_budget(model, "assessment")
        runnable.invoke([HumanMessage(content="offline verifier probe")])
    return captured
```

The subprocess base URL is `https://offline.invalid/v1`, but MockTransport handles the request locally. This module runs outside pytest so `tests/conftest.py` cannot replace LangChain.

- [ ] **Step 3: Characterize the current mapping**

Add to `tests/test_llm_observability.py`:

```python
def test_current_deepseek_request_mapping_is_measured_offline():
    payload = run_verifier_control_probe({})
    assert payload["max_completion_tokens"] == 1600
    assert payload["max_tokens"] is None
    assert payload["reasoning_effort"] is None
    assert payload["thinking_present"] is False
```

This passing test freezes the observed pre-fix behavior. It is changed to the desired assertion in Task 6.

- [ ] **Step 4: Extend the benchmark collector and artifact writer**

Extend `collect_verifier_pass_metrics()` with all component-character fields plus existing semantic call and mock completion metrics. Make `tests/measure_verifier_controls.py --output PATH` construct the artifact from live deterministic measurements:

```python
artifact = {
    "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "offline_only": True,
    "character_metrics_are_provider_tokens": False,
    "online_latency_remeasured": False,
    "requires_real_run": True,
    "verifier_prompt_components": collect_verifier_pass_metrics(),
    "request_mapping": {
        "expected_provider_field": "max_tokens",
        "observed_budget_field_before_fix": "max_completion_tokens",
        "reasoning_effort_configured": False,
        "thinking_controlled": False,
    },
}
```

The command never copies an old prompt builder.

- [ ] **Step 5: Run baseline measurement twice**

```bash
.venv/bin/python -m pytest -q tests/test_llm_observability.py tests/test_offline_pipeline_benchmark.py
.venv/bin/python tests/measure_verifier_controls.py --output docs/benchmarks/2026-08-29-verifier-controls-baseline.json
git diff --check
```

Expected: tests pass; the artifact records `max_completion_tokens` as the observed incorrect pre-fix field and stable character components.

- [ ] **Step 6: Commit**

```bash
git add tests/benchmark_support.py tests/test_offline_pipeline_benchmark.py tests/test_llm_observability.py tests/measure_verifier_controls.py docs/benchmarks/2026-08-29-verifier-controls-baseline.json
git commit -m "test: measure verifier provider controls"
```

---

### Task 6: Deduplicate semantic evidence and prove DeepSeek request controls

**Files:**
- Modify: `src/evidence/claims.py`
- Modify: `src/nodes/verifier.py:179-304`
- Modify: `src/prompts/verifier.md`
- Modify: `src/config.py:10-373`
- Modify: `src/llm.py:63-289`
- Modify: `.env.example`
- Modify: `tests/test_claim_grounding.py`
- Modify: `tests/test_llm_observability.py`
- Modify: `tests/test_offline_pipeline_benchmark.py`
- Modify: `docs/benchmarks/2026-08-29-verifier-controls-baseline.json`

**Interfaces:**
- Produces: `build_semantic_claim_payload(content, citations) -> {"claims": list, "evidence_catalog": dict}`.
- Produces: optional `VERIFIER_MODEL` and `VERIFIER_REASONING_EFFORT=low|high|max`.
- Produces: DeepSeek requests with `max_tokens`, no `max_completion_tokens`, optional faithful `reasoning_effort`, and no claimed thinking control.
- Preserves: `derive_claims()` compatibility, one call, semantic excerpt bound, assessment schema, and default model.

- [ ] **Step 1: Write RED deduplicated-payload tests**

Add to `tests/test_claim_grounding.py`:

```python
def test_semantic_payload_serializes_shared_evidence_once(fake_verifier):
    evidence = {"evidence_id": "E3", "title": "聚乙烯质量控制手册", "locator": "section 3.2", "supporting_text": "氢气用量影响熔融指数，温度变化也会改变产品指标。"}
    content = "氢气影响熔指。[E3] 温度影响指标。[E3] 两者需要联合排查。[E3]"
    result = fake_verifier.run(content, [evidence], _assessment())
    prompt = fake_verifier.calls[-1]
    assert result["assessment"]["status"] == "PASS"
    assert prompt.count(evidence["supporting_text"]) == 1
    assert prompt.count('"E3"') >= 4
    assert len(fake_verifier.calls) == 1
```

Add `test_catalog_preserves_late_e3_and_e6_support` with 4,000-character backgrounds followed by the critical support sentences; assert both late sentences occur in the rendered semantic prompt and each full projected excerpt occurs once.

- [ ] **Step 2: Change request assertions to desired provider fields and confirm RED**

Replace Task 5 characterization with:

```python
def test_deepseek_request_uses_max_tokens_not_max_completion_tokens():
    payload = run_verifier_control_probe({})
    assert payload["max_tokens"] == 1600
    assert payload["max_completion_tokens"] is None
    assert payload["reasoning_effort"] is None
    assert payload["thinking_present"] is False


def test_verifier_reasoning_effort_reaches_actual_request():
    payload = run_verifier_control_probe({"VERIFIER_REASONING_EFFORT": "low"})
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == 1600


def test_default_verifier_model_remains_global_model():
    payload = run_verifier_control_probe({"DEEPSEEK_MODEL": "deepseek-v4-flash"})
    assert payload["model"] == "deepseek-v4-flash"
```

Run:

```bash
.venv/bin/python -m pytest -q tests/test_llm_observability.py tests/test_claim_grounding.py
```

Expected RED: actual request contains `max_completion_tokens` and the semantic prompt repeats E3.

- [ ] **Step 3: Implement one semantic evidence catalog**

Preserve `derive_claims()` and add:

```python
def build_semantic_claim_payload(content: str, citations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    claims = derive_claims(content, citations)
    catalog: dict[str, dict[str, Any]] = {}
    compact_claims: list[dict[str, Any]] = []
    for claim in claims:
        compact_claims.append({"claim_id": claim["claim_id"], "text": claim["text"], "claim_type": claim["claim_type"], "evidence_ids": list(claim["evidence_ids"])})
        for evidence in claim["evidence"]:
            catalog.setdefault(evidence["evidence_id"], dict(evidence))
    return {"claims": compact_claims, "evidence_catalog": catalog}
```

P0 preflight owns conflicts. Update `verifier.md` to accept `{claims}` and `{evidence_catalog}` and serialize them from the same payload in `verifier.py`.

- [ ] **Step 4: Add optional Verifier configuration**

Add optional `verifier_model` and `verifier_reasoning_effort` to `AppConfig`. Parse `VERIFIER_REASONING_EFFORT` as empty/None or `low`, `high`, `max`; reject other non-empty values. Document commented examples in `.env.example`.

```python
@dataclass(frozen=True)
class AppConfig:
    deepseek_api_key: str | None = field(repr=False)
    deepseek_base_url: str
    deepseek_model: str
    verifier_model: str | None
    verifier_reasoning_effort: str | None


def _optional_reasoning_effort_from_env(name: str) -> str | None:
    value = str(get_env(name) or "").strip().lower()
    if not value:
        return None
    if value not in {"low", "high", "max"}:
        raise ValueError(f"{name} must be one of: low, high, max")
    return value
```

`get_llm(..., purpose="assessment")` selects `VERIFIER_MODEL` only when configured and passes reasoning effort only when configured. Change the AutoVerifier call to include `purpose="assessment"`.

- [ ] **Step 5: Map budgets to actual DeepSeek `max_tokens`**

In `src/llm.py`, pop budget from constructor/bind kwargs and merge it into supported `extra_body`:

```python
def _deepseek_extra_body(existing: Mapping[str, Any] | None, *, max_tokens: int) -> dict[str, Any]:
    return {**dict(existing or {}), "max_tokens": int(max_tokens)}
```

`get_llm()` passes `extra_body={"max_tokens": budget}`. `with_completion_budget()` binds `extra_body` with the purpose budget and never binds `max_tokens`. Preserve existing extra-body keys. Do not add `thinking`.

- [ ] **Step 6: Update optimized benchmark evidence**

Run the same fixture and add an `optimized` object to the Task 5 artifact. It contains component characters, one semantic call, one serialized E3 excerpt, and corrected request fields. Keep provider tokens and online latency as null, `online_latency_remeasured=false`, and `requires_real_run=true`. Assert optimized total prompt characters are below frozen pre-dedupe baseline and component chars sum to actual prompt chars.

- [ ] **Step 7: Run focused GREEN tests**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_llm_observability.py tests/test_offline_pipeline_benchmark.py
python -m compileall -q src tests
git diff --check
```

Expected: captured JSON has `max_tokens=1600`, no `max_completion_tokens`, optional effort only when configured, no thinking, one E3 excerpt, and one semantic call.

- [ ] **Step 8: Commit**

```bash
git add src/evidence/claims.py src/nodes/verifier.py src/prompts/verifier.md src/config.py src/llm.py .env.example tests/test_claim_grounding.py tests/test_auto_verifier_routing.py tests/test_llm_observability.py tests/test_offline_pipeline_benchmark.py docs/benchmarks/2026-08-29-verifier-controls-baseline.json
git commit -m "verifier: deduplicate semantic evidence payload"
```

---

### Task 7: Replace the narrow appendix table with safe evidence blocks

**Files:**
- Modify: `src/evidence/text_projection.py:1-31`
- Modify: `src/evidence/reporting.py:237-282`
- Modify: `tests/test_report_renderers.py:12-498`
- Modify: `tests/test_summarizer_deterministic.py`

**Interfaces:**
- Consumes: `canonical_citation_identity()` after P0 validation.
- Produces: unchanged `format_grouped_evidence_appendix(...) -> str` signature with block layout.
- Preserves: source grouping, safe labels, provenance, renderer interface, natural E-ID order, and input immutability.

- [ ] **Step 1: Write RED sanitizer and layout tests**

Update `tests/test_report_renderers.py`:

```python
def test_grouped_appendix_uses_evidence_blocks_without_table_columns():
    citations = [{"evidence_id": "E1", "title": "工艺手册", "locator": "section 1", "section_title": "工艺控制", "supporting_text": "温度影响分子量。"}]
    markdown = format_grouped_evidence_appendix(citations)
    assert "| 证据编号 |" not in markdown
    assert "**[E1] section 1**" in markdown
    assert "支撑章节：" in markdown
    assert "摘要：" in markdown


def test_sanitizer_removes_ascii_and_fullwidth_table_noise():
    item = {"evidence_id": "E1", "title": "工艺手册", "locator": "section 1", "supporting_text": "甲 |||| 乙 ｜｜｜｜ 丙"}
    markdown = format_grouped_evidence_appendix([item])
    assert "|||" not in markdown
    assert "｜｜" not in markdown
    assert all(value in markdown for value in ("甲", "乙", "丙"))
```

Add `test_appendix_dedupes_only_exact_identity_without_mutation`: deep-copy one E1 citation, render both entries, assert `[E1]` occurs once, and assert the original list is deep-equal to its pre-render copy. Add the conflict case below.

```python
def test_appendix_never_dedupes_conflicting_display_identity():
    left = {"evidence_id": "E1", "file_path": "/docs/a.docx", "locator": "1", "supporting_text": "甲"}
    right = {"evidence_id": "E1", "file_path": "/docs/b.docx", "locator": "2", "supporting_text": "乙"}
    with pytest.raises(ValueError, match="FINAL_DISPLAY_IDENTITY_CONFLICT"):
        format_grouped_evidence_appendix([left, right])
```

- [ ] **Step 2: Run and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_report_renderers.py
```

Expected: current four-column layout and full-width noise fail.

- [ ] **Step 3: Extend only presentation sanitization**

Change:

```python
_REPEATED_TABLE_SEPARATORS = re.compile(r"[|｜](?:\s*[|｜]){2,}")
```

Do not write normalized text back to citations. Keep existing scientific text, unit, URL, and redaction regressions.

- [ ] **Step 4: Render deterministic per-evidence blocks**

Preserve group insertion order and natural E sort, then emit:

```python
lines.extend(["", f"**[{evidence_id}] {locator}**", f"支撑章节：{section_title}", f"摘要：{summary}"])
```

Presentation dedupe keys are `(display_id, canonical_citation_identity)`. If one display ID has multiple identities, raise `ValueError("FINAL_DISPLAY_IDENTITY_CONFLICT")` rather than choosing one; exact duplicate pairs render once. In the integrated report path, Task 2 blocks the same conflict before formatting. Never expose identity components.

- [ ] **Step 5: Run GREEN integration tests**

```bash
.venv/bin/python -m pytest -q tests/test_report_renderers.py tests/test_summarizer_deterministic.py tests/test_report_evidence_integration.py
python -m compileall -q src/evidence
git diff --check
```

Expected: all pass, long PDF fixture renders, inputs remain unchanged, and no four-column appendix remains.

- [ ] **Step 6: Commit**

```bash
git add src/evidence/text_projection.py src/evidence/reporting.py tests/test_report_renderers.py tests/test_summarizer_deterministic.py
git commit -m "report: simplify evidence appendix projection"
```

---

### Task 8: Run final verification and adversarial architecture review

**Files:**
- Review only: `src/state.py`, `src/persistence.py`, `src/job_store.py`, `src/graph.py`
- Review only: all files changed since `cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb`

**Interfaces:**
- Consumes: completed Tasks 1-7.
- Produces: evidence-backed final verification; no production change unless a new red-green correction is required.

- [ ] **Step 1: Run every required focused suite**

```bash
.venv/bin/python -m pytest -q tests/test_claim_grounding.py
.venv/bin/python -m pytest -q tests/test_auto_verifier_routing.py
.venv/bin/python -m pytest -q tests/test_failure_decision_policy.py
.venv/bin/python -m pytest -q tests/test_recovery_policy.py
.venv/bin/python -m pytest -q tests/test_recovery_compatibility.py
.venv/bin/python -m pytest -q tests/test_report_evidence_integration.py
.venv/bin/python -m pytest -q tests/test_summarizer_deterministic.py
.venv/bin/python -m pytest -q tests/test_report_renderers.py
.venv/bin/python -m pytest -q tests/test_offline_pipeline_benchmark.py
```

Expected: every command passes without network access.

- [ ] **Step 2: Run full verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src app.py run.py
git diff --check
git status --short
git log --oneline cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb..HEAD
```

Expected: pytest passes; compileall and diff check are silent; status contains only the pre-existing archive.

- [ ] **Step 3: Prove forbidden surfaces are unchanged**

```bash
git diff --exit-code cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb..HEAD -- src/state.py src/persistence.py src/job_store.py src/graph.py
```

Expected: exit 0. Also inspect changed filenames and confirm no checkpoint migration, workflow route, or renderer implementation was added.

- [ ] **Step 4: Run the adversarial checklist**

For each item, identify a passing test and exact authority function:

```text
local and global E-IDs cannot alias
lossless registry reaches both validation phases
raw provenance is not mutated
issue permutations choose identical action and selected code
degradable issues cannot absorb non-degradable issues
Verifier PASS cannot be followed by final citation pollution
semantic evidence head/tail support remains present
actual request JSON contains max_tokens and optional reasoning_effort
presentation dedupe cannot hide an identity conflict
report failure occurs before path resolution and artifact creation
```

If evidence is missing, write a failing regression, implement the smallest correction, rerun the focused suite, and commit the correction separately.

- [ ] **Step 5: Collect final handoff facts**

```bash
git diff --stat cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb..HEAD
git log --reverse --oneline cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb..HEAD
```

Report changed files and reasons, regression tests, exact pytest result, final citation data flow, recovery precedence, schema/resume compatibility, semantic call count, before/after prompt characters and repeated-evidence count, proven provider controls, appendix change, remaining risks, diff stat, and commits. Mark provider-token and online-latency optimization as requiring a real run.

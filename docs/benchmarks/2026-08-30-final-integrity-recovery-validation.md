# Final Integrity and Recovery Validation

Date: 2026-08-30 (verification rerun 2026-08-31 Asia/Shanghai)

Branch: `codex/final-integrity-recovery`

Baseline production commit: `cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb`

Branch start: `1e7ef1b5a1084f84c769661c8546c2d8d252d0e3`

Initial Task 8 verification base: `6b38d8f` (`report: simplify evidence appendix projection`). Initial final-review correction base: `afdf678` (`docs: clarify final verification evidence`). Follow-up correction base: `7409410` (`fix: preserve citation bindings through final assembly`).

## Final-review correction verification

The four final-review findings were reproduced before production edits: the citation-integrity and Summarizer suites reported **11 failed, 51 passed**. Corrections now preserve inherited binding scope, check current per-section visible IDs before remapping, allocate collision-free unused report IDs without dropping raw records, and validate actual body slices from the single assembled Markdown independently of the appendix. Both removed unused-E2 fixtures have been restored.

Re-review resolved those four findings but exposed one remaining raw-marker alias: body `已绑定结论 [E8]。未知结论 [E1]。` with only visible citation E8 could pass after E8 was allocated display E1. Six follow-up regressions first failed (**6 failed, 74 deselected in 10.66s**), covering all three body-field aliases, another section's E1 binding, and no-normalization/no-path/no-artifact delivery checks. Preflight now reuses `_section_body()` and the current section's visible citation IDs to emit `LOCAL_CITATION_BINDING_MISSING` for every unbound raw body marker. Origin and canonical-conflict checks remain unchanged; the allocator, assembly, and metadata contract are untouched by this follow-up.

Fresh offline verification on 2026-08-31 used `/Users/synzzz/Documents/work_space/agent/agent-master/.venv/bin/python`:

| Command | Result |
| --- | --- |
| `.../.venv/bin/python -m pytest -q tests/test_citation_integrity.py tests/test_report_evidence_integration.py tests/test_summarizer_deterministic.py tests/test_synthesis.py tests/test_evidence_pipeline.py tests/test_report_renderers.py` | **126 passed in 10.69s** |
| `.../.venv/bin/python -m pytest -q` | **941 passed in 27.68s** |
| `.../.venv/bin/python -m compileall -q src app.py run.py` | exit 0, silent |
| `git diff --check` | exit 0, silent |
| `git diff --exit-code cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb -- src/state.py src/persistence.py src/job_store.py src/graph.py` | exit 0, silent |

Additional regressions cover same-visible canonical duplicates with different inherited keys, including keys allocated by prior sections; ambiguous uncited records still fail preflight. Unused allocation also reserves existing body IDs so it cannot accidentally bind an unknown marker. The real synthesis-context → synthesis-consistency → Summarizer delivery regression uses VERIFIED_PASS sections and stubs only model/provider and rendering boundaries.

Assembly span metadata is a local list of `(start, end)` offsets. It is not a State/checkpoint field, not returned as report data, and not embedded as control markers in Markdown. Tests verify one assembly, an appendix inserted between body sections, heading-name independence, exact body slices, plain `str` inputs to both renderers, unchanged source State/provenance, and no artifacts or path resolution after a failed gate. Rendering code and appendix presentation were unchanged, so no new presentation rendering was required.

Independent final branch re-review is **pending**; these test results are implementation verification, not an independent-review pass.

## Initial Task 8 verification result (historical)

All required focused suites and the full suite passed offline. The required relative `.venv/bin/python` path is not present inside the worktree; the same configured interpreter was invoked by its explicit path, `/Users/synzzz/Documents/work_space/agent/agent-master/.venv/bin/python`.

Focused commands:

| Command | Result |
| --- | --- |
| `.../.venv/bin/python -m pytest -q tests/test_claim_grounding.py` | 58 passed |
| `.../.venv/bin/python -m pytest -q tests/test_auto_verifier_routing.py` | 71 passed |
| `.../.venv/bin/python -m pytest -q tests/test_failure_decision_policy.py` | 43 passed |
| `.../.venv/bin/python -m pytest -q tests/test_recovery_policy.py` | 53 passed |
| `.../.venv/bin/python -m pytest -q tests/test_recovery_compatibility.py` | 76 passed |
| `.../.venv/bin/python -m pytest -q tests/test_report_evidence_integration.py` | 5 passed |
| `.../.venv/bin/python -m pytest -q tests/test_summarizer_deterministic.py` | 35 passed |
| `.../.venv/bin/python -m pytest -q tests/test_report_renderers.py` | 15 passed |
| `.../.venv/bin/python -m pytest -q tests/test_offline_pipeline_benchmark.py` | 6 passed |

Focused total: 362 passed. Additional integrity/control tests, `tests/test_citation_integrity.py tests/test_llm_observability.py`: 43 passed. Full command `/Users/synzzz/Documents/work_space/agent/agent-master/.venv/bin/python -m pytest -q`: **914 passed in 25.92s**. `compileall -q src app.py run.py`: exit 0 and silent. `git diff --check`: exit 0 and silent.

The first attempted focused invocations using the brief's relative `.venv/bin/python` all exited 127 because that path does not exist in the worktree; no pytest test body ran in those attempts. The reruns used the configured absolute interpreter and passed.

## Adversarial architecture checklist

Each invariant has a passing regression and an exact authority function:

| Invariant | Authority | Passing regression |
| --- | --- | --- |
| Local and global E-IDs cannot alias | `canonical_citation_identity()`, `citation_binding_key()`, `evidence_key()`, `validate_pre_remap_citation_integrity()`, `normalize_sections_evidence()`, and `validate_final_citation_integrity()` in `src/evidence/identity.py` and `src/evidence/integrity.py` | `test_pre_remap_scopes_same_local_id_by_task`, `test_normalized_synthesis_is_delivered_with_original_citation_scopes`, `test_visible_id_conflict_blocks_before_remap_and_delivery`, `test_unused_raw_ids_do_not_alias_allocated_display_ids_or_mutate_sources` |
| Lossless registry reaches both validation phases | `validate_pre_remap_citation_integrity()` → `normalize_sections_evidence()` → `project_lossless_used_citations()` → `validate_final_citation_integrity()` | `test_pre_remap_rejects_same_task_local_id_with_two_identities`, `test_lossless_projection_retains_conflicting_entries`, `test_final_gate_rejects_registry_id_unused_by_body` |
| Unknown raw body markers cannot acquire a binding from remapping or another section | `_section_body()` and current-section visible IDs in `validate_pre_remap_citation_integrity()` | `test_pre_remap_rejects_unknown_raw_marker_before_used_id_collision`, `test_pre_remap_rejects_unknown_raw_marker_despite_other_section_binding`, `test_unknown_raw_marker_blocks_before_normalization_paths_and_artifacts` |
| Raw provenance is not mutated | `normalize_evidence_text()` and copy-based `normalize_sections_evidence()` / appendix projection | `test_text_projection_normalizes_noise_without_mutating_source_object`, `test_appendix_dedupes_only_exact_identity_without_mutation` |
| Issue permutations choose the identical action and selected code | `_profile_assessment()` and `_set_decision()` in `src/recovery/policy.py` | `test_multi_issue_policy_is_order_independent` |
| Degradable issues cannot absorb non-degradable issues | `_profile_assessment()`, `_can_commit_with_warning()`, `_commit_degraded_result()`, and `NON_DEGRADABLE_ISSUE_CODES` | `test_mixed_gap_and_semantic_issue_never_commits_warning`, `test_shared_warning_gate_rejects_semantic_issue_and_allows_waivable_gaps`, `test_shared_warning_gate_rejects_blocker_verifier_and_fatal_profiles` |
| Verifier PASS cannot be followed by final citation pollution | `verifier()` ordering: `_sanitize_assessment()` → `_apply_citation_integrity()` → `_apply_deterministic_validation()`; final report gate in `summarizer()` | `test_verifier_requires_inline_binding_when_citations_are_available`, `test_verifier_rejects_inline_id_when_structured_citations_are_empty`, `test_sanitizer_rejects_pass_assessment_that_still_contains_issues`, `test_final_citation_conflict_blocks_before_delivery_paths` |
| Semantic evidence head/tail support remains present | `semantic_evidence_excerpt()` in `src/evidence/text_projection.py` and `build_semantic_claim_payload()` | `test_semantic_excerpt_preserves_late_support_while_presentation_stays_short`, `test_catalog_preserves_late_e3_and_e6_support` |
| Actual request JSON contains `max_tokens` and optional `reasoning_effort` | `with_completion_budget()` / `_with_deepseek_budget()` and `invoke_llm()` in `src/llm.py` | `test_deepseek_request_uses_max_tokens_not_max_completion_tokens`, `test_verifier_reasoning_effort_reaches_actual_request`, `test_assessment_budget_preserves_bound_extra_body_fields`, `test_assessment_budget_preserves_bound_listener_and_type_semantics` |
| Presentation dedupe cannot hide an identity conflict | `format_grouped_evidence_appendix()` in `src/evidence/reporting.py` | `test_appendix_never_dedupes_conflicting_display_identity`; exact duplicate behavior is covered by `test_appendix_dedupes_only_exact_identity_without_mutation` |
| Report failure occurs before path resolution and artifact creation | `summarizer()` preflight/final gate before `get_session_cache_dir()`, `os.makedirs()`, and renderer calls | `test_final_citation_conflict_blocks_before_delivery_paths`, plus `test_blocked_report_does_not_create_delivery_files` |
| A surviving appendix citation cannot mask loss from the final body | `_assemble_markdown()` invocation-local body offsets and `validate_final_citation_integrity(..., body_spans=...)` | `test_final_body_slice_excludes_appendix_when_body_marker_was_lost`, `test_lost_body_marker_cannot_be_masked_by_surviving_appendix`, `test_assembly_body_spans_are_invocation_local_and_markdown_stays_plain_str`, `test_assembly_body_spans_do_not_classify_body_by_heading_name` |

The initial Task 8 checklist missed four concrete defects; its green tests were not sufficient evidence of final citation correctness. The final-review correction regressions and fresh verification above supersede that initial conclusion.

## Citation and recovery flow

The final citation flow is invocation-local and lossless:

1. `_ordered_sections()` selects admitted sections.
2. `validate_pre_remap_citation_integrity()` checks all raw records, including unused ones, by the shared `citation_binding_key()` authority in their original inherited scope, separately checks each current section's visible-ID canonical identities, and requires every raw body marker to have a current-section visible binding before remapping. A binding in another section or an inherited local-ID field cannot satisfy this raw-body requirement.
3. `normalize_sections_evidence()` deep-copies sections, assigns display IDs from task-scoped binding keys, coalesces only same-visible canonical duplicate aliases, and rewrites report-only fields. Unused records remain intact with deterministic noncolliding IDs; the public display map still describes used bindings.
4. `_assemble_markdown()` creates one plain-string report projection and records body offsets in a local output list. No heading-name inference, reassembly, report markers, or persistent metadata are used. `project_lossless_used_citations()` retains every normalized citation record referenced by a body marker, including conflicting entries rather than overwriting by ID.
5. `validate_final_citation_integrity()` slices the exact final Markdown at those offsets and independently checks actual final-body IDs, normalized-body IDs, lossless registry IDs, whole-Markdown pollution, remap aliases, identity conflicts, missing bindings, and unused bindings. Appendix IDs cannot satisfy the body-preservation check.
6. Only after that gate succeeds does `summarizer()` resolve report paths, create directories, write Markdown, or call PDF/DOCX renderers.

Recovery computes one complete assessment profile in `_profile_assessment()` before action selection. Category priority and explicit requirement linkage make the selected action and representative code independent of input issue order. Any non-degradable issue prevents warning commit; fatal/verifier failures outrank semantic/content issues. Existing `decide_recovery_action()` actions, counters, blocker records, and graph routes remain unchanged.

## Schema and resume compatibility

The forbidden production surfaces are unchanged:

```text
git diff --exit-code cebebea8fc38a5c7d9abff8512c9dee6d1d8d8cb..HEAD \
  -- src/state.py src/persistence.py src/job_store.py src/graph.py
=> exit 0
```

No State/checkpoint schema, persistence format, migration, graph topology, action enum, or Recovery branch was added. Compatibility evidence includes `test_legacy_checkpoint_uses_safe_recovery_defaults_without_full_replan`, `test_staged_full_replan_can_resume_old_plan_without_mutation`, `test_replacement_model_failure_stages_error_and_can_resume_old_plan`, `test_restored_checkpoint_without_registry_reserves_historical_task_ids`, `test_legacy_integer_counter_keys_map_to_tasks_or_are_ignored`, and `test_state_keeps_legacy_replan_literal_for_checkpoint_compatibility` in `tests/test_recovery_compatibility.py`.

## Verifier controls and measurements

The controls artifact is the actual offline probe result from `tests/measure_verifier_controls.py` and `tests/test_llm_observability.py`:

```json
{
  "model": "deepseek-v4-flash",
  "max_tokens": 1600,
  "max_completion_tokens": null,
  "reasoning_effort": null,
  "thinking_present": false
}
```

When configured, `VERIFIER_REASONING_EFFORT=low` reaches the actual request as `reasoning_effort: "low"`. Bound `extra_body`, tools, listener, and input/output type semantics remain preserved.

The verifier-only compact-JSON comparison uses the same frozen single-claim fixture and one semantic call in both cases:

| Metric | Frozen pre-dedupe | Current optimized | Difference |
| --- | ---: | ---: | ---: |
| Serialized prompt characters | 3353 | 3340 | -13 characters |
| Template prompt characters | 3184 | 3168 | -16 characters |
| Semantic LLM calls | 1 | 1 | unchanged |

The separate semantic-catalog deduplication comparison uses its repeated-evidence fixture:

| Metric | Before dedupe | Current optimized | Difference |
| --- | ---: | ---: | ---: |
| Serialized shared E3 excerpt occurrences | 3 | 1 | -2 |

The 13-character serialized reduction is a minor serialization/label-encoding effect, not an additional model-call reduction. The semantic catalog comparison also remains one semantic call and emits one shared E3 excerpt; the output schema remains the existing five-field PASS shape. `test_compact_verifier_pass_and_failed_outputs_keep_one_semantic_call`, `test_verifier_pass_benchmark_records_exact_snapshot_comparison`, `test_optimized_verifier_controls_snapshot_preserves_frozen_baseline`, and `test_catalog_preserves_late_e3_and_e6_support` provide the regression evidence.

There is no cross-lane comparison. The historical real online reference run reported 7 calls and is context only; the full-pipeline 6→5 total-call comparison uses fixed offline fixtures on both sides (3→2 Worker calls/generations/loop iterations and 1→0 duplicate query requests/guard rejections). The verifier-only artifact separately compares a one-call baseline fixture with a one-call optimized fixture. None of these offline fixture values is a re-run of the real online path.

Frozen online reference values remain `provider_total_tokens=42961`, `verifier_provider_tokens=17717`, `llm_latency_seconds=172.2`, and `verifier_latency_seconds=84.7`. The optimized provider token and online latency fields are null, `online_latency_remeasured=false`, and `requires_real_run=true`. No provider-token or online-latency gain is claimed from mocks or character counts; real-run validation is still required.

## Appendix and remaining risks

`format_grouped_evidence_appendix()` now emits grouped evidence blocks rather than a narrow four-column table. The Task 7 long-evidence PDF fixture was rendered and inspected by the root agent with Poppler: full-width blocks, readable Chinese glyphs, no overlap/clipping, and no narrow columns. Existing pagination may split long entries; PDF/DOCX renderer implementations were not changed. The focused renderer suite passes 15 tests.

Remaining risks include real-provider behavior, the intentionally conservative uncited-material detector (it is not an exhaustive prose extractor), and renderer pagination on unusually long evidence entries. Python 3.14 direct real-wrapper probing emits the known Pydantic-v1 dependency warning; no dependency change is in scope.

## Changed files and commits

Before adding the initial Task 8 validation report, the production-baseline range (`cebebea8...` to `6b38d8f`) was 31 files, 4,324 insertions and 145 deletions. Task 8 added one tracked validation artifact. Final-review corrections modify existing files only, so the branch still changes 32 files against the production baseline. The correction scope is three production files (`identity.py`, `integrity.py`, `summarizer_v2.py`), two regression suites, the implementation plan, and this report. The companion `.superpowers/sdd/task-8-report.md` and `.superpowers/sdd/final-fix-report.md` remain local ignored reports and are intentionally not tracked. The pre-existing archive is preserved in the original checkout at `/Users/synzzz/Documents/work_space/agent/agent-master/agent-master-clean-20260828.zip`; it is not a branch change.

The branch commits, in order, are:

```text
ccf0b7d docs: design final integrity and recovery fixes
eb8978c docs: refine final integrity invariants
1e7ef1b docs: plan final integrity and recovery fixes
241fa4f evidence: preserve lossless citation identity
2be7d43 evidence: reject basename citation identities
6a1c579 report: validate final citation identity
09b36a8 fix: preserve final display ID aliases
528d731 recovery: aggregate unresolved issue severity
555cdd6 fix: harden recovery warning gate
61494c5 verifier: reject uncited material factual claims
0ba1376 verifier: tighten uncited claim boundaries
40d3fae verifier: handle markdown and ASCII claim boundaries
e6a7517 verifier: split ASCII claim boundaries
444c560 test: measure verifier provider controls
1935af1 verifier: deduplicate semantic evidence payload
910440e fix: preserve bound verifier request controls
228b1f1 fix: preserve verifier binding semantics
6b38d8f report: simplify evidence appendix projection
5249d95 report: record final integrity verification
afdf678 docs: clarify final verification evidence
7409410 fix: preserve citation bindings through final assembly
```

The bounded citation-correction commit contains the current verification update; its independent final re-review is owned by the root agent.

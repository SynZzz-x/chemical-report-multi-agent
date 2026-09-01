# Verifier / terminal UI / Planner diagnostics validation

## Scope and initial diagnosis

Base commit: `6a045e3` (`planner: handle completion truncation explicitly`).
Implementation branch: `codex/verifier-terminal-diagnostics`.
Fresh offline baseline: **967 passed in 37.07s**.

The real-run facts supplied for this patch are distinct from earlier runs:

- Planner attempt 1 used a 16384 completion budget and returned 7547 completion / 6608 reasoning tokens, then failed local contract validation. Attempt 2 returned 6090 / 4949 and produced five tasks. This is not evidence of Planner truncation, and does not identify the failed validator.
- Both Verifier attempts used 1600 completion tokens, all reported as reasoning, and raised `LengthFinishReasonError`. Existing recovery retried once and then terminated with FAIL_JOB / FATAL_SYSTEM.
- Streamlit still presented ordinary input after terminal failure.

Code inspection before changes found:

1. `assessment` had a hardcoded 1600 budget in `src/llm.py`. The existing shared wrapper correctly maps its internal completion option to DeepSeek `max_tokens`. Verifier catches invocation exceptions as retryable `LLM_ERROR`, while the shared invocation logger already safely reads SDK partial usage. Existing optional reasoning-effort configuration defaults to absent.
2. `app.py` already projects the snapshot with `project_job_outcome` after normal stream completion and saves failed status without needing a Python exception. The missing behavior is persistent failure presentation after rerender, and a submit guard before ordinary input unconditionally writes `running`. Restored history needs the same projection. Pending interrupts/blockers retain the existing precedence; job status must not become an independent workflow authority.
3. Planner logs `plan_contract_invalid`, but loses validation stage/code. Possible sources include JSON/envelope validation, task field/schema checks, requirements, dependencies, sequential IDs, RAG query consistency, resource resolution/data requirements, synthesis/visualization/web policy, section coverage and replan normalization. The supplied log does not justify ranking one as the most likely cause. The earlier CSV/resource diagnostic is not evidence for this later run.

## Budget evidence boundary

Verifier assessment's selected default is 8192, providing headroom above the supplied successful-run reasoning envelope of roughly 5–6k. This is not a guarantee that every assessment fits. A positive environment override uses the same cached configuration/restart behavior as Planner. Contract repair and other node budgets are outside this change.

DeepSeek's [Chat Completions reference](https://api-docs.deepseek.com/api/create-chat-completion/) defines `max_tokens`, and the [model details](https://api-docs.deepseek.com/quick_start/pricing/) listed a maximum output of 384K when checked on 2026-08-31. The selected cap is within that range. Offline MockTransport tests prove serialization only, not online budget sufficiency, quality, latency or cost.

## Frozen boundaries

No State/checkpoint schema or migration changes; no graph/Recovery policy or branch changes; no retry-count changes; no additional semantic calls; no assessment schema or grounding reductions; no Planner contract/validator/prompt changes; no default reasoning-effort changes; no lexical/embedding support gate; no unrelated renderer/report/citation cleanup.

## Execution evidence

### P0-1 Verifier

Commit: `32b2406 verifier: handle completion truncation explicitly`.

RED: `python -m pytest tests/test_verifier_completion.py -q` → **5 failed, 7 passed in 7.38s**. Failures captured the absent configuration, actual old cap/ignored override, and absent truncation diagnostics.

GREEN: `python -m pytest tests/test_verifier_completion.py tests/test_llm_observability.py tests/test_auto_verifier_routing.py tests/test_recovery_policy.py tests/test_planner_completion.py -q` → **187 passed in 23.91s**.

Primary-agent additional verification: `python -m pytest tests/test_failure_decision_policy.py tests/test_runtime_config.py tests/test_verifier_completion.py -q` → **59 passed in 7.48s**.

Configuration is in `src/config.py` (`DEFAULT_VERIFIER_MAX_COMPLETION_TOKENS`, cached `AppConfig.verifier_max_completion_tokens`, positive environment parsing), documented in `.env.example`. Only `completion_budget("assessment")` selects it. `assessment_contract_repair` remains 900; Planner 16384, Intake 1200, default Worker 3200, Synthesis 4000 remain unchanged.

Safe warning `reason=verifier_generation_truncated` records SDK type, purpose, requested cap and task/attempt. Shared `LLM_CALL_ERROR` retains numeric provider usage when available, otherwise unknown. Recovery still receives the exact existing retryable `LLM_ERROR` failure object, not a semantic claim issue. The integration fixture confirms one retry then FATAL_SYSTEM; complete PASS/FAILED fixtures use one semantic call each. Repair truncation is attributed to repair/900, not assessment/8192.

The warning's `attempt` is the invocation-local contract attempt: assessment is 1, contract repair is 2. Recovery's cross-invocation retry count remains the existing `verifier_retry_count`; it is not replaced by the local field.

Actual installed-wrapper MockTransport checks assert default `max_tokens=8192`, override `max_tokens=10000`, no `max_completion_tokens`, no default reasoning/thinking override, and unchanged other-purpose caps. Existing explicit-low plumbing tests still pass.

### P0-2 Streamlit terminal projection

Initial commit: `665d23e ui: project terminal workflow failures`.

RED sequences in `tests/test_ui_projection.py`: three expected failures for absent helper/guard; three for snapshot authority and unsafe diagnostic echo; two for failed-history overwrite and misleading restore wording. All were observed before their corresponding corrections.

Focused command: `python -m pytest tests/test_ui_projection.py tests/test_job_store.py tests/test_fatal_system.py tests/test_recovery_compatibility.py -q` → **99 passed in 0.35s**. Syntax compilation of `app.py` / `src/ui_projection.py` and the staged diff check passed.

Primary full checkpoint after the first two commits: `python -m pytest -q` → **987 passed in 41.73s**.

On rerender, the app reads the graph snapshot and pending interrupt, calls existing `project_job_outcome`, and passes that outcome to a presentation-only helper. The graph projection wins over stale JobStore status; stored fatal information provides fallback if the snapshot is unreadable. Existing failed-to-completed history protection remains. A safe persistent message plus disabled chat and a `st.stop()` dispatch guard prevent ordinary input from persisting uploads, changing the fatal job to running, or invoking the graph. History opens the task for inspection and labels terminal tasks as stopped. Checkpoints and intermediate results are retained.

New tests execute extracted app functions and the actual guard/dispatch AST fragment with fake Streamlit/snapshot/store interfaces. They assert no dispatch side effects after the stop, rather than relying solely on source-text matching. No browser automation dependency was added.

Independent review found one Important gap: a partial legacy graph fatal record missing `origin` caused the presentation helper to miss an authoritative failed outcome. Fix commit `cce14fa ui: fail close partial terminal outcomes` followed an executable RED regression (one failure); focused UI/store/fatal/recovery verification then gave **100 passed in 0.30s**. Independent rereview approved with no remaining findings. Supplied failed snapshot outcomes now fail closed despite missing metadata; explicitly runner-origin failures retain the existing resubmission behavior.

Primary full checkpoint after the review fix: **988 passed in 40.86s**. Historical message rendering was inspected and is display-only; older text is preserved, with a current stopped banner/guard. An actual remote persisted-history browser smoke test was not performed.

### P1 Planner validation observability

Commit: `d23af30 planner: expose contract validation failures`.

RED: `python -m pytest tests/test_planner_completion.py -q` → **6 failed, 26 passed in 7.49s**.

GREEN: `python -m pytest tests/test_planner_completion.py tests/test_planner_contract.py -q` → **183 passed in 9.25s**.

Existing exception objects receive invocation-local metadata; their original class and text are retained. Contract-invalid warnings add `validation_stage`, `validation_code` and a bounded canonical task ID (otherwise `-`). Non-contract failures retain their existing format. A mixed invalid-contract → SDK-length fixture confirms that the second log contains no stale validation metadata. No response body, user request or arbitrary malformed task ID is logged by the new telemetry.

A quota-interrupted review fix was recovered before completion. It restricts task attribution to the task's already-valid positional `Tn`, so a malformed-but-plausible ID such as `T999999999` cannot reach telemetry. It also keeps the shared legacy replacement dependency helper free of generation-only annotations. A standalone desired-behavior probe against `d23af30` failed with both violations; the same probe against the fix passed. Focused Planner verification after the fix: **184 passed in 10.76s**. Commit: `cb41c3c planner: constrain validation telemetry context`.

| Stage | Example stable codes |
| --- | --- |
| `json_envelope` | `invalid_json`, `invalid_envelope`, `invalid_tasks` |
| `task_schema` | `invalid_task_schema`, `task_limit_exceeded`, `non_sequential_task_ids` |
| `requirement_dependency` | `unknown_requirement`, `invalid_dependency` |
| `task_semantics` | `rag_query_consistency`, `synthesis_semantics`, `visualization_semantics` |
| `resource_policy` | `invalid_resource_assignment`, `data_resource_required`, `figure_data_resource_required` |
| `tool_policy` | `evidence_channel_required`, `web_authorization_required` |
| `section_coverage` | `invalid_section_coverage` |

Initial, replacement and refinement generation share the same logging boundary. Separate post-generation legacy replacement normalization remains unchanged; this patch does not refactor those validators. The existing refinement exception logger is also unchanged.

Primary AST audit against `6a045e3`: after removing only diagnostic wrappers, generation-only context plumbing and local task-ID attribution, nine parser/validator/resource/replacement helpers have identical ASTs. This supplements, rather than replaces, the existing contract tests. The two-attempt loop and JSON-repair versus truncation guidance remain unchanged.

There is still no evidence to identify or rank the failed validator in the supplied real Planner attempt 1. Resource checks are one possible source, not an established diagnosis. New logs will make a future failure attributable.

## Final verification and review

Primary verification before the recovered review fix, at `d23af30`:

- `python -m pytest -q` → **995 passed in 40.91s**.
- `python -m compileall -q src app.py run.py tests` → exit 0, no output.
- `git diff --check` and `git diff --check 6a045e3..HEAD` → exit 0, no output.
- Tracked file-range audit: changes limited to configuration, LLM assessment-budget selection, Verifier diagnostic catches, UI projection, Planner diagnostic metadata, and their tests. State, persistence, graph, Recovery policy, prompts, assessment contract, citation and renderer files are unchanged.
- Task reviews approved. The UI review's Important partial-fatal finding was fixed and rereviewed. Planner review suggested optional additional direct metadata cases for resource/tool/coverage stages; final review retained this as non-blocking test polish.

Fresh final verification after `cb41c3c`:

- `python -m pytest -q` → **996 passed in 44.35s**.
- `python -m compileall -q src app.py run.py tests` → exit 0, no output.
- Worktree and `6a045e3..HEAD` `git diff --check` → exit 0, no output.

Final verification after the reviewed UI wording fix `e9d8391`:

- `python -m pytest -q` → **997 passed in 44.76s**.
- `python -m compileall -q src app.py run.py tests` → exit 0, no output.
- Worktree and `6a045e3..HEAD` `git diff --check` → exit 0, no output.
- Independent focused rereview closed the restore-toast finding. Whole-range verdict: **Ready to merge: Yes**, with no Critical or Important findings.

## Changed files

Production/configuration: `.env.example`, `app.py`, `src/config.py`, `src/llm.py`, `src/nodes/verifier.py`, `src/ui_projection.py`, `src/nodes/planner.py`.

Tests: `tests/measure_verifier_controls.py`, `tests/test_llm_observability.py`, `tests/test_planner_completion.py`, `tests/test_planner_contract.py`, `tests/test_ui_projection.py`, `tests/test_verifier_completion.py`.

Documentation: this validation report.

## Remaining limits

- Independent whole-branch review plus focused rereview of `6a045e3..e9d8391` found no Critical or Important issues and returned **Ready to merge: Yes**. One directly relevant Minor—misleading restore-toast wording for partial legacy fatal records—was reproduced (`1 failed, 13 deselected`) and fixed in `e9d8391 ui: align restored fatal messaging`; the focused UI/job/fatal/recovery verification passed **220 tests in 0.67s**. Direct code assertions for resource/tool/section telemetry remain an optional non-blocking test gap.
- No live provider calls were made. Online outcome remains **not re-measured online / requires real-run validation**. Increased budget removes the hardcoded 1600 cap; it does not prove 8192 sufficient for all real assessments or imply lower latency/cost.
- Real Streamlit browser smoke against the user's remote persisted job was not performed. Offline tests exercise the production projection/guard code with fake snapshot/store/Streamlit boundaries.
- `assessment_contract_repair` intentionally stays at 900 and can still truncate; it now has correctly attributed diagnostics and retains the existing exhaustion policy.
- Historical UI messages/checkpoints are retained, not rewritten or deleted. New terminal display/guards describe the current outcome.
- No automatic push, merge, or online replay is part of this patch.

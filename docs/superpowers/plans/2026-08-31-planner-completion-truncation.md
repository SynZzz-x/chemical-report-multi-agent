# Planner Completion Truncation Implementation Plan

> **For agentic workers:** Execute the user-approved bounded repair with red-green TDD and independent review before the single implementation commit.

**Goal:** Give Planner a separately configured completion budget and distinguish SDK length truncation from invalid complete plans.

**Architecture:** Keep the shared DeepSeek `extra_body.max_tokens` mapping. Resolve `plan_generation` through `AppConfig.planner_max_completion_tokens`; classify failures at the existing Planner retry boundary without changing its two-attempt loop. Read only numeric usage from `LengthFinishReasonError.completion.usage`, never prompt/completion bodies.

**Tech Stack:** Python, existing ChatOpenAI/OpenAI SDK, pytest, httpx.MockTransport.

## Global constraints and approved design

- No Planner schema, validator, contract, State/checkpoint, workflow, Recovery branch, or retry-count changes.
- No base Planner prompt optimization, reasoning-effort changes, new provider parameters, or dynamic retry budget escalation.
- `DEFAULT_PLANNER_MAX_COMPLETION_TOKENS = 16384`; `PLANNER_MAX_COMPLETION_TOKENS` accepts positive integers using the existing cached AppConfig/environment style. Both attempts use the one resolved budget.
- The user observed about 10.2k provider completion (about 9.3k reasoning). 16,384 provides roughly 60% headroom; it is not an offline proof of sufficiency.
- Checked 2026-08-31: [DeepSeek model limits](https://api-docs.deepseek.com/quick_start/pricing/) list V4 maximum output 384K; [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) supports `max_tokens`. Installed ChatOpenAI accepts integer budgets via the already-tested mapping. Overrides must suit the operator's actual model/provider limit.
- `LengthFinishReasonError` → `plan_generation_truncated`; complete response parsing/validation errors → `plan_contract_invalid`; other invocation errors → `plan_generation_failed` (local logging/guidance only).
- Truncation retry requests complete regeneration after an output-length interruption, not JSON repair; invalid-response retry retains current contract-repair guidance.
- SDK usage may be absent: preserve `None`, do not estimate reasoning or missing totals, do not serialize completion content or exception text into logs.
- Intake/Worker/Verifier/Synthesis budgets unchanged. Tests are offline; no claim of recovered online success or latency. Stop after report and local commit, no push.

## Task 1: Budget, classification and metadata-only observability

**Files:** `src/config.py`, `src/llm.py`, `src/nodes/planner.py`, `.env.example`; tests in `tests/test_planner_completion.py`, request fixture in `tests/measure_verifier_controls.py`, existing metadata expectation in `tests/test_llm_observability.py`.

- [x] RED: add configured/default/invalid Planner budget cases; actual SDK length error tests with/without usage; assert two attempts, truncation reason, distinct retry guidance, and no content leaks.
- [x] RED/compatibility: complete malformed JSON and invalid schema remain `plan_contract_invalid`; other request errors are not contract failures.
- [x] RED: MockTransport request asserts `max_tokens=16384` (or configured override), no `max_completion_tokens`, no reasoning/thinking additions. Check the other four purpose budgets under a Planner override.
- [x] Run new tests before production edits and record expected failures.
- [x] GREEN: add the AppConfig field/env lookup, route only `completion_budget('plan_generation')` to it, remove the old map entry.
- [x] GREEN: classify invocation/response failures in the unchanged two-attempt loop; log reason/attempt/requested budget and select retry guidance without changing validation.
- [x] GREEN: add best-effort length-error numeric usage to existing LLM error log; preserve exception type/re-raise, `None` for unavailable fields, no content serialization.
- [x] Run `python -m pytest -q tests/test_planner_completion.py tests/test_planner_contract.py tests/test_planner_helpers.py tests/test_llm_observability.py tests/test_runtime_config.py tests/test_offline_pipeline_benchmark.py` with the configured interpreter.
- [x] Run full pytest, compileall and diff-check; independently review forbidden-surface diff against `50c7420`.
- [x] Independent review; fix any important findings and reverify.
- [x] Commit with this document as `planner: handle completion truncation explicitly`; stop without merge/push.

## Verification record

Base commit: `50c7420`. Configured interpreter: `/Users/synzzz/Documents/work_space/agent/agent-master/.venv/bin/python`.

| Check | Result |
| --- | --- |
| Baseline full `python -m pytest -q` | 941 passed in 27.41s |
| RED `python -m pytest -q tests/test_planner_completion.py` before production edits | 15 failed, 7 passed in 8.27s |
| First GREEN focused / full | 208 passed / 963 passed |
| Final focused command above, including telemetry edge cases | 212 passed in 21.44s |
| Final full `python -m pytest -q` | 967 passed in 40.11s |
| `python -m compileall -q src app.py run.py` | exit 0 |
| `git diff --check` | exit 0 |
| Forbidden diff vs `50c7420`: State, persistence, job store, graph, Recovery, prompts, task contract | unchanged |

The RED failures covered absent config/override validation, unchanged 3500 requests, and misclassified truncation/request errors. Existing complete-invalid-response and other-purpose compatibility cases passed before implementation. The new suite uses real SDK `LengthFinishReasonError` objects; MockTransport uses installed `langchain-openai 1.0.0`, `openai 2.54.0`, and `httpx 0.28.1` with a fake key and offline transport.

Actual request assertions cover Planner defaults and 20000 override (constructor and binding), no `max_completion_tokens`, and no added `thinking`/`reasoning_effort`. A Planner override leaves Intake 1200, Worker default 3200, Verifier 1600, and Synthesis 4000 unchanged. Existing description-dependent Worker tests are also retained.

Independent review found no Critical/Important issues. Its optional telemetry coverage note was addressed with tests for missing totals, nonnumeric fields, malformed details, and extraction failure retaining the original SDK exception; rereview closed the note. No production changes followed the first GREEN run.

Unavailable usage remains `None` internally (rendered `-` by the existing logger). No prompt or partial completion is logged. No benchmark simulates reasoning latency or proves token sufficiency: online budget sufficiency and latency are **not re-measured online / require real-run validation**. This commit is local on `codex/planner-completion-truncation`; the previously pushed target branch is unchanged.

# AutoVerifier Contract-Local Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate AutoVerifier output against one canonical schema, repair contract failures locally without rerunning Worker or Synthesis, and escalate exhausted failures as verifier infrastructure failures.

**Architecture:** Add strict Pydantic models for semantic assessments and verifier execution failures. The Verifier node invokes the full verification prompt once, then performs at most two structure-only repair calls for JSON parsing, schema, or consistency errors; only a valid assessment reaches deterministic content validation. Exhaustion is returned through a separate `verifier_failure` state field so DecisionPolicy cannot mix it with content issues.

**Tech Stack:** Python 3, Pydantic 2, LangChain JSON mode, LangGraph, pytest.

## Global Constraints

- Preserve the existing assessment business fields: `status`, `current_section`, `issues`, `requirements_met`, and `requirements_missing`.
- Allow two contract repair retries after the initial Verifier attempt.
- Contract repair must not rerun Worker, Synthesis, RAG, citation recovery, or plan patching.
- Contract repair may update only `verifier_retry_count`; it must not update content, asset, evidence, or plan counters and revisions.
- Do not change length thresholds, tolerance, or rewrite policy.
- Do not add dependencies or perform SQLite schema migrations.
- Do not log complete prompts, report bodies, or user-sensitive content.

---

### Task 1: Canonical assessment contract and local repair loop

**Files:**
- Create: `src/verifier_contract.py`
- Modify: `src/nodes/verifier.py`
- Test: `tests/test_auto_verifier_routing.py`

**Interfaces:**
- Produces: `VerifierAssessment.model_validate_json(...)`, `VerifierExecutionFailure`, and `AssessmentContractError`.
- Produces: `verifier(state, config) -> {"assessment": dict, "verifier_failure": dict, "verifier_retry_count"?: dict}`.

- [x] **Step 1: Write failing tests for invalid-then-PASS and invalid-then-TOO_SHORT.**

```python
def test_contract_failure_is_repaired_locally_before_pass(monkeypatch):
    responses = ["not-json", json.dumps(valid_pass)]
    update = verifier(state, config)
    assert update["assessment"]["status"] == "PASS"
    assert update["verifier_failure"] == {}
    assert update["verifier_retry_count"] == {"T1": 1}
    assert model.calls == 2

def test_repaired_semantic_length_failure_enters_policy_once(monkeypatch):
    responses = [json.dumps(invalid_contract), json.dumps(valid_failed_too_short)]
    update = verifier(state, config)
    decision = decide_recovery_action({**state, **update}, update["assessment"])
    assert decision["workflow_action"] == "LENGTH_REWRITE"
    assert decision["task_retry_count"] == {"T1": 1}
```

- [x] **Step 2: Run the focused tests and confirm RED because Verifier currently calls the model once and converts parsing failures to `LLM_ERROR`.**

Run: `.venv/bin/python -m pytest -q tests/test_auto_verifier_routing.py -k 'repaired_locally or repaired_semantic'`

- [x] **Step 3: Add strict Pydantic models and consistency validation.**

```python
class VerifierAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: AssessmentStatus
    current_section: str
    issues: list[AssessmentIssue]
    requirements_met: list[str]
    requirements_missing: list[str]

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.status is AssessmentStatus.PASS and (
            self.issues or self.requirements_missing
        ):
            raise ValueError("PASS assessment cannot contain failures")
        if self.status is not AssessmentStatus.PASS and not self.issues:
            raise ValueError("FAILED/BLOCKED assessment must contain issues")
        return self
```

- [x] **Step 4: Implement the bounded local loop.**

The first call uses the existing full prompt. On `JSONDecodeError` or Pydantic validation failure, invoke a short repair prompt containing only the bounded previous response, schema, and bounded validation error. Increment a task-ID keyed copy of `verifier_retry_count` for each repair. Do not apply citation or deterministic validation until a canonical assessment succeeds.

- [x] **Step 5: Run focused tests and confirm GREEN.**

Run: `.venv/bin/python -m pytest -q tests/test_auto_verifier_routing.py -k 'contract or repaired or service_error'`

### Task 2: Exhausted verifier failure and policy separation

**Files:**
- Modify: `src/state.py`
- Modify: `src/recovery/policy.py`
- Modify: `src/nodes/recovery.py`
- Test: `tests/test_auto_verifier_routing.py`
- Test: `tests/test_recovery_policy.py`
- Test: `tests/test_recovery_graph.py`

**Interfaces:**
- Adds state field: `verifier_failure: Dict[str, Any]`.
- Extends: `decide_recovery_action` to give active `state["verifier_failure"]` precedence over semantic assessment.
- Preserves compatibility with legacy `ASSESSMENT_CONTRACT_ERROR` assessment issues.

- [x] **Step 1: Write failing tests for exhausted contract retries, unchanged counters, and synthesis routing.**

```python
assert update["assessment"] == {}
assert update["verifier_failure"]["code"] == "VERIFIER_UNAVAILABLE"
assert update["verifier_retry_count"] == {"T7": 2}
assert decision["workflow_action"] == "NEEDS_USER_INPUT"
assert "worker_state" not in decision
assert content_counters(decision) == content_counters(state)
```

- [x] **Step 2: Run focused tests and confirm RED because no separate failure channel exists.**

Run: `.venv/bin/python -m pytest -q tests/test_auto_verifier_routing.py tests/test_recovery_policy.py tests/test_recovery_graph.py -k 'verifier_unavailable or contract_failure_does_not_rewrite or synthesis_contract'`

- [x] **Step 3: Add `verifier_failure` state and policy precedence.**

On a non-empty verifier failure, route directly to `NEEDS_USER_INPUT` when local retries are exhausted. Build blocker copy that explicitly says the automatic verifier could not produce a valid result. Never classify it as `CONTENT_DEFECT`, never attach `worker_state`, and never mutate content recovery counters.

- [x] **Step 4: Keep legacy checkpoint behavior bounded.**

Old assessments containing only `ASSESSMENT_CONTRACT_ERROR` may still use the existing graph-level `RETRY_VERIFIER` once. New Verifier executions clear `verifier_failure` on success and never emit `ASSESSMENT_CONTRACT_ERROR` as a semantic issue.

- [x] **Step 5: Run focused policy and graph tests and confirm GREEN.**

Run: `.venv/bin/python -m pytest -q tests/test_recovery_policy.py tests/test_recovery_graph.py tests/test_graph_state_contract.py`

### Task 3: Logging and regression verification

**Files:**
- Modify: `src/nodes/verifier.py`
- Test: `tests/test_auto_verifier_routing.py`

**Interfaces:**
- Logs bounded contract failure/retry/final-attempt metadata without prompt or report content.

- [x] **Step 1: Add a failing `caplog` test for attempt metadata and sanitized errors.**

- [x] **Step 2: Emit WARNING for validation failure, INFO before repair, and INFO for the final valid assessment.**

- [x] **Step 3: Run all requested focused suites.**

```bash
.venv/bin/python -m pytest -q tests/test_auto_verifier_routing.py
.venv/bin/python -m pytest -q tests/test_recovery_policy.py
.venv/bin/python -m pytest -q tests/test_recovery_graph.py
```

- [x] **Step 4: Run the full suite and diff checks.**

```bash
.venv/bin/python -m pytest -q
git diff --check
git status --short
```

- [x] **Step 5: Review the diff for scope, checkpoint compatibility, sensitive logs, and unchanged length policy.**

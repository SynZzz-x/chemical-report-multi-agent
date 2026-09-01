"""Offline budget/truncation regressions, not evidence of token sufficiency."""

import json
from types import SimpleNamespace

import pytest
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion

from src import config as config_module
from src.llm import ProviderTokenUsage, completion_budget, extract_truncation_token_usage
from src.nodes import planner as planner_module
from tests.measure_verifier_controls import run_verifier_control_probe
from tests.test_planner_contract import _intake_summary, _task


@pytest.fixture(autouse=True)
def isolated_planner_config(monkeypatch):
    monkeypatch.delenv("PLANNER_MAX_COMPLETION_TOKENS", raising=False)
    config_module.get_app_config.cache_clear()
    yield
    config_module.get_app_config.cache_clear()


def _truncated(usage=None):
    return LengthFinishReasonError(completion=ChatCompletion.model_validate({
        "id": "offline", "object": "chat.completion", "created": 0,
        "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "finish_reason": "length", "message": {
            "role": "assistant", "content": "PRIVATE_PARTIAL_COMPLETION",
        }}],
        "usage": usage,
    }))


def _model(monkeypatch, outcomes):
    class Model:
        model_name = "deepseek-v4-flash"

        def __init__(self):
            self.outcomes = iter(outcomes)
            self.calls = []

        def bind(self, **kwargs):
            self.request_options = kwargs
            return self

        def invoke(self, messages, **kwargs):
            self.calls.append(messages)
            outcome = next(self.outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(content=outcome)

    model = Model()
    monkeypatch.setattr(planner_module, "get_llm", lambda *a, **kw: model)
    monkeypatch.setattr(planner_module, "load_active_catalog", lambda: [])
    return model


def test_planner_default_budget_is_independent():
    assert config_module.get_app_config().planner_max_completion_tokens == 16384
    assert completion_budget("plan_generation") == 16384


def test_planner_budget_environment_override(monkeypatch):
    monkeypatch.setenv("PLANNER_MAX_COMPLETION_TOKENS", "20000")
    assert config_module.get_app_config().planner_max_completion_tokens == 20000
    assert completion_budget("plan_generation") == 20000


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_planner_budget_rejects_invalid_config(monkeypatch, value):
    monkeypatch.setenv("PLANNER_MAX_COMPLETION_TOKENS", value)
    with pytest.raises(ValueError, match="PLANNER_MAX_COMPLETION_TOKENS.*positive integer"):
        config_module.get_app_config()


@pytest.mark.parametrize("usage, expected", [
    (None, ("-", "-", "-", "-")),
    ({"prompt_tokens": 17, "completion_tokens": 41, "total_tokens": 58,
      "completion_tokens_details": {"reasoning_tokens": 29}}, ("17", "41", "29", "58")),
    ({"prompt_tokens": 17, "completion_tokens": 41, "total_tokens": 58}, ("17", "41", "-", "58")),
])
def test_sdk_truncation_classification_retry_count_and_safe_usage(monkeypatch, caplog, usage, expected):
    error = _truncated(usage)
    model = _model(monkeypatch, [error, error])
    with caplog.at_level("INFO"):
        with pytest.raises(planner_module.PlannerGenerationError) as raised:
            planner_module._build_tasks_with_llm(_intake_summary(), {})
    assert raised.value.__cause__ is error
    assert len(model.calls) == 2
    assert model.request_options == {"extra_body": {"max_tokens": 16384}}
    planner_logs = [r.message for r in caplog.records if r.name == "src.nodes.planner"]
    assert len(planner_logs) == 2
    for attempt, log in enumerate(planner_logs, 1):
        assert "reason=plan_generation_truncated" in log
        assert f"attempt={attempt}" in log
        assert "requested_max_completion_tokens=16384" in log
    error_logs = [m for m in caplog.messages if m.startswith("LLM_CALL_ERROR")]
    assert len(error_logs) == 2
    for log in error_logs:
        assert "error_type=LengthFinishReasonError" in log
        for field, value in zip(("prompt", "completion", "reasoning", "total"), expected):
            assert f"provider_{field}_tokens={value}" in log
    assert "plan_contract_invalid" not in caplog.text
    assert "PRIVATE_PARTIAL_COMPLETION" not in caplog.text
    assert "PRIVATE_PARTIAL_COMPLETION" not in str(model.calls)
    retry = model.calls[1][-1].content
    assert "截断" in retry
    assert "校验错误" not in retry
    assert "上一次输出未通过 Plan JSON 校验" not in retry


@pytest.mark.parametrize("payload", ["not-json", '{"tasks": []}', json.dumps({"tasks": [_task(use_rag="yes")]})])
def test_complete_invalid_response_keeps_contract_retry(monkeypatch, caplog, payload):
    model = _model(monkeypatch, [payload, json.dumps({"tasks": [_task()]})])
    with caplog.at_level("WARNING"):
        assert planner_module._build_tasks_with_llm(_intake_summary(), {}) == [_task()]
    assert len(model.calls) == 2
    assert "reason=plan_contract_invalid" in caplog.text
    assert "plan_generation_truncated" not in caplog.text
    assert "上一次输出未通过 Plan JSON 校验" in model.calls[1][-1].content


@pytest.mark.parametrize(
    ("payload", "intake", "stage", "code", "task_id", "error_type"),
    [
        (
            "not-json PRIVATE_MODEL_OUTPUT",
            _intake_summary(user_intent="PRIVATE_USER_REQUEST"),
            "json_envelope",
            "invalid_json",
            "-",
            "JSONDecodeError",
        ),
        (
            json.dumps({"tasks": [_task(task_id="T999999999", use_rag="yes")]}),
            _intake_summary(),
            "task_schema",
            "invalid_task_schema",
            "-",
            "ValueError",
        ),
        (
            json.dumps({"tasks": [_task(requirement_ids=["REQ-404"])]}),
            _intake_summary(requirements=[{"requirement_id": "REQ-001"}]),
            "requirement_dependency",
            "unknown_requirement",
            "T1",
            "ValueError",
        ),
        (
            json.dumps({"tasks": [
                _task(task_id="T1"),
                _task(task_id="T2", depends_on_task_ids=["T404"]),
            ]}),
            _intake_summary(sections=[]),
            "requirement_dependency",
            "invalid_dependency",
            "T2",
            "ValueError",
        ),
        (
            json.dumps({"tasks": [_task(use_rag=False, query="unexpected")]}),
            _intake_summary(),
            "task_semantics",
            "rag_query_consistency",
            "T1",
            "ValueError",
        ),
    ],
)
def test_contract_invalid_logging_uses_safe_validation_metadata(
    monkeypatch,
    caplog,
    payload,
    intake,
    stage,
    code,
    task_id,
    error_type,
):
    model = _model(monkeypatch, [payload, payload])

    with caplog.at_level("WARNING", logger="src.nodes.planner"):
        with pytest.raises(planner_module.PlannerGenerationError):
            planner_module._build_tasks_with_llm(intake, {})

    logs = [record.message for record in caplog.records if record.name == "src.nodes.planner"]
    assert len(logs) == len(model.calls) == 2
    for log in logs:
        assert "reason=plan_contract_invalid" in log
        assert f"error_type={error_type}" in log
        assert f"validation_stage={stage}" in log
        assert f"validation_code={code}" in log
        assert f"task_id={task_id}" in log
        assert "PRIVATE_MODEL_OUTPUT" not in log
        assert "PRIVATE_USER_REQUEST" not in log
        assert "T999999999" not in log

    if error_type == "JSONDecodeError":
        assert "Expecting value" in model.calls[1][-1].content


def test_contract_invalid_telemetry_does_not_prevent_recovery(monkeypatch, caplog):
    invalid = json.dumps({"tasks": [_task(use_rag=False, query="unexpected")]})
    model = _model(monkeypatch, [invalid, json.dumps({"tasks": [_task()]})])

    with caplog.at_level("WARNING", logger="src.nodes.planner"):
        assert planner_module._build_tasks_with_llm(_intake_summary(), {}) == [_task()]

    logs = [record.message for record in caplog.records if record.name == "src.nodes.planner"]
    assert len(logs) == 1
    assert "reason=plan_contract_invalid" in logs[0]
    assert "validation_stage=task_semantics" in logs[0]
    assert "validation_code=rag_query_consistency" in logs[0]
    assert "task_id=T1" in logs[0]
    assert len(model.calls) == 2


def test_truncation_log_does_not_reuse_prior_contract_validation_metadata(
    monkeypatch,
    caplog,
):
    invalid = json.dumps({"tasks": [_task(use_rag=False, query="unexpected")]})
    model = _model(monkeypatch, [invalid, _truncated()])

    with caplog.at_level("WARNING", logger="src.nodes.planner"):
        with pytest.raises(planner_module.PlannerGenerationError):
            planner_module._build_tasks_with_llm(_intake_summary(), {})

    logs = [record.message for record in caplog.records if record.name == "src.nodes.planner"]
    assert len(logs) == len(model.calls) == 2
    assert "reason=plan_contract_invalid" in logs[0]
    assert "validation_stage=task_semantics" in logs[0]
    assert "reason=plan_generation_truncated" in logs[1]
    assert "validation_stage=" not in logs[1]
    assert "validation_code=" not in logs[1]
    assert "task_id=" not in logs[1]


def test_truncation_can_retry_to_valid_plan_without_changing_contract(monkeypatch, caplog):
    monkeypatch.setenv("PLANNER_MAX_COMPLETION_TOKENS", "20000")
    model = _model(monkeypatch, [_truncated(), json.dumps({"tasks": [_task()]})])
    with caplog.at_level("INFO"):
        assert planner_module._build_tasks_with_llm(_intake_summary(), {}) == [_task()]
    starts = [m for m in caplog.messages if m.startswith("LLM_CALL_START")]
    assert len(starts) == len(model.calls) == 2
    assert all("requested_max_completion_tokens=20000" in m for m in starts)
    assert model.request_options == {"extra_body": {"max_tokens": 20000}}
    assert "截断" in model.calls[1][-1].content


def test_request_failure_is_not_a_contract_failure(monkeypatch, caplog):
    model = _model(monkeypatch, [RuntimeError("PRIVATE_TRANSPORT_DETAIL")] * 2)
    with caplog.at_level("WARNING"):
        with pytest.raises(planner_module.PlannerGenerationError):
            planner_module._build_tasks_with_llm(_intake_summary(), {})
    assert len(model.calls) == 2
    assert "reason=plan_generation_failed" in caplog.text
    assert "plan_contract_invalid" not in caplog.text
    assert "PRIVATE_TRANSPORT_DETAIL" not in caplog.text
    assert "校验错误" not in model.calls[1][-1].content


@pytest.mark.parametrize("usage, expected", [
    ({"prompt_tokens": 17, "completion_tokens": 41},
     ProviderTokenUsage(provider_prompt_tokens=17, provider_completion_tokens=41)),
    ({"prompt_tokens": "17", "completion_tokens": True, "total_tokens": "58",
      "completion_tokens_details": {"reasoning_tokens": "29"}}, ProviderTokenUsage()),
    ({"prompt_tokens": 17, "completion_tokens_details": "invalid"},
     ProviderTokenUsage(provider_prompt_tokens=17)),
])
def test_partial_usage_stays_numeric_without_inferred_totals(usage, expected):
    error = _truncated()
    # The SDK documents that exception completions may be partial (e.g. streams).
    error.completion = SimpleNamespace(usage=usage)
    assert extract_truncation_token_usage(error) == expected


def test_usage_extraction_failure_never_replaces_original_error(monkeypatch, caplog):
    class BrokenUsage:
        def model_dump(self):
            raise ValueError("PRIVATE_USAGE_DETAIL")

    error = _truncated()
    error.completion = SimpleNamespace(usage=BrokenUsage())
    model = _model(monkeypatch, [error, error])
    with caplog.at_level("WARNING"):
        with pytest.raises(planner_module.PlannerGenerationError) as raised:
            planner_module._build_tasks_with_llm(_intake_summary(), {})
    assert raised.value.__cause__ is error
    assert len(model.calls) == 2
    assert "error_type=LengthFinishReasonError" in caplog.text
    assert "provider_total_tokens=-" in caplog.text
    assert "PRIVATE_USAGE_DETAIL" not in caplog.text


@pytest.mark.parametrize("override, expected", [(None, 16384), ("20000", 20000)])
@pytest.mark.parametrize("bind_budget", [False, True])
def test_actual_planner_request_uses_configured_max_tokens(override, expected, bind_budget):
    env = {} if override is None else {"PLANNER_MAX_COMPLETION_TOKENS": override}
    payload = run_verifier_control_probe(env, purpose="plan_generation", apply_completion_budget=bind_budget)
    assert payload["max_tokens"] == expected
    assert payload["max_completion_tokens"] is None
    assert payload["reasoning_effort"] is None
    assert payload["thinking_present"] is False


@pytest.mark.parametrize("purpose, expected", [
    ("canonical_intake_generation", 1200), ("task_generation", 3200),
    ("assessment", 8192), ("report_synthesis", 4000),
])
def test_other_actual_request_budgets_unchanged_by_planner_override(purpose, expected):
    payload = run_verifier_control_probe({"PLANNER_MAX_COMPLETION_TOKENS": "20000"}, purpose=purpose)
    assert payload["max_tokens"] == expected
    assert payload["max_completion_tokens"] is None
    assert payload["reasoning_effort"] is None
    assert payload["thinking_present"] is False

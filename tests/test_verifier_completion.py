"""Offline regressions for verifier completion budgeting and truncation handling."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion

from src import config as config_module
from src.llm import completion_budget
from src.nodes import verifier as verifier_module
from src.recovery.policy import decide_recovery_action
from tests.measure_verifier_controls import run_verifier_control_probe
from tests.test_auto_verifier_routing import _state


@pytest.fixture(autouse=True)
def isolated_verifier_config(monkeypatch):
    monkeypatch.delenv("VERIFIER_MAX_COMPLETION_TOKENS", raising=False)
    config_module.get_app_config.cache_clear()
    yield
    config_module.get_app_config.cache_clear()


def _truncated(usage=None):
    return LengthFinishReasonError(
        completion=ChatCompletion.model_validate(
            {
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "PRIVATE_PARTIAL_COMPLETION",
                        },
                    }
                ],
                "usage": usage,
            }
        )
    )


def _model(monkeypatch, outcomes):
    class Model:
        model_name = "deepseek-v4-flash"

        def __init__(self):
            self.outcomes = iter(outcomes)
            self.calls = []

        def bind(self, **kwargs):
            self.request_options = kwargs
            return self

        def invoke(self, payload, **kwargs):
            self.calls.append((payload, kwargs))
            outcome = next(self.outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(content=outcome)

    model = Model()
    monkeypatch.setattr(verifier_module, "get_llm", lambda *args, **kwargs: model)
    return model


def test_verifier_assessment_budget_defaults_and_overrides(monkeypatch):
    assert config_module.get_app_config().verifier_max_completion_tokens == 8192
    assert completion_budget("assessment") == 8192

    monkeypatch.setenv("VERIFIER_MAX_COMPLETION_TOKENS", "10000")
    config_module.get_app_config.cache_clear()

    assert config_module.get_app_config().verifier_max_completion_tokens == 10000
    assert completion_budget("assessment") == 10000


@pytest.mark.parametrize("override, expected", [(None, 8192), ("10000", 10000)])
def test_actual_verifier_request_uses_configured_max_tokens(override, expected):
    env = {} if override is None else {"VERIFIER_MAX_COMPLETION_TOKENS": override}

    payload = run_verifier_control_probe(env, purpose="assessment")

    assert payload["max_tokens"] == expected
    assert payload["max_completion_tokens"] is None
    assert payload["reasoning_effort"] is None
    assert payload["thinking_present"] is False


@pytest.mark.parametrize(
    ("purpose", "expected"),
    [
        ("canonical_intake_generation", 1200),
        ("assessment_contract_repair", 900),
        ("plan_generation", 16384),
        ("task_generation", 3200),
        ("report_synthesis", 4000),
    ],
)
def test_verifier_budget_override_does_not_change_other_actual_requests(
    purpose, expected
):
    payload = run_verifier_control_probe(
        {"VERIFIER_MAX_COMPLETION_TOKENS": "10000"}, purpose=purpose
    )

    assert payload["max_tokens"] == expected
    assert payload["max_completion_tokens"] is None


def test_verifier_truncation_is_execution_failure_not_semantic_and_exhausts_retry(
    monkeypatch, caplog
):
    state = _state()
    model = _model(monkeypatch, [_truncated(), _truncated()])

    with caplog.at_level("INFO"):
        first_update = verifier_module.verifier(
            state, {"configurable": {"use_llm": True}}
        )
        first_decision = decide_recovery_action(
            {**state, **first_update}, first_update["assessment"]
        )
        second_state = {**state, **first_update, **first_decision}
        second_update = verifier_module.verifier(
            second_state, {"configurable": {"use_llm": True}}
        )
        second_decision = decide_recovery_action(
            {**second_state, **second_update}, second_update["assessment"]
        )

    assert len(model.calls) == 2
    assert model.request_options == {"extra_body": {"max_tokens": 8192}}
    assert first_update["assessment"] == second_update["assessment"] == {}
    assert first_update["verifier_failure"] == second_update["verifier_failure"] == {
        "code": "LLM_ERROR",
        "category": "VERIFIER_FAILURE",
        "message": "自动校验服务未能返回可用的结构化结果。",
        "retryable": True,
        "contract_attempts": 1,
    }
    assert first_decision["workflow_action"] == "RETRY_VERIFIER"
    assert first_decision["verifier_retry_count"] == {"T1": 1}
    assert second_decision["workflow_action"] == "FATAL_SYSTEM"
    assert second_decision["pending_user_action"] == {}
    assert "AutoVerifier contract validation failed" not in caplog.text
    assert "PRIVATE_PARTIAL_COMPLETION" not in caplog.text
    truncation_logs = [
        record.message
        for record in caplog.records
        if "reason=verifier_generation_truncated" in record.message
    ]
    assert len(truncation_logs) == 2
    for message in truncation_logs:
        assert "task=T1" in message
        assert "attempt=1" in message
        assert "purpose=assessment" in message
        assert "requested_max_completion_tokens=8192" in message
        assert "error_type=LengthFinishReasonError" in message
    error_logs = [
        record.message
        for record in caplog.records
        if record.message.startswith("LLM_CALL_ERROR")
    ]
    assert len(error_logs) == 2
    for message in error_logs:
        for field in (
            "provider_prompt_tokens=-",
            "provider_completion_tokens=-",
            "provider_reasoning_tokens=-",
            "provider_total_tokens=-",
        ):
            assert field in message


def test_contract_repair_truncation_logs_repair_budget_and_purpose(monkeypatch, caplog):
    model = _model(monkeypatch, ["not-json", _truncated()])

    with caplog.at_level("INFO"):
        update = verifier_module.verifier(
            _state(), {"configurable": {"use_llm": True}}
        )

    assert len(model.calls) == 2
    assert update["assessment"] == {}
    assert update["verifier_failure"]["code"] == "LLM_ERROR"
    truncation_logs = [
        record.message
        for record in caplog.records
        if "reason=verifier_generation_truncated" in record.message
    ]
    assert len(truncation_logs) == 1
    for field in (
        "task=T1",
        "attempt=2",
        "purpose=assessment_contract_repair",
        "requested_max_completion_tokens=900",
        "error_type=LengthFinishReasonError",
    ):
        assert field in truncation_logs[0]
    assert "PRIVATE_PARTIAL_COMPLETION" not in caplog.text


@pytest.mark.parametrize("status", ["PASS", "FAILED"])
def test_complete_verifier_assessments_keep_existing_single_call_semantics(
    monkeypatch, status
):
    issue = []
    if status == "FAILED":
        issue = [
            {
                "code": "TOO_SHORT",
                "category": "CONTENT_DEFECT",
                "description": "内容过短",
                "suggestion": "扩写",
                "severity": "major",
            }
        ]
    model = _model(
        monkeypatch,
        [
            json.dumps(
                {
                    "status": status,
                    "current_section": "引言",
                    "issues": issue,
                    "requirements_met": [],
                    "requirements_missing": [],
                },
                ensure_ascii=False,
            )
        ],
    )

    update = verifier_module.verifier(
        _state(), {"configurable": {"use_llm": True}}
    )

    assert len(model.calls) == 1
    assert update["verifier_failure"] == {}
    assert update["assessment"]["status"] == status

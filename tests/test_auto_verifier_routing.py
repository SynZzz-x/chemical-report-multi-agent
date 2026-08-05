import json
from types import SimpleNamespace

import pytest

from src.nodes import verifier as auto_verifier_module
from src.recovery.policy import decide_recovery_action


def _state(*, cursor=0):
    tasks = [
        {
            "task_id": "T1",
            "task_name": "引言",
            "task_description": "撰写简短引言。",
            "generate_figure": False,
            "generate_table": False,
        },
        {
            "task_id": "T2",
            "task_name": "质量指标体系",
            "task_description": "必须生成质量指标表格。",
            "generate_figure": False,
            "generate_table": True,
        },
    ]
    current = tasks[cursor]
    return {
        "tasks": tasks,
        "cursor": cursor,
        "current_result": {
            "task_id": current["task_id"],
            "status": "COMPLETED",
            "text_output": f"{current['task_name']}正文",
            "tables": [],
            "figures": [],
            "citations": [{"evidence_id": "E1"}],
            "sources_used": ["process.docx"],
        },
        "results": [{"task_id": "T0", "text_output": "accepted"}],
        "messages": [],
        "task_retry_count": {"T1": 1},
        "replan_count": 1,
        "plan_revision": 4,
    }


def _run(monkeypatch, state, assessment):
    captured = {}

    class Model:
        def invoke(self, payload):
            captured.update(payload)
            return SimpleNamespace(content=json.dumps(assessment, ensure_ascii=False))

    monkeypatch.setattr(auto_verifier_module, "get_llm", lambda *args, **kwargs: Model())
    update = auto_verifier_module.verifier(
        state,
        {"configurable": {"use_llm": True}},
    )
    return update, captured


def test_verifier_is_assessment_only_and_classifies_evidence_gap(monkeypatch):
    assessment = {
        "status": "BLOCKED",
        "current_section": "引言",
        "issues": [
            {
                "code": "MISSING_EVIDENCE",
                "category": "EVIDENCE_GAP",
                "description": "关键结论缺少知识库依据",
                "suggestion": "扩大检索覆盖并补充引用",
                "severity": "major",
            }
        ],
        "requirements_met": ["包含背景"],
        "requirements_missing": ["关键结论来源"],
        "recommended_decision": "REPLAN",
    }

    update, _ = _run(monkeypatch, _state(), assessment)

    assert set(update) == {"assessment"}
    assert "decision" not in update
    assert "recommended_decision" not in update["assessment"]
    assert update["assessment"]["issues"][0]["category"] == "EVIDENCE_GAP"
    assert update["assessment"]["requirements_missing"] == ["关键结论来源"]


def test_sanitizer_checks_current_task_requirements_not_first_task():
    state = _state(cursor=1)
    assessment = {
        "status": "FAILED",
        "current_section": "质量指标体系",
        "issues": [
            {
                "code": "MISSING_TABLE",
                "category": "CONTENT_DEFECT",
                "description": "缺少任务要求的表格",
                "suggestion": "补充表格",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["质量指标表格"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)

    assert sanitized["status"] == "FAILED"
    assert sanitized["issues"][0]["code"] == "MISSING_TABLE"


def test_verifier_receives_full_task_and_asset_context(monkeypatch):
    assessment = {
        "status": "PASS",
        "current_section": "质量指标体系",
        "issues": [],
        "requirements_met": ["质量指标表格"],
        "requirements_missing": [],
    }

    _, captured = _run(monkeypatch, _state(cursor=1), assessment)

    assert "必须生成质量指标表格" in captured["task_requirements"]
    assets = json.loads(captured["worker_assets"])
    assert assets["citations"] == [{"evidence_id": "E1"}]
    assert assets["tables"] == []


def test_sanitizer_rejects_pass_assessment_that_still_contains_issues():
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [
            {
                "code": "TOO_SHORT",
                "category": "CONTENT_DEFECT",
                "description": "内容过短",
                "suggestion": "扩写正文",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["正文深度"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, _state())

    assert sanitized["status"] == "FAILED"


def test_sanitizer_keeps_malformed_failed_assessment_from_becoming_pass():
    assessment = {
        "status": "BLOCKED",
        "current_section": "引言",
        "issues": [None, 42],
        "requirements_met": [],
        "requirements_missing": ["外部依赖"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, _state())

    assert sanitized["status"] == "FAILED"
    assert sanitized["issues"][0]["code"] == "ASSESSMENT_CONTRACT_ERROR"


@pytest.mark.parametrize("code", ["LLM_ERROR", "LLM_NOT_ENABLED"])
def test_verifier_service_failures_require_user_input_without_consuming_retry(
    monkeypatch, code
):
    state = _state()
    original_results = list(state["results"])
    if code == "LLM_ERROR":
        class FailingModel:
            def invoke(self, payload):
                raise RuntimeError("verification unavailable")

        monkeypatch.setattr(
            auto_verifier_module,
            "get_llm",
            lambda *args, **kwargs: FailingModel(),
        )
        config = {"configurable": {"use_llm": True}}
    else:
        monkeypatch.setattr(
            auto_verifier_module,
            "get_app_config",
            lambda: SimpleNamespace(deepseek_api_key=None),
        )
        config = {"configurable": {"use_llm": False}}

    sanitized = auto_verifier_module.verifier(state, config)["assessment"]
    update = decide_recovery_action(state, sanitized)

    assert sanitized["issues"][0]["category"] == "EXTERNAL_BLOCKER"
    assert update["workflow_action"] == "NEEDS_USER_INPUT"
    assert update["task_retry_count"] == {"T1": 1}
    assert "results" not in update
    assert state["results"] == original_results

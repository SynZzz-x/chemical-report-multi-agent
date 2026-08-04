import json
from types import SimpleNamespace

from src.nodes import verifier as auto_verifier_module


def _state(*, cursor=0, retry_count=None, replan_count=0):
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
        "results": [],
        "messages": [],
        "task_retry_count": dict(retry_count or {}),
        "replan_count": replan_count,
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


def test_sanitizer_checks_current_task_requirements_not_first_task():
    state = _state(cursor=1)
    assessment = {
        "status": "FAILED",
        "current_section": "质量指标体系",
        "issues": [
            {
                "code": "MISSING_TABLE",
                "description": "缺少任务要求的表格",
                "suggestion": "补充表格",
            }
        ],
        "recommended_decision": "RETRY_WORKER",
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)

    assert sanitized["status"] == "FAILED"
    assert sanitized["issues"][0]["code"] == "MISSING_TABLE"


def test_content_problem_cannot_trigger_replan(monkeypatch):
    assessment = {
        "status": "FAILED",
        "current_section": "引言",
        "issues": [
            {
                "code": "TOO_SHORT",
                "description": "内容过短",
                "suggestion": "扩写正文",
            }
        ],
        "recommended_decision": "REPLAN",
    }

    update, _ = _run(monkeypatch, _state(), assessment)

    assert update["decision"] == "RETRY_WORKER"
    assert update["feedback"] == update["assessment"]
    assert update["replan_count"] == 0


def test_retry_limit_advances_with_warning_instead_of_replanning(monkeypatch):
    assessment = {
        "status": "FAILED",
        "current_section": "引言",
        "issues": [
            {
                "code": "TOO_SHORT",
                "description": "仍然过短",
                "suggestion": "扩写正文",
            }
        ],
        "recommended_decision": "RETRY_WORKER",
    }

    update, _ = _run(monkeypatch, _state(retry_count={0: 2}), assessment)

    assert update["decision"] == "NEXT"
    assert update["results"][0]["task_id"] == "T1"
    assert any(
        issue["code"] == "AUTO_RETRY_LIMIT_REACHED"
        for issue in update["assessment"]["issues"]
    )
    message = json.loads(update["messages"][-1].content)
    assert message["type"] == "PROCEED"


def test_explicit_plan_defect_can_replan_only_once(monkeypatch):
    assessment = {
        "status": "BLOCKED",
        "current_section": "引言",
        "issues": [
            {
                "code": "MISSING_RESOURCE",
                "description": "缺少任务必须的数据源",
                "suggestion": "调整任务计划",
            }
        ],
        "recommended_decision": "REPLAN",
    }

    first, _ = _run(monkeypatch, _state(), assessment)
    assert first["decision"] == "REPLAN"
    assert first["replan_count"] == 1
    assert first["feedback"] == first["assessment"]

    capped, _ = _run(
        monkeypatch,
        _state(retry_count={0: 2}, replan_count=1),
        assessment,
    )
    assert capped["decision"] == "NEXT"
    assert capped["replan_count"] == 1


def test_verifier_receives_full_task_and_asset_context(monkeypatch):
    assessment = {
        "status": "PASS",
        "current_section": "质量指标体系",
        "issues": [],
        "recommended_decision": "NEXT",
    }

    _, captured = _run(monkeypatch, _state(cursor=1), assessment)

    assert "必须生成质量指标表格" in captured["task_requirements"]
    assets = json.loads(captured["worker_assets"])
    assert assets["citations"] == [{"evidence_id": "E1"}]
    assert assets["tables"] == []

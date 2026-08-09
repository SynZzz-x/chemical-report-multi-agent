import json

import pytest

from src.nodes import verifier_manual as manual_verifier_module
from src.nodes.planner import planner
from src.nodes.task_controller import task_controller
from src.nodes.verifier_manual import decision, verifier_manual


def test_decision_returns_state_decision():
    assert decision({"decision": "NEXT"}, {}) == "NEXT"
    assert decision({"decision": "FULL_REPLAN"}, {}) == "FULL_REPLAN"
    assert decision({"decision": "REWORK"}, {}) == "REWORK"
    assert decision({"decision": "DONE"}, {}) == "DONE"


def test_decision_defaults_to_done():
    assert decision({}, {}) == "DONE"


def _manual_state(*, cursor=0, results=None):
    tasks = [
        {"task_id": "T1", "task_name": "任务一"},
        {"task_id": "T2", "task_name": "任务二"},
    ]
    current_result = {
        "task_id": tasks[cursor]["task_id"],
        "artifact_id": f"A-{tasks[cursor]['task_id']}",
        "text_output": f"{tasks[cursor]['task_name']}的结果",
    }
    return {
        "tasks": tasks,
        "cursor": cursor,
        "current_result": current_result,
        "results": list(results or []),
        "messages": [],
        "active_artifact_ids": {
            tasks[cursor]["task_id"]: current_result["artifact_id"]
        },
        "review_records": [],
        "task_records": {
            task["task_id"]: {
                "task_id": task["task_id"],
                "sequence": index,
                "status": "RUNNING" if index == cursor else "PENDING",
                "attempt_count": 1 if index == cursor else 0,
                "active_artifact_id": (
                    current_result["artifact_id"] if index == cursor else None
                ),
                "dependencies": [],
            }
            for index, task in enumerate(tasks)
        },
    }


@pytest.mark.parametrize("feedback_text", ["继续工作", "通过"])
def test_manual_approval_bypasses_llm_and_advances_to_next_task(
    monkeypatch,
    feedback_text,
):
    state = _manual_state()
    analyzer_called = False

    def unexpected_analyzer(*args, **kwargs):
        nonlocal analyzer_called
        analyzer_called = True
        return {"decision": "REWORK", "reason": "不应调用", "suggestions": ""}

    monkeypatch.setattr(
        manual_verifier_module,
        "interrupt",
        lambda payload: {"text": feedback_text, "message_id": "feedback-1", "docs": []},
    )
    monkeypatch.setattr(
        manual_verifier_module,
        "_analyze_feedback",
        unexpected_analyzer,
    )

    verifier_update = verifier_manual(state, {})

    assert analyzer_called is False
    assert verifier_update["decision"] == "NEXT"
    assert verifier_update["task_records"]["T1"]["status"] == "PASSED"
    assert verifier_update["review_record"]["reviewer"] == "human"
    assert verifier_update["review_record"]["artifact_id"] == "A-T1"
    control_message = json.loads(verifier_update["messages"][-1].content)
    assert control_message["type"] == "PROCEED"

    controller_update = task_controller({**state, **verifier_update}, {})
    assert controller_update["cursor"] == 1
    assert controller_update["current_task"]["task_id"] == "T2"


@pytest.mark.parametrize("decision_alias", ["NEXT", "CONTINUE"])
def test_manual_pass_alias_emits_proceed_control_message(
    monkeypatch,
    decision_alias,
):
    state = _manual_state()
    monkeypatch.setattr(
        manual_verifier_module,
        "interrupt",
        lambda payload: {"text": "好的", "message_id": "feedback-2", "docs": []},
    )
    monkeypatch.setattr(
        manual_verifier_module,
        "_analyze_feedback",
        lambda *args, **kwargs: {
            "decision": decision_alias,
            "reason": "用户要求继续",
            "suggestions": "",
        },
    )

    verifier_update = verifier_manual(state, {})

    assert verifier_update["decision"] == "NEXT"
    control_message = json.loads(verifier_update["messages"][-1].content)
    assert control_message["type"] == "PROCEED"


def test_manual_pass_does_not_duplicate_an_accepted_task_result(monkeypatch):
    accepted_result = {"task_id": "T1", "text_output": "任务一的结果"}
    state = _manual_state(results=[accepted_result])
    state["current_result"] = dict(accepted_result)
    monkeypatch.setattr(
        manual_verifier_module,
        "interrupt",
        lambda payload: {"text": "通过", "message_id": "feedback-3", "docs": []},
    )
    monkeypatch.setattr(
        manual_verifier_module,
        "_analyze_feedback",
        lambda *args, **kwargs: {
            "decision": "PASS",
            "reason": "用户确认",
            "suggestions": "",
        },
    )

    verifier_update = verifier_manual(state, {})

    assert verifier_update["results"] == [accepted_result]


def test_manual_pass_on_last_task_returns_to_controller_for_summary(monkeypatch):
    state = _manual_state(cursor=1)
    state["task_records"]["T1"]["status"] = "PASSED"
    monkeypatch.setattr(
        manual_verifier_module,
        "interrupt",
        lambda payload: {"text": "通过", "message_id": "feedback-4", "docs": []},
    )

    verifier_update = verifier_manual(state, {})

    assert verifier_update["decision"] == "NEXT"
    control_message = json.loads(verifier_update["messages"][-1].content)
    assert control_message["type"] == "PROCEED"
    assert verifier_update["results"] == [state["current_result"]]

    controller_update = task_controller({**state, **verifier_update}, {})
    assert controller_update["controller_action"] == "SUMMARIZE"


def test_manual_rework_keeps_cursor_result_unaccepted(monkeypatch):
    state = _manual_state()
    monkeypatch.setattr(
        manual_verifier_module,
        "interrupt",
        lambda payload: {
            "text": "内容缺少依据，请返工",
            "message_id": "feedback-5",
            "docs": [],
        },
    )
    monkeypatch.setattr(
        manual_verifier_module,
        "_analyze_feedback",
        lambda *args, **kwargs: {
            "decision": "REWORK",
            "reason": "缺少依据",
            "suggestions": "补充来源",
        },
    )

    verifier_update = verifier_manual(state, {})

    assert verifier_update["decision"] == "REWORK"
    assert verifier_update["task_records"]["T1"]["status"] == "REVISE_REQUIRED"
    assert verifier_update["results"] == []
    control_message = json.loads(verifier_update["messages"][-1].content)
    assert control_message["type"] == "REWORK"

    controller_update = task_controller({**state, **verifier_update}, {})
    assert controller_update["controller_action"] == "DISPATCH"
    assert controller_update["task_records"]["T1"]["attempt_count"] == 2


def test_manual_feedback_analysis_failure_never_defaults_to_pass(monkeypatch):
    state = _manual_state()
    monkeypatch.setattr(
        manual_verifier_module,
        "interrupt",
        lambda payload: {
            "text": "内容错误，请返工",
            "message_id": "feedback-failure",
            "docs": [],
        },
    )
    monkeypatch.setattr(
        manual_verifier_module,
        "get_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    update = verifier_manual(state, {})

    assert update["decision"] == "REWORK"
    assert update["task_records"]["T1"]["status"] == "REVISE_REQUIRED"
    assert update["results"] == []


def test_unknown_manual_decision_is_fail_closed():
    assert manual_verifier_module._normalize_decision("MAYBE") == "REWORK"


def test_manual_full_replan_preserves_results_and_routes_to_planner(monkeypatch):
    accepted_result = {"task_id": "T0", "text_output": "已完成结果"}
    state = _manual_state(results=[accepted_result])
    monkeypatch.setattr(
        manual_verifier_module,
        "interrupt",
        lambda payload: {
            "text": "当前章节方向不对，请重新规划",
            "message_id": "feedback-6",
            "docs": [],
        },
    )
    monkeypatch.setattr(
        manual_verifier_module,
        "_analyze_feedback",
        lambda *args, **kwargs: {
            "decision": "REPLAN",
            "reason": "方向错误",
            "suggestions": "调整章节结构",
        },
    )

    verifier_update = verifier_manual(state, {})

    assert verifier_update["decision"] == "FULL_REPLAN"
    assert verifier_update["results"] == [accepted_result]
    control_message = json.loads(verifier_update["messages"][-1].content)
    assert control_message["type"] == "FULL_REPLAN"
    assert control_message["to"] == "Planner"

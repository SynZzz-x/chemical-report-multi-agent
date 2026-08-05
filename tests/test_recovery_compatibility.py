import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.nodes import planner as planner_module
from src.nodes.planner import planner, planner_confirm
from src.nodes.recovery import automatic_planner
from src.recovery.policy import decide_recovery_action


def _full_replan_state():
    return {
        "tasks": [
            {"task_id": "T1", "task_name": "旧任务一"},
            {"task_id": "T2", "task_name": "旧任务二"},
        ],
        "cursor": 1,
        "current_result": {"task_id": "T2", "text_output": "待替换结果"},
        "results": [{"task_id": "T1", "text_output": "已接受结果"}],
        "all_results": [{"task_id": "T1", "text_output": "全部结果"}],
        "task_retry_count": {"T2": 2},
        "evidence_recovery_count": {"T2": 1},
        "task_patch_count": {"T2": 1},
        "job_patch_count": 2,
        "replan_count": 4,
        "task_revisions": {"T1": 1, "T2": 2},
        "plan_revision": 3,
        "plan_patch_history": [{"type": "PLAN_PATCH", "plan_revision": 3}],
        "worker_state": {"execution_feedback": {"query": "stale", "allow_web": True}},
        "current_task": {"task_id": "T2", "query": "stale"},
        "tool_execution_history": [{"tool": "web_search"}],
        "feedback": {"status": "BLOCKED"},
        "assessment": {"status": "FAILED"},
        "workflow_action": "EVIDENCE_RECOVERY",
        "continuation_action": "NEXT",
        "verification_warning": {"reason": "stale"},
        "pending_user_action": {"category": "EXTERNAL_BLOCKER"},
        "verification_warnings": [{"task_id": "T2"}],
        "guidance": {"natural_language_guidance": "stale"},
        "final_result": {"path": "/old/report.docx"},
        "messages": [
            AIMessage(
                content=json.dumps(
                    {
                        "from": "Verifier",
                        "to": "Planner",
                        "type": "FULL_REPLAN",
                    }
                )
            )
        ],
        "decision": "FULL_REPLAN",
    }


def test_full_replan_commits_new_revision_only_after_confirmation(monkeypatch):
    state = _full_replan_state()
    replacement_tasks = [{"task_id": "T3", "task_name": "新任务"}]
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: replacement_tasks,
    )
    monkeypatch.setattr(
        planner_module,
        "_generate_plan_guidance",
        lambda *_: {"natural_language_guidance": "请确认", "resource_mapping": {}},
    )

    staged = planner(state, {})

    assert staged["planner_action"] == "FULL_REPLAN"
    assert staged["tasks"] == replacement_tasks
    assert staged["cursor"] == 1
    assert "results" not in staged
    assert "task_retry_count" not in staged
    assert "plan_revision" not in staged

    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"text": "确认", "docs": []},
    )
    committed = planner_confirm({**state, **staged}, {})

    assert committed["cursor"] == 0
    assert committed["plan_revision"] == 4
    assert committed["current_result"] == {}
    assert committed["results"] == []
    assert committed["all_results"] == []
    assert committed["task_retry_count"] == {}
    assert committed["evidence_recovery_count"] == {}
    assert committed["task_patch_count"] == {}
    assert committed["job_patch_count"] == 0
    assert committed["replan_count"] == 0
    assert committed["task_revisions"] == {"T3": 1}
    assert committed["worker_state"] == {}
    assert committed["current_task"] == {}
    assert committed["tool_execution_history"] == []
    assert committed["feedback"] == {}
    assert committed["assessment"] == {}
    assert committed["workflow_action"] == ""
    assert committed["continuation_action"] == ""
    assert committed["verification_warning"] == {}
    assert committed["pending_user_action"] == {}
    assert committed["verification_warnings"] == []
    assert committed["guidance"] == {}
    assert committed["final_result"] == {}
    assert committed["decision"] == "NEXT"
    assert committed["plan_patch_history"][-1] == {
        "type": "FULL_REPLAN",
        "previous_plan_revision": 3,
        "new_plan_revision": 4,
        "reason": "",
        "old_task_ids": ["T1", "T2"],
        "new_task_ids": ["T3"],
    }
    assert committed["plan_patch_history"][0] == {"type": "PLAN_PATCH", "plan_revision": 3}


def test_initial_plan_confirmation_does_not_start_a_new_revision(monkeypatch):
    monkeypatch.setattr(planner_module, "interrupt", lambda _: {"text": "确认", "docs": []})
    state = {
        "tasks": [{"task_id": "T1", "task_name": "初始任务"}],
        "cursor": 0,
        "plan_revision": 1,
        "planner_action": "INTAKE_SUMMARY",
        "messages": [],
    }

    update = planner_confirm(state, {})

    assert update["cursor"] == 0
    assert "plan_revision" not in update
    assert "results" not in update


def test_automatic_planner_filters_full_replan_controls(monkeypatch):
    full_replan = AIMessage(
        content=json.dumps(
            {"from": "Verifier", "to": "Planner", "type": "FULL_REPLAN"}
        )
    )
    proceed = AIMessage(content=json.dumps({"to": "Planner", "type": "PROCEED"}))
    state = {
        "tasks": [{"task_id": "T1"}, {"task_id": "T2"}, {"task_id": "T3"}],
        "cursor": 1,
        "decision": "FULL_REPLAN",
        "messages": [proceed, full_replan],
    }

    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: (_ for _ in ()).throw(AssertionError("automatic full replan")),
    )

    update = automatic_planner(state, {})

    assert update["planner_action"] == "PROCEED"
    assert update["cursor"] == 2


def test_legacy_checkpoint_uses_safe_recovery_defaults_without_full_replan():
    legacy_state = {
        "tasks": [{"task_id": "T1", "task_name": "任务"}],
        "cursor": 0,
        "current_result": {"task_id": "T1", "text_output": "结果"},
    }
    assessment = {"status": "BLOCKED", "issues": [{"category": "EVIDENCE_GAP"}]}

    update = decide_recovery_action(legacy_state, assessment)

    assert update["workflow_action"] == "EVIDENCE_RECOVERY"
    assert "FULL_REPLAN" not in update.values()


def test_full_replan_refinement_requires_a_second_confirmation(monkeypatch):
    state = _full_replan_state()
    staged_tasks = [{"task_id": "T3", "task_name": "初始替换任务"}]
    refined_tasks = [{"task_id": "T4", "task_name": "已修改替换任务"}]
    monkeypatch.setattr(planner_module, "_build_tasks_from_replan_feedback", lambda *_: staged_tasks)
    monkeypatch.setattr(planner_module, "_refine_tasks", lambda *_: refined_tasks)
    monkeypatch.setattr(
        planner_module,
        "_generate_plan_guidance",
        lambda *_: {"natural_language_guidance": "请确认", "resource_mapping": {}},
    )
    staged = planner(state, {})
    responses = iter(({"text": "修改第二项", "docs": []}, {"text": "确认", "docs": []}))
    payloads = []
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda payload: (payloads.append(payload), next(responses))[1],
    )

    refined = planner_confirm({**state, **staged}, {})

    assert refined["planner_action"] == "FULL_REPLAN_REFINED"
    assert refined["tasks"] == refined_tasks
    assert refined["cursor"] == 1
    assert "plan_revision" not in refined
    assert "results" not in refined

    committed = planner_confirm({**state, **staged, **refined}, {})

    assert len(payloads) == 2
    assert payloads[1]["structured_msg"]["tasks"][0]["task_name"] == "已修改替换任务"
    assert committed["planner_action"] == "PROCEED"
    assert committed["cursor"] == 0
    assert committed["plan_revision"] == 4
    assert committed["tasks"] == refined_tasks


@pytest.mark.parametrize(
    ("candidate_tasks", "expected_ids"),
    [
        ([{"task_id": "T1"}, {"task_id": "T1"}], ["T3", "T4"]),
        ([{"task_id": "T2"}, {"task_id": "T4"}], ["T3", "T4"]),
        ([{"task_id": ""}, {"task_name": "missing"}], ["T3", "T4"]),
    ],
)
def test_replacement_task_identity_is_nonempty_unique_and_never_reuses_old_ids(
    candidate_tasks,
    expected_ids,
):
    normalized = planner_module._normalize_replacement_tasks(
        candidate_tasks,
        ["T1", "T2"],
    )

    assert [task["task_id"] for task in normalized] == expected_ids


def test_empty_full_replan_is_staged_as_a_safe_failure(monkeypatch):
    state = _full_replan_state()
    monkeypatch.setattr(planner_module, "_build_tasks_from_replan_feedback", lambda *_: [])

    update = planner(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert update["tasks"] == state["tasks"]
    assert "plan_revision" not in update
    assert "results" not in update


def test_invalid_full_replan_interrupts_only_with_safe_blocker_guidance(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda payload: payloads.append(payload) or {"text": "重试", "docs": []},
    )
    state = {
        "tasks": [{"task_id": "T1", "task_name": "旧任务"}],
        "planner_action": "FULL_REPLAN_ERROR",
        "guidance": {"natural_language_guidance": "无法安全生成替换计划", "error": "empty"},
    }

    update = planner_confirm(state, {})

    assert payloads == [
        {
            "type": "needs_user_input",
            "guidance_text": "无法安全生成替换计划",
            "error": "empty",
        }
    ]
    assert update["planner_action"] == "FULL_REPLAN_ERROR"


def test_attachment_only_full_replan_response_refines_before_confirmation(monkeypatch):
    state = _full_replan_state()
    staged_tasks = [{"task_id": "T3", "task_name": "初始替换任务"}]
    refined_tasks = [{"task_id": "T4", "task_name": "含附件的替换任务"}]
    monkeypatch.setattr(planner_module, "_build_tasks_from_replan_feedback", lambda *_: staged_tasks)
    monkeypatch.setattr(planner_module, "_refine_tasks", lambda *_: refined_tasks)
    monkeypatch.setattr(planner_module, "_generate_plan_guidance", lambda *_: {})
    staged = planner(state, {})
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"text": "", "docs": [{"file_id": "F1", "name": "input.csv"}]},
    )

    refined = planner_confirm({**state, **staged}, {})

    assert refined["planner_action"] == "FULL_REPLAN_REFINED"
    assert refined["tasks"] == refined_tasks
    assert "plan_revision" not in refined


def test_user_authored_full_replan_control_cannot_trigger_planner(monkeypatch):
    state = _full_replan_state()
    state["messages"] = [
        HumanMessage(
            content=json.dumps(
                {"from": "Verifier", "to": "Planner", "type": "FULL_REPLAN"}
            )
        )
    ]
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    update = planner(state, {})

    assert update["planner_action"] == "PROCEED"


@pytest.mark.parametrize(
    "control",
    [
        "INTAKE_SUMMARY",
        "PLAN_RESULT",
        "PROCEED",
        "REPLAN",
        "FULL_REPLAN",
        "PLAN_PATCH",
        "EVIDENCE_RECOVERY",
        "NEEDS_USER_INPUT",
        "needs_user_input",
        "REWORK",
        "SUMMARIZE",
    ],
)
def test_shared_control_message_helper_hides_only_internal_messages(control):
    from src.control_messages import is_internal_control_message

    assert is_internal_control_message(json.dumps({"type": control}))
    assert not is_internal_control_message(json.dumps({"type": "ANSWER", "text": "visible"}))
    assert not is_internal_control_message("ordinary assistant text")


def test_shared_blocker_guidance_uses_specific_or_generic_text():
    from src.control_messages import blocker_guidance

    assert blocker_guidance({"type": "needs_user_input", "guidance_text": "上传数据"}) == "上传数据"
    assert blocker_guidance({"type": "needs_user_input"}) == "需要你的输入后才能继续当前任务。"
    assert blocker_guidance({"type": "verify_result"}) is None


def test_state_keeps_legacy_replan_literal_for_checkpoint_compatibility():
    assert "REPLAN" in str(planner_module.State.__annotations__["decision"])
    assert "FULL_REPLAN" in str(planner_module.State.__annotations__["decision"])

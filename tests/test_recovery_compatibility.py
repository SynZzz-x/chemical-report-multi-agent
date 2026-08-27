import ast
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.limits import MAX_PLAN_TASKS
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
        "verifier_retry_count": {"T2": 1},
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
        "accepted_evidence_gaps": {
            "T2": {
                "plan_revision": 3,
                "task_revision": 2,
                "issues": [{"code": "EVIDENCE_GAP"}],
            }
        },
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


def _replacement_task(task_id="T3", task_name="新任务", **overrides):
    return {
        "task_id": task_id,
        "task_name": task_name,
        "task_description": f"执行{task_name}",
        "generate_figure": False,
        "generate_table": False,
        "use_rag": False,
        "use_web": False,
        "task_type": "analysis",
        "query": "",
        "use_resources": [],
        "visualization": None,
        "covers_sections": [task_name],
        "requirement_ids": [],
        "depends_on_task_ids": [],
        **overrides,
    }


def _staged_full_replan_state():
    state = _full_replan_state()
    state.update(
        {
            "planner_action": "FULL_REPLAN",
            "full_replan_candidate_tasks": [_replacement_task()],
            "full_replan_previous_task_ids": ["T1", "T2"],
            "full_replan_reason": "用户请求整体重规划",
            "guidance": {"natural_language_guidance": "请确认替换计划"},
        }
    )
    return state


def test_full_replan_commits_new_revision_only_after_confirmation(monkeypatch):
    state = _full_replan_state()
    replacement_tasks = [_replacement_task()]
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
    assert staged["tasks"] == state["tasks"]
    assert staged["full_replan_candidate_tasks"] == replacement_tasks
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
    assert committed["verifier_retry_count"] == {}
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
    assert committed["accepted_evidence_gaps"] == {}
    assert committed["guidance"] == {}
    assert committed["final_result"] == {}
    assert committed["decision"] == "NEXT"
    assert committed["full_replan_previous_task_ids"] == []
    assert committed["full_replan_reason"] == ""
    assert committed["full_replan_candidate_tasks"] == []
    assert committed["task_id_registry"] == ["T1", "T2", "T3"]
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


def test_modified_initial_plan_is_staged_and_requires_second_confirmation(monkeypatch):
    state = {
        "tasks": [_replacement_task("T1", "初始任务")],
        "cursor": 0,
        "plan_revision": 1,
        "planner_action": "INTAKE_SUMMARY",
        "guidance": {"natural_language_guidance": "请确认初始计划"},
        "messages": [],
    }
    revised = [
        _replacement_task(
            "T1",
            "修订任务",
            tool_requirements=["SpiderTool"],
            use_web=True,
            allow_web_fallback=False,
        )
    ]
    monkeypatch.setattr(planner_module, "_refine_tasks", lambda *_: revised)
    monkeypatch.setattr(
        planner_module,
        "_generate_plan_guidance",
        lambda *_: {"natural_language_guidance": "请确认修订计划"},
    )
    responses = iter(
        (
            {"text": "修改任务", "docs": []},
            {"text": "确认", "docs": []},
        )
    )
    payloads = []
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda payload: (payloads.append(payload), next(responses))[1],
    )

    staged = planner_confirm(state, {})

    assert staged["planner_action"] == "INTAKE_SUMMARY_REFINED"
    assert staged["tasks"][0]["tool_requirements"] == ["spider_tool"]
    assert not any(
        json.loads(message.content).get("type") == "PLAN_RESULT"
        for message in staged["messages"]
        if isinstance(message, AIMessage)
    )

    committed = planner_confirm({**state, **staged}, {})

    assert len(payloads) == 2
    displayed = payloads[1]["structured_msg"]["tasks"][0]
    assert displayed["tool_requirements"] == ["spider_tool"]
    assert displayed["use_web"] is True
    assert displayed["allow_web_fallback"] is False
    assert committed["tasks"] == staged["tasks"]
    assert committed["cursor"] == 0
    assert any(
        json.loads(message.content).get("type") == "PLAN_RESULT"
        for message in committed["messages"]
        if isinstance(message, AIMessage)
    )


@pytest.mark.parametrize(
    "invalid_refinement",
    [
        [
            _replacement_task(f"T{index}", f"任务 {index}")
            for index in range(1, 101)
        ],
        [_replacement_task("T1", "联网任务", tool_requirements=["SpiderTool"])],
    ],
)
def test_invalid_initial_refinement_preserves_candidate_and_stays_recoverable(
    monkeypatch, invalid_refinement
):
    original_tasks = [_replacement_task("T1", "初始任务")]
    state = {
        "tasks": original_tasks,
        "cursor": 0,
        "planner_action": "INTAKE_SUMMARY",
        "guidance": {"natural_language_guidance": "请确认初始计划"},
        "messages": [],
    }
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"text": "修改计划", "docs": []},
    )
    monkeypatch.setattr(
        planner_module,
        "_refine_tasks",
        lambda *_: invalid_refinement,
    )

    update = planner_confirm(state, {})

    assert update["planner_action"] == "INTAKE_SUMMARY_REFINED"
    assert "tasks" not in update
    assert "cursor" not in update
    assert "plan_revision" not in update
    assert update["guidance"]["error"]
    assert state["tasks"] == original_tasks


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


def test_automatic_planner_clears_restored_full_replan_retry_state(monkeypatch):
    proceed = AIMessage(content=json.dumps({"to": "Planner", "type": "PROCEED"}))
    state = {
        "tasks": [{"task_id": "T1"}, {"task_id": "T2"}, {"task_id": "T3"}],
        "cursor": 1,
        "decision": "FULL_REPLAN",
        "planner_action": "FULL_REPLAN_RETRY",
        "full_replan_previous_task_ids": ["T1", "T2"],
        "full_replan_reason": "stale retry",
        "full_replan_candidate_tasks": [{"task_id": "T9", "task_name": "stale"}],
        "guidance": {"error": "stale replacement failure"},
        "messages": [proceed],
    }
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: (_ for _ in ()).throw(AssertionError("automatic full replan")),
    )

    update = automatic_planner(state, {})

    assert update["planner_action"] == "PROCEED"
    assert update["cursor"] == 2
    assert update["full_replan_previous_task_ids"] == []
    assert update["full_replan_reason"] == ""
    assert update["full_replan_candidate_tasks"] == []
    assert update["guidance"] == {}


def test_automatic_planner_preserves_initial_plan_confirmation(monkeypatch):
    intake = AIMessage(
        content=json.dumps(
            {
                "from": "Intake",
                "to": "Planner",
                "type": "INTAKE_SUMMARY",
                "sections": ["摘要"],
            }
        )
    )
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_with_llm",
        lambda *_: [{"task_id": "T1", "task_name": "初始任务"}],
    )
    monkeypatch.setattr(
        planner_module,
        "_generate_plan_guidance",
        lambda *_: {"natural_language_guidance": "请确认初始计划"},
    )

    update = automatic_planner({"messages": [intake]}, {})

    assert update["planner_action"] == "INTAKE_SUMMARY"
    assert update["guidance"] == {"natural_language_guidance": "请确认初始计划"}
    assert update["planner_action"] != "PROCEED"


def test_automatic_planner_clears_stale_full_replan_state_but_keeps_initial_confirmation(
    monkeypatch,
):
    full_replan = AIMessage(
        content=json.dumps(
            {"from": "Verifier", "to": "Planner", "type": "FULL_REPLAN"}
        )
    )
    intake = AIMessage(
        content=json.dumps(
            {
                "from": "Intake",
                "to": "Planner",
                "type": "INTAKE_SUMMARY",
                "sections": ["摘要"],
            }
        )
    )
    state = {
        "planner_action": "FULL_REPLAN_RETRY",
        "full_replan_previous_task_ids": ["T1"],
        "full_replan_reason": "stale retry",
        "full_replan_candidate_tasks": [{"task_id": "T9"}],
        "guidance": {"error": "stale replacement failure"},
        "messages": [full_replan, intake],
    }
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: (_ for _ in ()).throw(AssertionError("automatic full replan")),
    )
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_with_llm",
        lambda *_: [{"task_id": "T2", "task_name": "初始任务"}],
    )
    monkeypatch.setattr(
        planner_module,
        "_generate_plan_guidance",
        lambda *_: {"natural_language_guidance": "请确认新的初始计划"},
    )

    update = automatic_planner(state, {})

    assert update["planner_action"] == "INTAKE_SUMMARY"
    assert update["guidance"] == {"natural_language_guidance": "请确认新的初始计划"}
    assert update["planner_action"] != "PROCEED"
    assert update["full_replan_previous_task_ids"] == []
    assert update["full_replan_reason"] == ""
    assert update["full_replan_candidate_tasks"] == []


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
    staged_tasks = [_replacement_task(task_name="初始替换任务")]
    refined_tasks = [_replacement_task(task_id="T4", task_name="已修改替换任务")]
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
    assert "tasks" not in refined
    assert refined["full_replan_candidate_tasks"] == refined_tasks
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
        (
            [_replacement_task("T1", "一"), _replacement_task("T1", "二")],
            ["T3", "T4"],
        ),
        (
            [_replacement_task("T2", "一"), _replacement_task("T4", "二")],
            ["T3", "T4"],
        ),
        (
            [
                _replacement_task("", "一"),
                {key: value for key, value in _replacement_task("T9", "二").items() if key != "task_id"},
            ],
            ["T3", "T4"],
        ),
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


@pytest.mark.parametrize(
    "candidate",
    [
        {},
        _replacement_task(_recovery_allow_web=True),
    ],
)
def test_malformed_full_replan_candidate_is_staged_as_safe_failure(
    monkeypatch, candidate
):
    state = _full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: [candidate],
    )

    update = planner(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert update["tasks"] == state["tasks"]
    assert "full_replan_candidate_tasks" not in update
    assert "plan_revision" not in update


def test_malformed_refined_full_replan_candidate_is_recoverable_error(monkeypatch):
    state = _staged_full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "REFINE", "text": "修改第二项", "docs": []},
    )
    monkeypatch.setattr(
        planner_module,
        "_refine_tasks",
        lambda *_: [_replacement_task(generate_table="false")],
    )

    update = planner_confirm(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert "tasks" not in update
    assert "results" not in update
    assert "plan_revision" not in update


def test_oversized_full_replan_candidate_is_recoverable_without_mutation(monkeypatch):
    state = _full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: [
            _replacement_task(f"T{index}", f"替换任务 {index}")
            for index in range(1, MAX_PLAN_TASKS + 2)
        ],
    )

    update = planner(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert update["tasks"] == state["tasks"]
    assert "full_replan_candidate_tasks" not in update
    assert "plan_revision" not in update


def test_oversized_refined_full_replan_candidate_is_recoverable(monkeypatch):
    state = _staged_full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "REFINE", "text": "扩展计划", "docs": []},
    )
    monkeypatch.setattr(
        planner_module,
        "_refine_tasks",
        lambda *_: [
            _replacement_task(f"T{index}", f"细化任务 {index}")
            for index in range(1, MAX_PLAN_TASKS + 2)
        ],
    )

    update = planner_confirm(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert "tasks" not in update
    assert "results" not in update
    assert "plan_revision" not in update


def test_full_replan_refinement_model_error_is_recoverable(monkeypatch):
    state = _staged_full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "REFINE", "text": "修改第二项", "docs": []},
    )
    monkeypatch.setattr(
        planner_module,
        "get_llm",
        lambda *_: _ReplanModel("not valid json"),
    )

    update = planner_confirm(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert "tasks" not in update
    assert "results" not in update
    assert "plan_revision" not in update


@pytest.mark.parametrize(
    "resume_value",
    [
        {"action": "CANCEL", "text": "取消", "docs": []},
        {"action": "RESUME_OLD_PLAN", "text": "继续旧计划", "docs": []},
        "取消整体重规划",
        "恢复旧计划",
    ],
)
def test_staged_full_replan_can_resume_old_plan_without_mutation(
    monkeypatch, resume_value
):
    state = _staged_full_replan_state()
    monkeypatch.setattr(planner_module, "interrupt", lambda _: resume_value)

    update = planner_confirm(state, {})

    assert update["planner_action"] == "PROCEED"
    assert update["decision"] == "NEXT"
    assert update["full_replan_candidate_tasks"] == []
    assert update["full_replan_previous_task_ids"] == []
    assert update["full_replan_reason"] == ""
    for field in ("tasks", "results", "all_results", "cursor", "plan_revision"):
        assert field not in update


def test_staged_full_replan_explicit_confirm_action_commits(monkeypatch):
    state = _staged_full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "CONFIRM", "text": "执行", "docs": []},
    )

    update = planner_confirm(state, {})

    assert update["planner_action"] == "PROCEED"
    assert update["tasks"] == state["full_replan_candidate_tasks"]
    assert update["plan_revision"] == state["plan_revision"] + 1


def test_staged_full_replan_explicit_refine_action_overrides_confirmation_text(
    monkeypatch,
):
    state = _staged_full_replan_state()
    refined = [_replacement_task(task_id="T4", task_name="已细化任务")]
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "REFINE", "text": "确认", "docs": []},
    )
    monkeypatch.setattr(planner_module, "_refine_tasks", lambda *_: refined)
    monkeypatch.setattr(planner_module, "_generate_plan_guidance", lambda *_: {})

    update = planner_confirm(state, {})

    assert update["planner_action"] == "FULL_REPLAN_REFINED"
    assert update["full_replan_candidate_tasks"] == refined
    assert "plan_revision" not in update


class _ReplanResponse:
    def __init__(self, content):
        self.content = content


class _ReplanModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def invoke(self, *_args, **_kwargs):
        if self.error is not None:
            raise self.error
        return _ReplanResponse(self.response)


def test_initial_plan_rejects_oversized_generation_without_fallback(monkeypatch):
    oversized = [
        _replacement_task(f"T{index}", f"初始任务 {index}")
        for index in range(1, MAX_PLAN_TASKS + 2)
    ]
    monkeypatch.setattr(
        planner_module,
        "get_llm",
        lambda *_, **__: _ReplanModel(
            json.dumps({"tasks": oversized}, ensure_ascii=False)
        ),
    )

    with pytest.raises(ValueError, match="initial plan generation failed"):
        planner_module._build_tasks_with_llm(
            {"sections": ["摘要", "结论"], "resources": []}, {}
        )


def test_initial_plan_accepts_exactly_max_tasks(monkeypatch):
    candidates = [
        _replacement_task(f"T{index}", f"初始任务 {index}")
        for index in range(1, MAX_PLAN_TASKS + 1)
    ]
    monkeypatch.setattr(
        planner_module,
        "get_llm",
        lambda *_, **__: _ReplanModel(
            json.dumps({"tasks": candidates}, ensure_ascii=False)
        ),
    )

    tasks = planner_module._build_tasks_with_llm({"resources": []}, {})

    assert len(tasks) == MAX_PLAN_TASKS


def test_full_replan_rejects_spider_alias_without_web_authorization(monkeypatch):
    state = _full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: [_replacement_task(tool_requirements=["SpiderTool"])],
    )

    update = planner(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert update["tasks"] == state["tasks"]
    assert "full_replan_candidate_tasks" not in update


@pytest.mark.parametrize(
    "web_authorization",
    [
        {"use_web": True},
        {"allow_web_fallback": True},
        {"visualization": {"allow_web_fallback": True}},
    ],
)
def test_full_replan_canonicalizes_authorized_spider_alias(web_authorization):
    normalized = planner_module._normalize_replacement_tasks(
        [
            _replacement_task(
                tool_requirements=["SpiderTool"],
                **web_authorization,
            )
        ],
        ["T1", "T2"],
    )

    assert normalized[0]["tool_requirements"] == ["spider_tool"]


def test_full_replan_confirmation_exposes_tool_and_web_requirements(monkeypatch):
    state = _staged_full_replan_state()
    state["full_replan_candidate_tasks"] = [
        _replacement_task(
            tool_requirements=["SpiderTool"],
            use_web=True,
            allow_web_fallback=False,
            visualization={"allow_web_fallback": False},
        )
    ]
    payloads = []
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda payload: payloads.append(payload)
        or {"action": "CANCEL", "text": "取消", "docs": []},
    )

    planner_confirm(state, {})

    displayed = payloads[0]["structured_msg"]["tasks"][0]
    assert displayed["tool_requirements"] == ["spider_tool"]
    assert displayed["use_web"] is True
    assert displayed["allow_web_fallback"] is False
    assert displayed["visualization"] == {"allow_web_fallback": False}


@pytest.mark.parametrize(
    "response",
    [
        "not valid json",
        json.dumps({"task_id": "T9"}),
        json.dumps([{"task_name": "valid"}, "not-a-task"]),
    ],
)
def test_invalid_replacement_generation_stages_recoverable_error(response, monkeypatch):
    state = _full_replan_state()
    monkeypatch.setattr(planner_module, "get_llm", lambda *_: _ReplanModel(response))

    update = planner(state, {})

    assert update["planner_action"] == "FULL_REPLAN_ERROR"
    assert update["tasks"] == state["tasks"]
    assert "full_replan_candidate_tasks" not in update
    assert "plan_revision" not in update


def test_replacement_model_failure_stages_error_and_can_resume_old_plan(monkeypatch):
    state = _full_replan_state()
    monkeypatch.setattr(
        planner_module,
        "get_llm",
        lambda *_: _ReplanModel(error=RuntimeError("model unavailable")),
    )

    failed = planner(state, {})
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "CANCEL", "text": "continue old", "docs": []},
    )
    resumed = planner_confirm({**state, **failed}, {})

    assert failed["planner_action"] == "FULL_REPLAN_ERROR"
    assert failed["tasks"] == state["tasks"]
    assert resumed["planner_action"] == "PROCEED"
    assert "tasks" not in resumed
    assert "results" not in resumed


def test_restored_checkpoint_without_registry_reserves_historical_task_ids(monkeypatch):
    state = _full_replan_state()
    state.pop("task_id_registry", None)
    state["current_task"] = {"task_id": "T3"}
    state["current_result"] = {"task_id": "T4"}
    state["results"] = [{"task_id": "T5"}]
    state["all_results"] = [{"task_id": "T6"}]
    state["tool_execution_history"] = [
        {"task_id": "T7", "id": "unrelated-record-id"},
        {"id": "also-unrelated"},
    ]
    state["task_revisions"] = {"T8": 1}
    state["task_retry_count"] = {"T9": 1}
    state["evidence_recovery_count"] = {"T10": 1}
    state["task_patch_count"] = {"T11": 1}
    state["verification_warnings"] = [{"task_id": "T12"}]
    state["pending_user_action"] = {"task_id": "T13"}
    state["verifier_retry_count"] = {"T14": 1}
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: [_replacement_task("T1", "replacement")],
    )
    monkeypatch.setattr(planner_module, "_generate_plan_guidance", lambda *_: {})

    staged = planner(state, {})

    assert "unrelated-record-id" not in planner_module._job_task_ids(state)
    assert staged["full_replan_candidate_tasks"][0]["task_id"] == "T15"


def test_legacy_integer_counter_keys_map_to_tasks_or_are_ignored():
    task_ids = planner_module._counter_task_ids(
        {0: 1, 3: 1, "T_HISTORICAL": 1},
        [{"task_id": "T1"}, {"task_id": "T2"}],
    )

    assert task_ids == ["T1", "T_HISTORICAL"]
    assert "0" not in task_ids
    assert "3" not in task_ids


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
            "accepted_choices": ["RETRY_FULL_REPLAN", "RESUME_OLD_PLAN", "CANCEL"],
        }
    ]
    assert update["planner_action"] == "FULL_REPLAN_RETRY"


def test_attachment_only_full_replan_response_refines_before_confirmation(monkeypatch):
    state = _full_replan_state()
    staged_tasks = [_replacement_task(task_name="初始替换任务")]
    refined_tasks = [_replacement_task(task_id="T4", task_name="含附件的替换任务")]
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
    assert "tasks" not in refined
    assert refined["full_replan_candidate_tasks"] == refined_tasks
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
        "TASK_RESULT",
        "FINAL_RESULT",
    ],
)
def test_shared_control_message_helper_hides_only_internal_messages(control):
    from src.control_messages import is_internal_control_message

    assert is_internal_control_message(json.dumps({"type": control}))
    assert not is_internal_control_message(json.dumps({"type": "ANSWER", "text": "visible"}))
    assert not is_internal_control_message("ordinary assistant text")


def test_shared_blocker_guidance_uses_specific_or_generic_text():
    from src.control_messages import (
        blocker_choices,
        blocker_guidance,
        build_resume_payload,
    )

    assert blocker_guidance({"type": "needs_user_input", "guidance_text": "上传数据"}) == "上传数据"
    assert blocker_guidance({"type": "needs_user_input"}) == "需要你的输入后才能继续当前任务。"
    assert blocker_guidance({"type": "verify_result"}) is None
    payload = {
        "type": "needs_user_input",
        "accepted_choices": ["UPLOAD_RESOURCES", "AUTHORIZE_WEB"],
    }
    assert blocker_choices(payload) == ["UPLOAD_RESOURCES", "AUTHORIZE_WEB"]
    assert build_resume_payload(
        text="允许联网",
        docs=[],
        action="AUTHORIZE_WEB",
        message_id="msg-1",
    ) == {
        "text": "允许联网",
        "message_id": "msg-1",
        "docs": [],
        "action": "AUTHORIZE_WEB",
    }


def test_blocker_choices_preserve_all_evidence_gap_actions():
    from src.control_messages import blocker_choices

    unavailable_payload = {
        "type": "needs_user_input",
        "category": "EVIDENCE_GAP",
        "accepted_choices": [
            "UPLOAD_RESOURCES",
            "ADJUST_REQUIREMENT",
            "ACCEPT_EVIDENCE_GAP",
        ],
    }
    available_payload = {
        **unavailable_payload,
        "accepted_choices": [
            "UPLOAD_RESOURCES",
            "AUTHORIZE_WEB",
            "ADJUST_REQUIREMENT",
            "ACCEPT_EVIDENCE_GAP",
        ],
    }

    assert blocker_choices(unavailable_payload) == unavailable_payload[
        "accepted_choices"
    ]
    assert blocker_choices(available_payload) == available_payload[
        "accepted_choices"
    ]


def test_blocker_action_specs_define_direct_submission_requirements():
    from src.control_messages import blocker_action_spec

    assert blocker_action_spec("AUTHORIZE_WEB") == {
        "label": "授权公开网络检索",
        "button_label": "授权并继续",
        "default_text": "已授权公开网络检索，请继续。",
        "requires_text": False,
        "requires_documents": False,
    }
    assert blocker_action_spec("ACCEPT_EVIDENCE_GAP") == {
        "label": "接受现有证据及缺口报告",
        "button_label": "接受并继续",
        "default_text": "接受当前可豁免的证据缺口，请继续修复其他问题。",
        "requires_text": False,
        "requires_documents": False,
    }
    assert blocker_action_spec("ACCEPT_AS_DRAFT")["default_text"] == (
        "接受当前缺陷并作为带风险草稿继续。"
    )
    assert blocker_action_spec("RETRY_ASSET") == {
        "label": "重新生成图形或表格",
        "button_label": "重新生成资产",
        "default_text": "仅重新生成当前缺失的图形或表格，不重写正文。",
        "requires_text": False,
        "requires_documents": False,
    }
    assert blocker_action_spec("ADJUST_REQUIREMENT")["requires_text"] is True
    assert blocker_action_spec("UPLOAD_RESOURCES")["requires_documents"] is True
    assert blocker_action_spec("RETRY_INITIAL_PLAN") == {
        "label": "重新生成初始规划",
        "button_label": "重新生成",
        "default_text": "请重新生成初始规划。",
        "requires_text": False,
        "requires_documents": False,
    }
    assert blocker_action_spec("CANCEL") == {
        "label": "取消本次处理",
        "button_label": "确认取消",
        "default_text": "取消本次处理。",
        "requires_text": False,
        "requires_documents": False,
    }


def test_blocker_submission_validation_rejects_missing_required_input():
    from src.control_messages import validate_blocker_submission

    assert validate_blocker_submission("ADJUST_REQUIREMENT", "", 0) == (
        "请输入调整后的任务要求。"
    )
    assert validate_blocker_submission("ADJUST_REQUIREMENT", "缩小指标范围", 0) is None
    assert validate_blocker_submission("UPLOAD_RESOURCES", "", 0) == (
        "请先上传补充资料。"
    )
    assert validate_blocker_submission("UPLOAD_RESOURCES", "", 1) is None
    assert validate_blocker_submission("AUTHORIZE_WEB", "", 0) is None
    assert validate_blocker_submission("ACCEPT_EVIDENCE_GAP", "", 0) is None
    assert validate_blocker_submission("", "", 0) == "请选择有效的处理方式。"
    assert validate_blocker_submission("RETRY_INITIAL_PLAN", "", 0) is None


def test_blocker_resume_payload_applies_defaults_and_preserves_action_and_docs():
    from src.control_messages import build_blocker_resume_payload

    assert build_blocker_resume_payload(
        action="AUTHORIZE_WEB",
        text="",
        docs=[],
        message_id="msg-web",
    ) == {
        "text": "已授权公开网络检索，请继续。",
        "message_id": "msg-web",
        "docs": [],
        "action": "AUTHORIZE_WEB",
    }

    assert build_blocker_resume_payload(
        action="ACCEPT_EVIDENCE_GAP",
        text="",
        docs=[],
        message_id="msg-gap",
    ) == {
        "text": "接受当前可豁免的证据缺口，请继续修复其他问题。",
        "message_id": "msg-gap",
        "docs": [],
        "action": "ACCEPT_EVIDENCE_GAP",
    }

    docs = [{"file_id": "file-1", "path": "/tmp/standard.pdf"}]
    assert build_blocker_resume_payload(
        action="UPLOAD_RESOURCES",
        text="",
        docs=docs,
        message_id="msg-upload",
    ) == {
        "text": "我已上传补充资料，请结合附件继续处理当前任务。",
        "message_id": "msg-upload",
        "docs": docs,
        "action": "UPLOAD_RESOURCES",
    }


@pytest.mark.parametrize(
    ("action", "text", "docs", "message"),
    [
        ("ADJUST_REQUIREMENT", "", [], "请输入调整后的任务要求。"),
        ("UPLOAD_RESOURCES", "", [], "请先上传补充资料。"),
        ("", "", [], "请选择有效的处理方式。"),
    ],
)
def test_blocker_resume_payload_rejects_incomplete_submission(
    action,
    text,
    docs,
    message,
):
    from src.control_messages import build_blocker_resume_payload

    with pytest.raises(ValueError, match=message):
        build_blocker_resume_payload(
            action=action,
            text=text,
            docs=docs,
            message_id="msg-invalid",
        )


def test_streamlit_uses_action_specific_blocker_submission_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )

    assert "_render_pending_resume_submission" in source
    assert "blocker_action_spec" in source
    assert "validate_blocker_submission" in source
    assert "build_blocker_resume_payload" in source
    assert "st.file_uploader(" in source
    assert "st.text_area(" in source
    assert "has_blocker_actions = bool(blocker_choices(pending_interrupt))" in source
    assert "disabled=has_blocker_actions" in source
    assert "disabled=pending_interrupt is not None" not in source
    assert "pending_resume_action = _render_pending_resume_action()" not in source
    assert '"resume_action": resume_action' in source
    assert '"resume_action_label": resume_action_label' in source
    assert 'file_type=["csv", "pdf", "docx", "txt", "md"]' in source


def test_streamlit_blocker_selector_is_an_unselected_radio():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    render_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_pending_resume_submission"
    )
    radio_calls = [
        node
        for node in ast.walk(render_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "radio"
    ]
    selectbox_calls = [
        node
        for node in ast.walk(render_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "selectbox"
    ]

    assert len(radio_calls) == 1
    assert selectbox_calls == []
    keywords = {keyword.arg: keyword.value for keyword in radio_calls[0].keywords}
    assert isinstance(keywords["options"], ast.Name)
    assert keywords["options"].id == "choices"
    assert isinstance(keywords["index"], ast.Constant)
    assert keywords["index"].value is None


def test_shared_display_consumer_hides_controls_without_calling_stale_alias():
    from src.control_messages import is_displayable_assistant_content

    assert not is_displayable_assistant_content(json.dumps({"type": "FULL_REPLAN"}))
    assert is_displayable_assistant_content(json.dumps({"type": "ANSWER", "text": "visible"}))
    assert is_displayable_assistant_content("ordinary assistant text")


def test_shared_message_projection_consumer_hides_controls_and_keeps_assistant_text():
    from src.control_messages import (
        is_displayable_assistant_message,
        is_displayable_ui_message,
    )

    assert not is_displayable_assistant_message("ai", json.dumps({"type": "FULL_REPLAN"}))
    assert is_displayable_assistant_message("assistant", "visible assistant result")
    assert not is_displayable_assistant_message("human", "visible assistant result")
    assert not is_displayable_ui_message(
        "assistant", "text", json.dumps({"type": "TASK_RESULT"})
    )
    assert not is_displayable_ui_message(
        "assistant", "text", json.dumps({"type": "FINAL_RESULT"})
    )
    assert is_displayable_ui_message(
        "assistant", "plan", json.dumps({"type": "PLAN_RESULT"})
    )
    assert is_displayable_ui_message("user", "text", "用户输入")


def test_streamlit_history_rechecks_persisted_text_message_visibility():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )

    assert "if not is_displayable_ui_message(role, kind, content):" in source
    assert source.index("if not is_displayable_ui_message(role, kind, content):") < source.index(
        'with st.chat_message(role):'
    )


def test_invalid_full_replan_can_retry_without_mutating_the_old_plan(monkeypatch):
    state = _full_replan_state()
    state.update(
        {
            "planner_action": "FULL_REPLAN_ERROR",
            "full_replan_candidate_tasks": [],
            "guidance": {"natural_language_guidance": "无法安全生成替换计划"},
        }
    )
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "RETRY_FULL_REPLAN", "text": "改为两章", "docs": []},
    )

    retry = planner_confirm(state, {})

    assert retry["planner_action"] == "FULL_REPLAN_RETRY"
    assert retry["full_replan_reason"] == "改为两章"
    assert "tasks" not in retry and "results" not in retry
    assert retry["planner_action"] != "FULL_REPLAN_ERROR"


def test_invalid_full_replan_can_resume_old_plan_without_a_loop(monkeypatch):
    state = _full_replan_state()
    state.update(
        {
            "planner_action": "FULL_REPLAN_ERROR",
            "full_replan_candidate_tasks": [],
            "guidance": {"natural_language_guidance": "无法安全生成替换计划"},
        }
    )
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda _: {"action": "RESUME_OLD_PLAN", "text": "继续旧计划", "docs": []},
    )

    resumed = planner_confirm(state, {})

    assert resumed["planner_action"] == "PROCEED"
    assert resumed["full_replan_candidate_tasks"] == []
    assert resumed["full_replan_previous_task_ids"] == []
    assert resumed["planner_action"] != "FULL_REPLAN_ERROR"
    assert "tasks" not in resumed and "results" not in resumed


def test_sequential_full_replans_use_lifetime_ids_and_fresh_reasons(monkeypatch):
    state = _full_replan_state()
    state["messages"] = [
        AIMessage(content=json.dumps({"from": "Verifier", "to": "Planner", "type": "FULL_REPLAN", "reason": "first"}))
    ]
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        lambda *_: [_replacement_task("T1")],
    )
    monkeypatch.setattr(planner_module, "_generate_plan_guidance", lambda *_: {})
    monkeypatch.setattr(planner_module, "interrupt", lambda _: {"text": "确认", "docs": []})

    first_stage = planner(state, {})
    first_commit = planner_confirm({**state, **first_stage}, {})

    second_state = {
        **state,
        **first_commit,
        "messages": [
            AIMessage(content=json.dumps({"from": "Verifier", "to": "Planner", "type": "FULL_REPLAN", "reason": "second"}))
        ],
    }
    second_stage = planner(second_state, {})
    second_commit = planner_confirm({**second_state, **second_stage}, {})

    assert first_commit["tasks"][0]["task_id"] == "T3"
    assert second_stage["full_replan_candidate_tasks"][0]["task_id"] == "T4"
    assert second_commit["task_id_registry"] == ["T1", "T2", "T3", "T4"]
    assert second_commit["plan_patch_history"][-1]["reason"] == "second"
    assert second_commit["full_replan_reason"] == ""


def test_state_keeps_legacy_replan_literal_for_checkpoint_compatibility():
    assert "REPLAN" in str(planner_module.State.__annotations__["decision"])
    assert "FULL_REPLAN" in str(planner_module.State.__annotations__["decision"])

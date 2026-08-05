import json
from pathlib import Path

from langchain_core.messages import AIMessage

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
        "task_revisions": {"T1": 1, "T2": 2},
        "plan_revision": 3,
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
    assert committed["task_revisions"] == {"T3": 1}


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


def test_ui_and_cli_hide_recovery_controls_and_render_blocker_guidance():
    root = Path(__file__).parents[1]
    app_source = (root / "app.py").read_text(encoding="utf-8")
    cli_source = (root / "run.py").read_text(encoding="utf-8")

    for control in (
        "FULL_REPLAN",
        "PLAN_PATCH",
        "EVIDENCE_RECOVERY",
        "NEEDS_USER_INPUT",
    ):
        assert control in app_source
        assert control in cli_source
    assert 'payload_type == "needs_user_input"' in app_source

import ast
import copy
from pathlib import Path
from typing import Any


def test_terminal_failure_ui_state_is_persistent_safe_and_non_resumable():
    from src.ui_projection import terminal_job_ui_state

    state = terminal_job_ui_state(
        {
            "status": "failed",
            "fatal_system_error": {
                "diagnostic_code": "VERIFIER_UNAVAILABLE",
                "raw_exception": "secret token must not be displayed",
            },
        },
        {
            "status": "failed",
            "fatal_system_error": {
                "origin": "graph",
                "diagnostic_code": "VERIFIER_UNAVAILABLE",
            },
        },
    )

    assert state == {
        "is_terminal": True,
        "message": "当前报告任务已停止（审核服务不可用）。请新建报告任务后继续。",
    }


def test_terminal_failure_ui_state_leaves_success_and_waiting_resumable():
    from src.ui_projection import terminal_job_ui_state

    assert terminal_job_ui_state({"status": "completed"}) == {
        "is_terminal": False,
        "message": None,
    }
    assert terminal_job_ui_state(
        {
            "status": "waiting",
            "pending_interrupt": {"type": "needs_user_input"},
            "fatal_system_error": {"diagnostic_code": "STALE_FATAL"},
        }
    ) == {"is_terminal": False, "message": None}


def test_terminal_ui_uses_authoritative_snapshot_outcome_over_stale_store_status():
    from src.job_outcome import project_job_outcome
    from src.ui_projection import terminal_job_ui_state

    stale_failed_record = {
        "status": "failed",
        "fatal_system_error": {"origin": "graph"},
    }
    fatal_state = {
        "workflow_action": "FATAL_SYSTEM",
        "failure_decision": {
            "failure_class": "FATAL_SYSTEM",
            "action": "FAIL_JOB",
            "subtype": "UNRECOGNIZED_INTERNAL_CODE",
        },
    }
    waiting_outcome = project_job_outcome(
        fatal_state,
        {"type": "needs_user_input"},
    )
    fatal_outcome = project_job_outcome(fatal_state, None)

    assert terminal_job_ui_state(stale_failed_record, waiting_outcome) == {
        "is_terminal": False,
        "message": None,
    }
    assert terminal_job_ui_state({"status": "running"}, fatal_outcome) == {
        "is_terminal": True,
        "message": "当前报告任务已停止（系统工作流无法继续）。请新建报告任务后继续。",
    }


def test_terminal_ui_keeps_persisted_graph_fatal_terminal_but_runner_failure_resumable():
    from src.ui_projection import terminal_job_ui_state

    assert terminal_job_ui_state(
        {
            "status": "failed",
            "fatal_system_error": {
                "origin": "graph",
                "diagnostic_code": "VERIFIER_UNAVAILABLE",
            },
        }
    ) == {
        "is_terminal": True,
        "message": "当前报告任务已停止（审核服务不可用）。请新建报告任务后继续。",
    }
    assert terminal_job_ui_state(
        {
            "status": "failed",
            "fatal_system_error": {"origin": "runner"},
        }
    ) == {
        "is_terminal": False,
        "message": "上次执行未完成，可以重新提交。",
    }


def _app_function(name: str, namespace: dict[str, Any]):
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace[name]


def test_active_terminal_state_uses_snapshot_projection_without_mutating_state():
    from src.job_outcome import project_job_outcome
    from src.ui_projection import terminal_job_ui_state

    graph_state = {
        "workflow_action": "FATAL_SYSTEM",
        "failure_decision": {
            "failure_class": "FATAL_SYSTEM",
            "action": "FAIL_JOB",
            "subtype": "VERIFIER_UNAVAILABLE",
        },
    }
    original_state = copy.deepcopy(graph_state)
    writes: list[dict[str, Any]] = []

    class Snapshot:
        values = graph_state

    class App:
        def get_state(self, config):
            return Snapshot()

    class FakeSt:
        session_state = {"app": App(), "pending_interrupt": None}

    active_state = _app_function(
        "_active_terminal_job_ui_state",
        {
            "Any": Any,
            "st": FakeSt(),
            "_current_job": lambda: {"status": "running"},
            "_graph_config": lambda: {},
            "_update_job": lambda **changes: writes.append(changes),
            "interrupt_from_snapshot": lambda snapshot: None,
            "project_job_outcome": project_job_outcome,
            "terminal_job_ui_state": terminal_job_ui_state,
        },
    )

    assert active_state()["is_terminal"] is True
    assert writes == [
        {
            "status": "failed",
            "pending_interrupt": None,
            "fatal_system_error": {
                "failure_id": writes[0]["fatal_system_error"]["failure_id"],
                "failure_class": "FATAL_SYSTEM",
                "subtype": "VERIFIER_UNAVAILABLE",
                "origin": "graph",
                "component": "Workflow",
                "operation": "terminal_projection",
                "task_id": None,
                "diagnostic_code": "VERIFIER_UNAVAILABLE",
                "retryable": False,
                "metadata": {},
            },
        }
    ]
    assert graph_state == original_state


def test_active_terminal_state_preserves_failed_to_completed_restore_protection():
    from src.job_outcome import project_job_outcome
    from src.ui_projection import terminal_job_ui_state

    writes: list[dict[str, Any]] = []

    class Snapshot:
        values = {"workflow_action": "DONE"}

    class App:
        def get_state(self, config):
            return Snapshot()

    class FakeSt:
        session_state = {"app": App(), "pending_interrupt": None}

    active_state = _app_function(
        "_active_terminal_job_ui_state",
        {
            "Any": Any,
            "st": FakeSt(),
            "_current_job": lambda: {
                "status": "failed",
                "fatal_system_error": {"origin": "runner"},
            },
            "_graph_config": lambda: {},
            "_update_job": lambda **changes: writes.append(changes),
            "interrupt_from_snapshot": lambda snapshot: None,
            "project_job_outcome": project_job_outcome,
            "terminal_job_ui_state": terminal_job_ui_state,
        },
    )

    assert active_state() == {"is_terminal": False, "message": None}
    assert writes == []


def test_streamlit_terminal_guard_executes_stop_before_dispatch_side_effects():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    terminal_guard_index = next(
        index
        for index, node in enumerate(tree.body)
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "terminal_job"
        )
    )
    terminal_guard = tree.body[terminal_guard_index]
    guarded_source_end = terminal_guard.end_lineno
    assert guarded_source_end is not None

    chat_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "chat_value"
            for target in node.targets
        )
    )
    later_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.lineno > chat_assignment.lineno
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "_save_uploaded_files"
            or isinstance(node.func, ast.Name)
            and node.func.id == "_safe_stream_updates"
            or isinstance(node.func, ast.Name)
            and node.func.id == "_update_job"
            and any(
                keyword.arg == "status"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "running"
                for keyword in node.keywords
            )
        )
    ]
    assert later_calls
    assert guarded_source_end < min(later_calls)

    calls: list[str] = []

    class StopDispatch(Exception):
        pass

    class FakeSt:
        def stop(self):
            calls.append("stop")
            raise StopDispatch()

    namespace = {
        "terminal_job": True,
        "st": FakeSt(),
        "blocker_submission": None,
        "chat_value": None,
        "_save_uploaded_files": lambda value: calls.append("upload"),
        "_update_job": lambda **changes: calls.append("running_write"),
        "_safe_stream_updates": lambda *args, **kwargs: calls.append("stream"),
    }
    fragment = ast.Module(
        body=tree.body[terminal_guard_index : terminal_guard_index + 2],
        type_ignores=[],
    )
    try:
        exec(compile(fragment, "app.py", "exec"), namespace)
    except StopDispatch:
        pass
    else:
        raise AssertionError("terminal guard did not stop dispatch")
    assert calls == ["stop"]


def test_restore_sidebar_labels_graph_terminal_as_stopped_not_resumable():
    from src.ui_projection import terminal_job_ui_state

    class FakeSt:
        session_state: dict[str, Any] = {}

    restore_sidebar = _app_function(
        "_restore_job_from_sidebar",
        {
            "Any": Any,
            "st": FakeSt(),
            "_restore_job": lambda job_id: None,
            "_current_job": lambda: {
                "status": "failed",
                "fatal_system_error": {"origin": "graph"},
            },
            "terminal_job_ui_state": terminal_job_ui_state,
        },
    )

    restore_sidebar("job-fatal")

    assert FakeSt.session_state["restore_success"] == "任务已打开；该报告任务已停止。"


def test_verifier_summary_renders_assessment_status_and_issue_category():
    from src.ui_projection import summarize_step

    summary = summarize_step(
        "Verifier",
        {
            "assessment": {
                "status": "FAILED",
                "issues": [{"category": "EVIDENCE_GAP"}],
            }
        },
    )

    assert summary == "审核状态：FAILED，问题类型：EVIDENCE_GAP"
    assert "审核决策：-" not in summary


def test_decision_policy_summary_renders_recovery_action():
    from src.ui_projection import summarize_step

    assert (
        summarize_step("DecisionPolicy", {"workflow_action": "EVIDENCE_RECOVERY"})
        == "恢复动作：EVIDENCE_RECOVERY"
    )


def test_decision_policy_summary_projects_degradation_without_blocker_wording():
    from src.ui_projection import summarize_step

    summary = summarize_step(
        "DecisionPolicy",
        {
            "workflow_action": "NEXT",
            "failure_decision": {
                "failure_class": "DEGRADABLE_QUALITY",
                "subtype": "MISSING_FIGURE",
            },
        },
    )

    assert summary == "已记录非阻塞交付限制：MISSING_FIGURE"


def test_streamlit_consumer_uses_shared_summary_projection():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")

    assert "from src.ui_projection import summarize_step as _summarize_step" in source

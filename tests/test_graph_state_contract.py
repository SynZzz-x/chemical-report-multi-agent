import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_recovery_nodes_use_optional_runnable_config_compatibility_signature():
    import inspect

    from src.nodes import recovery

    for name in (
        "automatic_planner",
        "decision_policy",
        "evidence_recovery",
        "plan_patcher",
        "needs_user_input",
    ):
        parameter = inspect.signature(getattr(recovery, name)).parameters["config"]
        assert parameter.annotation == "Optional[RunnableConfig]"
        assert parameter.default is None

    assert "config" not in inspect.signature(recovery.route_policy).parameters
    assert "config" not in inspect.signature(recovery.route_after_blocker).parameters


def test_compiled_auto_graph_pass_advances_to_next_task():
    script = textwrap.dedent(
        r'''
        import importlib
        import json

        graph_module = importlib.import_module("src.graph")
        worker_visits = []
        tasks = [
            {"task_id": "T1", "task_name": "引言", "task_description": "撰写引言"},
            {"task_id": "T2", "task_name": "正文", "task_description": "撰写正文"},
        ]

        def fake_intake(state, config=None, **kwargs):
            return {
                "tasks": tasks,
                "cursor": 0,
                "results": [],
                "planner_action": "PROCEED",
                "decision": "NEXT",
            }

        def fake_worker(state, config=None, **kwargs):
            cursor = int(state.get("cursor", 0) or 0)
            task = state["tasks"][cursor]
            worker_visits.append(task["task_id"])
            return {
                "current_task": task,
                "current_result": {
                    "task_id": task["task_id"],
                    "status": "COMPLETED",
                    "text_output": f"{task['task_name']}正文",
                },
            }

        def passing_verifier(state, config=None, **kwargs):
            return {
                "assessment": {
                    "status": "PASS",
                    "issues": [],
                    "requirements_met": [],
                    "requirements_missing": [],
                }
            }

        graph_module.intake = fake_intake
        graph_module.create_worker_workflow = lambda: fake_worker
        graph_module.verifier_auto = passing_verifier
        graph_module.summarizer = lambda state, config=None, **kwargs: {
            "final_result": {"success": True}
        }
        graph_module.exiting = lambda state, config=None, **kwargs: {}

        app = graph_module.WorkFlowAuto().compile()
        result = app.invoke(
            {"messages": [], "tasks": [], "results": [], "cursor": 0},
            {"recursion_limit": 50},
        )
        print("GRAPH_RESULT=" + json.dumps({
            "worker_visits": worker_visits,
            "cursor": result.get("cursor"),
            "workflow_action": result.get("workflow_action"),
            "result_ids": [item.get("task_id") for item in result.get("results", [])],
        }, ensure_ascii=False))
        '''
    )
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-agent"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    payload_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("GRAPH_RESULT=")
    )
    payload = json.loads(payload_line.removeprefix("GRAPH_RESULT="))

    assert payload == {
        "worker_visits": ["T1", "T2"],
        "cursor": 1,
        "workflow_action": "DONE",
        "result_ids": ["T1", "T2"],
    }


def test_compiled_auto_graph_repairs_verifier_without_rerunning_worker():
    script = textwrap.dedent(
        r'''
        import importlib
        import json
        from types import SimpleNamespace
        from langchain_core.runnables import RunnableLambda

        graph_module = importlib.import_module("src.graph")
        verifier_module = importlib.import_module("src.nodes.verifier")
        worker_visits = []
        model_calls = []
        responses = [
            "not-json",
            json.dumps({
                "status": "PASS",
                "current_section": "引言",
                "issues": [],
                "requirements_met": ["引言正文"],
                "requirements_missing": [],
            }, ensure_ascii=False),
        ]
        task = {
            "task_id": "T1",
            "task_name": "引言",
            "task_description": "撰写引言",
            "task_type": "analysis",
            "generate_table": False,
            "generate_figure": False,
        }

        def invoke_model(payload):
            model_calls.append(payload)
            return SimpleNamespace(content=responses.pop(0))

        def fake_intake(state, config=None, **kwargs):
            return {
                "tasks": [task],
                "cursor": 0,
                "results": [],
                "planner_action": "PROCEED",
                "decision": "NEXT",
                "task_retry_count": {},
                "asset_retry_count": {},
                "evidence_recovery_count": {},
                "task_patch_count": {},
                "verifier_retry_count": {},
            }

        def fake_worker(state, config=None, **kwargs):
            worker_visits.append("T1")
            return {
                "current_task": task,
                "current_result": {
                    "task_id": "T1",
                    "status": "COMPLETED",
                    "text_output": "引言正文",
                    "citations": [],
                },
            }

        graph_module.intake = fake_intake
        graph_module.create_worker_workflow = lambda: fake_worker
        graph_module.verifier_auto = verifier_module.verifier
        verifier_module.get_llm = lambda *args, **kwargs: RunnableLambda(
            invoke_model
        )
        verifier_module.get_app_config = lambda: SimpleNamespace(
            deepseek_api_key="test-key"
        )
        graph_module.summarizer = lambda state, config=None, **kwargs: {
            "final_result": {"success": True}
        }
        graph_module.exiting = lambda state, config=None, **kwargs: {}

        app = graph_module.WorkFlowAuto().compile()
        result = app.invoke(
            {"messages": [], "tasks": [], "results": [], "cursor": 0},
            {"configurable": {"use_llm": True}, "recursion_limit": 30},
        )
        print("GRAPH_RESULT=" + json.dumps({
            "worker_visits": worker_visits,
            "model_calls": len(model_calls),
            "task_retry_count": result.get("task_retry_count"),
            "verifier_retry_count": result.get("verifier_retry_count"),
            "workflow_action": result.get("workflow_action"),
            "assessment": result.get("assessment"),
            "verifier_failure": result.get("verifier_failure"),
        }, ensure_ascii=False))
        '''
    )
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-agent"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    payload_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("GRAPH_RESULT=")
    )
    payload = json.loads(payload_line.removeprefix("GRAPH_RESULT="))

    assert payload == {
        "worker_visits": ["T1"],
        "model_calls": 2,
        "task_retry_count": {},
        "verifier_retry_count": {"T1": 1},
        "workflow_action": "DONE",
        "assessment": {
            "status": "PASS",
            "current_section": "引言",
            "issues": [],
            "requirements_met": ["引言正文"],
            "requirements_missing": [],
        },
        "verifier_failure": {},
    }


def test_compiled_auto_graph_has_no_invalid_runnable_config_warning():
    script = textwrap.dedent(
        r'''
        import warnings

        warnings.filterwarnings(
            "error",
            message="The 'config' parameter should be typed.*",
        )

        from src.graph import WorkFlowAuto

        WorkFlowAuto().compile()
        '''
    )
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-agent"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

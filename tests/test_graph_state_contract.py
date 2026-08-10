import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


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

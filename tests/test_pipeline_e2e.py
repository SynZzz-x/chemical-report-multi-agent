"""Real LangGraph integration coverage in a clean subprocess.

The unit-test suite installs lightweight dependency stubs in conftest.py. These
scenarios intentionally start a clean interpreter so they exercise the actual
LangGraph and SQLite checkpoint/store implementations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_RUNNER = r'''
import json
import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.nodes.artifact_commit import artifact_commit
from src.nodes.task_controller import route_task_controller, task_controller
from src.persistence import SQLitePersistence
from src.recovery.policy import decide_recovery_action
from src.runtime_config import execution_config
from src.state import State


class PipelineState(State, total=False):
    assessment: dict


def worker_stub(state):
    task_id = state["current_task"]["task_id"]
    return {
        "current_result": {
            "task_id": task_id,
            "status": "COMPLETED",
            "text_output": task_id,
            "citations": [],
        }
    }


def review_stub(state):
    artifact = state["current_result"]
    should_rework = (
        SCENARIO == "local_rework"
        and artifact["task_id"] == "T3"
        and artifact["attempt_no"] == 1
    )
    if should_rework:
        assessment = {
            "status": "FAILED",
            "artifact_id": artifact["artifact_id"],
            "issues": [
                {
                    "code": "TOO_SHORT",
                    "category": "CONTENT_DEFECT",
                    "description": "Deliberate first-attempt failure",
                    "suggestion": "Retry T3 only",
                    "responsible_handler": "worker_agent",
                }
            ],
        }
    else:
        assessment = {
            "status": "PASS",
            "artifact_id": artifact["artifact_id"],
            "issues": [],
        }
    return {"assessment": assessment}


def policy_node(state):
    return decide_recovery_action(state, state["assessment"])


def summarizer_stub(state):
    return {
        "final_result": {
            "task_ids": [item["task_id"] for item in state["results"]]
        }
    }


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("TaskController", task_controller)
    graph.add_node("Worker", worker_stub)
    graph.add_node("ArtifactCommit", artifact_commit)
    graph.add_node("Review", review_stub)
    graph.add_node("Policy", policy_node)
    graph.add_node("Summarizer", summarizer_stub)
    graph.add_edge(START, "TaskController")
    graph.add_conditional_edges(
        "TaskController",
        route_task_controller,
        {
            "Worker": "Worker",
            "Summarizer": "Summarizer",
            "NeedsUserInput": END,
        },
    )
    graph.add_edge("Worker", "ArtifactCommit")
    graph.add_edge("ArtifactCommit", "Review")
    graph.add_edge("Review", "Policy")
    graph.add_edge("Policy", "TaskController")
    graph.add_edge("Summarizer", END)
    return graph


def initial_state(job_id, task_count):
    return {
        "job_id": job_id,
        "user_id": "u1",
        "tasks": [{"task_id": f"T{i}"} for i in range(1, task_count + 1)],
        "results": [],
        "artifacts": {},
        "active_artifact_ids": {},
        "review_records": [],
    }


SCENARIO = sys.argv[1]
data_dir = Path(sys.argv[2])
graph = build_pipeline()

if SCENARIO in {"nine_tasks", "local_rework"}:
    app = graph.compile()
    final = app.invoke(initial_state(SCENARIO, 9))
else:
    config = execution_config({"configurable": {"thread_id": "resume-job"}})
    with SQLitePersistence.open(data_dir) as persistence:
        paused = graph.compile(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            interrupt_after=["ArtifactCommit"],
        )
        paused.invoke(initial_state("resume-job", 2), config)
        snapshot = paused.get_state(config)
        first_artifact_id = snapshot.values["active_artifact_ids"]["T1"]

    with SQLitePersistence.open(data_dir) as persistence:
        resumed = graph.compile(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
        )
        final = resumed.invoke(None, config)
    final["first_artifact_id"] = first_artifact_id

print("PIPELINE_RESULT=" + json.dumps(final, ensure_ascii=False, default=str))
'''


def _run_pipeline(scenario: str, tmp_path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER, scenario, str(tmp_path)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("PIPELINE_RESULT=")
    )
    return json.loads(result_line.removeprefix("PIPELINE_RESULT="))


def test_nine_tasks_execute_in_order_without_skips(tmp_path):
    final = _run_pipeline("nine_tasks", tmp_path)

    expected = [f"T{i}" for i in range(1, 10)]
    assert [item["task_id"] for item in final["results"]] == expected
    assert final["final_result"]["task_ids"] == expected
    assert all(
        record["status"] == "PASSED"
        for record in final["task_records"].values()
    )
    assert len(final["artifacts"]) == 9


def test_local_rework_creates_new_t3_artifact_without_rerunning_prior_tasks(tmp_path):
    final = _run_pipeline("local_rework", tmp_path)
    attempts_by_task = {}
    for artifact in final["artifacts"].values():
        attempts_by_task.setdefault(artifact["task_id"], []).append(
            artifact["attempt_no"]
        )

    assert attempts_by_task["T1"] == [1]
    assert attempts_by_task["T2"] == [1]
    assert sorted(attempts_by_task["T3"]) == [1, 2]
    assert [item["task_id"] for item in final["results"]] == [
        f"T{i}" for i in range(1, 10)
    ]


def test_sqlite_resume_reuses_committed_artifact(tmp_path):
    final = _run_pipeline("sqlite_resume", tmp_path)

    assert final["active_artifact_ids"]["T1"] == final["first_artifact_id"]
    assert len(final["artifacts"]) == 2
    assert len(
        {item["execution_id"] for item in final["artifacts"].values()}
    ) == 2
    assert [item["task_id"] for item in final["results"]] == ["T1", "T2"]

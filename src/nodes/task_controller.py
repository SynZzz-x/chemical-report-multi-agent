"""Deterministic task progression for the serial report workflow."""

from __future__ import annotations

from typing import Any

from src.workflow_records import (
    all_tasks_passed,
    ensure_task_records,
    first_runnable_task,
    make_execution_id,
    set_task_status,
)


def task_controller(state, config=None, **kwargs) -> dict[str, Any]:
    """Select exactly one runnable task or authorize final summarization."""
    tasks = list(state.get("tasks") or [])
    records = ensure_task_records(state)
    if all_tasks_passed(tasks, records):
        return {
            "task_records": records,
            "controller_action": "SUMMARIZE",
        }

    selected = first_runnable_task(tasks, records)
    if selected is None:
        return {
            "task_records": records,
            "controller_action": "NEEDS_USER_INPUT",
            "pending_user_action": {
                "category": "TASK_DEPENDENCY_BLOCKED",
                "issues": [],
            },
        }

    index, task = selected
    task_id = str(task.get("task_id") or f"T{index + 1}")
    if records[task_id].get("status") == "BLOCKED":
        return {
            "task_records": records,
            "current_task": task,
            "cursor": index,
            "controller_action": "NEEDS_USER_INPUT",
        }

    attempt_no = int(records[task_id].get("attempt_count", 0) or 0) + 1
    execution_id = make_execution_id(
        str(state.get("job_id") or "job"), task_id, attempt_no
    )
    records = set_task_status(
        records,
        task_id,
        "RUNNING",
        attempt_count=attempt_no,
    )
    return {
        "task_records": records,
        "current_task": task,
        "cursor": index,
        "current_execution_id": execution_id,
        "current_result": {},
        "controller_action": "DISPATCH",
    }


def route_task_controller(state, config=None, **kwargs) -> str:
    action = str(state.get("controller_action") or "NEEDS_USER_INPUT")
    return {
        "DISPATCH": "Worker",
        "SUMMARIZE": "Summarizer",
    }.get(action, "NeedsUserInput")

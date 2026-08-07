"""Pure workflow record helpers used by graph nodes and compatibility adapters."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal


TaskStatus = Literal[
    "PENDING",
    "RUNNING",
    "REVISE_REQUIRED",
    "EVIDENCE_REQUIRED",
    "PASSED",
    "BLOCKED",
]


def _task_id(task: dict[str, Any], sequence: int) -> str:
    return str(task.get("task_id") or f"T{sequence + 1}")


def ensure_task_records(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a ledger for current tasks, migrating legacy result-only states."""
    existing = {
        str(key): dict(value)
        for key, value in (state.get("task_records") or {}).items()
        if isinstance(value, dict)
    }
    passed = {
        str(item.get("task_id"))
        for item in state.get("results") or []
        if isinstance(item, dict) and item.get("task_id") is not None
    }

    records: dict[str, dict[str, Any]] = {}
    for sequence, task in enumerate(state.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        task_id = _task_id(task, sequence)
        prior = existing.get(task_id, {})
        records[task_id] = {
            **prior,
            "task_id": task_id,
            "sequence": sequence,
            "status": "PASSED" if task_id in passed else prior.get("status", "PENDING"),
            "attempt_count": int(prior.get("attempt_count", 0) or 0),
            "active_artifact_id": prior.get("active_artifact_id"),
            "dependencies": list(
                task.get("dependencies") or prior.get("dependencies") or []
            ),
        }
    return records


def first_runnable_task(
    tasks: list[dict[str, Any]],
    task_records: dict[str, dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    """Select the first unfinished task whose declared dependencies passed."""
    passed = {
        task_id
        for task_id, record in task_records.items()
        if record.get("status") == "PASSED"
    }
    for index, task in enumerate(tasks or []):
        if not isinstance(task, dict):
            continue
        task_id = _task_id(task, index)
        record = task_records.get(task_id, {})
        if record.get("status") == "PASSED":
            continue
        dependencies = [str(value) for value in record.get("dependencies") or []]
        if all(dependency in passed for dependency in dependencies):
            return index, task
    return None


def all_tasks_passed(
    tasks: list[dict[str, Any]],
    task_records: dict[str, dict[str, Any]],
) -> bool:
    """Return true only for a non-empty plan whose every task passed."""
    if not tasks:
        return False
    return all(
        isinstance(task, dict)
        and task_records.get(_task_id(task, index), {}).get("status") == "PASSED"
        for index, task in enumerate(tasks)
    )


def set_task_status(
    records: dict[str, dict[str, Any]],
    task_id: str,
    status: TaskStatus,
    **changes: Any,
) -> dict[str, dict[str, Any]]:
    """Return a copied ledger with one task status transition applied."""
    if task_id not in records:
        raise KeyError(f"Unknown task_id: {task_id}")
    updated = {key: dict(value) for key, value in records.items()}
    updated[task_id] = {**updated[task_id], "status": status, **changes}
    return updated


def make_execution_id(job_id: str, task_id: str, attempt_no: int) -> str:
    """Create a replay-stable identifier for one task attempt."""
    digest = hashlib.sha256(
        f"{job_id}|{task_id}|{attempt_no}".encode("utf-8")
    ).hexdigest()[:24]
    return f"execution_{digest}"


def build_artifact(
    state: dict[str, Any], current_result: dict[str, Any]
) -> dict[str, Any]:
    """Convert a legacy Worker result into an immutable, versioned Artifact."""
    task_id = str(current_result["task_id"])
    execution_id = str(state["current_execution_id"])
    artifact_id = "artifact_" + hashlib.sha256(
        execution_id.encode("utf-8")
    ).hexdigest()[:24]
    attempt_no = int(state["task_records"][task_id]["attempt_count"])
    return {
        **dict(current_result),
        "artifact_id": artifact_id,
        "task_id": task_id,
        "attempt_no": attempt_no,
        "artifact_type": "report_section",
        "producer": "worker",
        "content": current_result.get("text_output", ""),
        "evidence_refs": list(current_result.get("citations") or []),
        "source_scope": list(current_result.get("sources_used") or []),
        "created_at": current_result.get("generated_at")
        or datetime.now().astimezone().isoformat(timespec="seconds"),
        "supersedes": (state.get("active_artifact_ids") or {}).get(task_id),
        "execution_id": execution_id,
    }

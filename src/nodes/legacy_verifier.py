"""Compatibility adapters for checkpoints paused at the pre-Artifact Verifier."""

from __future__ import annotations

from typing import Any

from src.nodes.artifact_commit import artifact_commit
from src.nodes.quality_review import quality_review
from src.nodes.verifier_manual import verifier_manual
from src.workflow_records import ensure_task_records, make_execution_id, set_task_status


def _task_id(state: dict[str, Any], current: dict[str, Any]) -> str:
    if current.get("task_id"):
        return str(current["task_id"])
    task = state.get("current_task") or {}
    if isinstance(task, dict) and task.get("task_id"):
        return str(task["task_id"])
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], dict):
        return str(tasks[cursor].get("task_id") or f"T{cursor + 1}")
    raise ValueError("Legacy Verifier checkpoint has no task-scoped current_result")


def _upgrade_legacy_current_result(state, store=None):
    """Commit the old Worker payload before handing it to a current verifier."""
    current = dict(state.get("current_result") or {})
    if current.get("artifact_id"):
        return dict(state), {}

    task_id = _task_id(state, current)
    current["task_id"] = task_id
    records = ensure_task_records(state)
    attempt_no = max(int(records[task_id].get("attempt_count", 0) or 0), 1)
    records = set_task_status(
        records,
        task_id,
        "RUNNING",
        attempt_count=attempt_no,
    )
    execution_id = str(
        state.get("current_execution_id")
        or make_execution_id(str(state.get("job_id") or "job"), task_id, attempt_no)
    )
    prepared = {
        **state,
        "current_result": current,
        "task_records": records,
        "artifacts": dict(state.get("artifacts") or {}),
        "active_artifact_ids": dict(state.get("active_artifact_ids") or {}),
        "current_execution_id": execution_id,
    }
    committed = artifact_commit(prepared, store=store)
    upgraded = {**prepared, **committed}
    migration_update = {
        **committed,
        "current_execution_id": execution_id,
    }
    return upgraded, migration_update


def legacy_auto_verifier(state, config, store=None, **kwargs):
    """Resume an old automatic Verifier checkpoint through ArtifactCommit."""
    upgraded, migration_update = _upgrade_legacy_current_result(state, store=store)
    review_update = quality_review(upgraded, config, store=store, **kwargs)
    return {**migration_update, **review_update}


def legacy_manual_verifier(state, config, store=None, **kwargs):
    """Resume an old manual Verifier checkpoint through ArtifactCommit."""
    upgraded, migration_update = _upgrade_legacy_current_result(state, store=store)
    review_update = verifier_manual(upgraded, config, store=store, **kwargs)
    return {**migration_update, **review_update}

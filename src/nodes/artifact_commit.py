"""Commit legacy Worker output as one immutable, versioned Artifact."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
else:
    BaseStore = Any

from src.workflow_records import build_artifact, set_task_status
from src.workflow_store import WorkflowRecordStore


def artifact_commit(state, store: BaseStore | None = None, **kwargs):
    current = state.get("current_result") or {}
    if not current or not current.get("task_id"):
        raise ValueError("Worker completed without a task-scoped current_result")

    candidate = build_artifact(state, current)
    artifacts = dict(state.get("artifacts") or {})
    artifact = dict(artifacts.get(candidate["artifact_id"]) or candidate)
    artifacts[artifact["artifact_id"]] = artifact

    task_id = str(artifact["task_id"])
    active = {
        **(state.get("active_artifact_ids") or {}),
        task_id: artifact["artifact_id"],
    }
    records = set_task_status(
        state["task_records"],
        task_id,
        "RUNNING",
        active_artifact_id=artifact["artifact_id"],
    )

    if store is not None:
        WorkflowRecordStore(
            store,
            str(state.get("user_id") or ""),
            str(state.get("job_id") or ""),
        ).put_artifact(artifact)

    return {
        "current_result": artifact,
        "artifacts": artifacts,
        "active_artifact_ids": active,
        "task_records": records,
    }

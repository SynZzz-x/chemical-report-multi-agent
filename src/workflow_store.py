"""LangGraph Store persistence for auditable report workflow records."""

from __future__ import annotations

from typing import Any

from langgraph.store.base import BaseStore


class WorkflowRecordStore:
    def __init__(self, store: BaseStore, user_id: str, job_id: str):
        if not user_id or not job_id:
            raise ValueError("user_id and job_id are required")
        self.store = store
        self.root = (user_id, "report_jobs", job_id)

    def _put(self, record_type: str, key: str, value: dict[str, Any]) -> None:
        self.store.put((*self.root, record_type), key, value, index=False)

    def put_artifact(self, artifact: dict[str, Any]) -> None:
        self._put("artifacts", str(artifact["artifact_id"]), artifact)

    def put_review(self, review: dict[str, Any]) -> None:
        self._put("reviews", str(review["review_id"]), review)

    def put_report_manifest(self, manifest: dict[str, Any]) -> None:
        self._put("reports", str(manifest["report_id"]), manifest)

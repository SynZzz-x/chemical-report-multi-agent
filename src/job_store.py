from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
else:
    BaseStore = Any


JobStatus = Literal["created", "running", "waiting", "completed", "failed"]
VALID_JOB_STATUSES = {"created", "running", "waiting", "completed", "failed"}


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def job_namespace(user_id: str) -> tuple[str, str]:
    if not user_id.strip():
        raise ValueError("user_id must not be empty")
    return user_id, "jobs"


def interrupt_from_snapshot(snapshot: Any) -> Any | None:
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return getattr(interrupts[0], "value", interrupts[0])
    return None


class JobStore:
    def __init__(self, store: BaseStore):
        self.store = store

    def create_job(
        self,
        *,
        user_id: str,
        conversation_id: str,
        job_id: str,
        title: str,
        verifier_mode: str,
        web_authorized: bool = False,
        ui_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        record = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "job_id": job_id,
            "title": " ".join(title.split())[:120] or "未命名任务",
            "verifier_mode": verifier_mode,
            "web_authorized": web_authorized is True,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "pending_interrupt": None,
            "ui_messages": list(ui_messages or []),
            "report_paths": [],
        }
        self.store.put(job_namespace(user_id), job_id, record, index=False)
        return record

    def get_job(self, user_id: str, job_id: str) -> dict[str, Any] | None:
        item = self.store.get(job_namespace(user_id), job_id)
        if item is None:
            return None
        value = dict(item.value)
        if value.get("user_id") != user_id:
            return None
        return value

    def list_jobs(self, user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        items = self.store.search(job_namespace(user_id), limit=limit)
        records = [
            dict(item.value)
            for item in items
            if item.value.get("user_id") == user_id
        ]
        return sorted(
            records,
            key=lambda record: record.get("updated_at", ""),
            reverse=True,
        )

    def update_job(
        self,
        user_id: str,
        job_id: str,
        **changes: Any,
    ) -> dict[str, Any]:
        immutable_fields = {
            "user_id",
            "conversation_id",
            "job_id",
            "verifier_mode",
            "web_authorized",
            "created_at",
        }
        immutable_changes = immutable_fields.intersection(changes)
        if immutable_changes:
            names = ", ".join(sorted(immutable_changes))
            raise ValueError(f"Cannot update immutable job fields: {names}")

        record = self.get_job(user_id, job_id)
        if record is None:
            raise KeyError(f"Unknown job_id for user {user_id}: {job_id}")

        status = changes.get("status")
        if status is not None and status not in VALID_JOB_STATUSES:
            raise ValueError(f"Invalid job status: {status}")

        record.update(changes)
        record["updated_at"] = utc_now_iso()
        self.store.put(job_namespace(user_id), job_id, record, index=False)
        return record

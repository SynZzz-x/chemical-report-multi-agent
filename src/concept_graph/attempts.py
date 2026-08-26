"""Stable checkpoint keys for bounded semantic concept-graph extraction."""

from __future__ import annotations


def concept_graph_attempt_key(task_id: str, task_revision: int) -> str:
    return f"{str(task_id or '').strip()}:concept_graph:t{int(task_revision or 1)}"

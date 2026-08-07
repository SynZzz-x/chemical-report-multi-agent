"""Deterministic checks that run before semantic LLM review."""

from __future__ import annotations

from typing import Any

from .models import ReviewIssue


def _issue(
    code: str,
    category: str,
    description: str,
    instruction: str,
    handler: str,
    severity: str = "major",
) -> ReviewIssue:
    return ReviewIssue(
        code=code,
        category=category,
        severity=severity,
        description=description,
        responsible_handler=handler,
        revision_instruction=instruction,
    )


def _requires_table(task: dict[str, Any]) -> bool:
    description = str(task.get("task_description") or "")
    return task.get("generate_table") is True or any(
        marker in description for marker in ("表格", "数据表", "生成表")
    )


def _requires_figure(task: dict[str, Any]) -> bool:
    description = str(task.get("task_description") or "")
    return task.get("generate_figure") is True or any(
        marker in description for marker in ("趋势图", "因果图", "流程图", "生成图")
    )


def validate_artifact(
    task: dict[str, Any],
    artifact: dict[str, Any],
    *,
    active_artifact_id: str | None,
) -> list[ReviewIssue]:
    """Return contract/asset issues without making a routing decision."""
    issues: list[ReviewIssue] = []
    task_id = str(task.get("task_id") or "")
    artifact_task_id = str(artifact.get("task_id") or "")
    artifact_id = str(artifact.get("artifact_id") or "")

    if not task_id or artifact_task_id != task_id:
        issues.append(
            _issue(
                "ARTIFACT_TASK_MISMATCH",
                "REVIEW_FAILURE",
                "Artifact is not bound to the task being reviewed.",
                "Rebuild the review input from the active task and Artifact.",
                "quality_review",
                "error",
            )
        )
    if not artifact_id or artifact_id != str(active_artifact_id or ""):
        issues.append(
            _issue(
                "STALE_ARTIFACT",
                "REVIEW_FAILURE",
                "Artifact is not the active version for this task.",
                "Review only the current active Artifact version.",
                "quality_review",
                "error",
            )
        )
    if artifact.get("status") == "FAILED" or artifact.get("error"):
        issues.append(
            _issue(
                "WORKER_EXECUTION_FAILED",
                "WORKER_FAILURE",
                str(artifact.get("error") or "Worker returned a failed Artifact."),
                "Retry the current Worker attempt without changing the plan.",
                "section_writing",
                "error",
            )
        )
    if _requires_table(task) and not artifact.get("tables"):
        issues.append(
            _issue(
                "MISSING_TABLE",
                "CONTENT_DEFECT",
                "The task requires a table, but the Artifact contains none.",
                "Generate the required table for the current task.",
                "section_writing",
            )
        )
    if _requires_figure(task) and not artifact.get("figures"):
        issues.append(
            _issue(
                "MISSING_FIGURE",
                "VISUAL_DEFECT",
                "The task requires a figure, but the Artifact contains none.",
                "Generate only the missing figure from traceable evidence.",
                "visualization",
            )
        )
    citations = artifact.get("citations") or []
    malformed_citation = any(
        not isinstance(citation, dict)
        or not str(citation.get("evidence_id") or "").strip()
        for citation in citations
    )
    if malformed_citation:
        issues.append(
            _issue(
                "INVALID_CITATION",
                "EVIDENCE_GAP",
                "At least one citation lacks a stable evidence identifier.",
                "Rebuild citations from normalized RAG or authorized web evidence.",
                "evidence",
            )
        )
    return issues

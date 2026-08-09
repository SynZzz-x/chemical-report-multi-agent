"""Pure node-progress summaries shared with the Streamlit surface."""

from __future__ import annotations

from typing import Any


def summarize_step(node: str, delta: dict[str, Any]) -> str:
    if node == "Intake":
        return "已解析当前用户输入"
    if node == "Planner":
        tasks = delta.get("tasks") or []
        cursor = int(delta.get("cursor") or 0)
        return f"任务数：{len(tasks)}，当前序号：{cursor + 1 if tasks else 0}"
    if node == "Worker":
        result = delta.get("current_result") or {}
        name = result.get("section_name") or result.get("task_id") or "当前任务"
        status = result.get("status") or "-"
        return f"{name}：{status}"
    if node in {"Verifier", "QualityReview"}:
        assessment = delta.get("assessment") or {}
        status = str(assessment.get("status") or "UNKNOWN")
        categories = list(
            dict.fromkeys(
                str(issue.get("category"))
                for issue in assessment.get("issues") or []
                if isinstance(issue, dict) and issue.get("category")
            )
        )
        if categories:
            return f"审核状态：{status}，问题类型：{','.join(categories)}"
        return f"审核状态：{status}"
    if node == "DecisionPolicy":
        action = str(delta.get("workflow_action") or "待定")
        return f"恢复动作：{action}"
    if node == "Summarizer":
        return "报告及评价已生成"
    if node == "Exit":
        return "流程结束"
    return ""


def task_progress_view(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Project the task ledger and latest review into plan order."""
    records = state.get("task_records") or {}
    reviews = [
        review
        for review in state.get("review_records") or []
        if isinstance(review, dict)
    ]
    view = []
    for index, task in enumerate(state.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or f"T{index + 1}")
        record = records.get(task_id) or {}
        active_artifact_id = record.get("active_artifact_id") or (
            state.get("active_artifact_ids") or {}
        ).get(task_id)
        latest_review = next(
            (
                review
                for review in reversed(reviews)
                if str(review.get("task_id") or "") == task_id
                and review.get("artifact_id") == active_artifact_id
            ),
            {},
        )
        view.append(
            {
                "task_id": task_id,
                "task_name": task.get("task_name") or task_id,
                "status": record.get("status") or "PENDING",
                "attempt_count": int(record.get("attempt_count", 0) or 0),
                "active_artifact_id": active_artifact_id,
                "latest_review_id": latest_review.get("review_id"),
                "latest_review_status": latest_review.get("status"),
                "latest_review_issues": list(latest_review.get("issues") or []),
            }
        )
    return view


def report_status_view(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return one explicit status for every supported report format."""
    outputs = (state.get("report_manifest") or {}).get("outputs") or {}
    view = {}
    for name in ("md", "rewritten_md", "docx", "pdf"):
        outcome = outputs.get(name)
        if isinstance(outcome, dict):
            view[name] = {
                "status": outcome.get("status") or "NOT_ATTEMPTED",
                "path": outcome.get("path"),
                "error": outcome.get("error"),
            }
        else:
            view[name] = {
                "status": "NOT_ATTEMPTED",
                "path": None,
                "error": None,
            }
    return view

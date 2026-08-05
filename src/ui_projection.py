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
    if node == "Verifier":
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

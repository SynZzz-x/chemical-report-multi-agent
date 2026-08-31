"""Pure node-progress summaries shared with the Streamlit surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_SAFE_FATAL_SUMMARIES = {
    "VERIFIER_UNAVAILABLE": "审核服务不可用",
}


def terminal_job_ui_state(
    job: Mapping[str, Any] | None,
    outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project graph-terminal status while preserving runner resubmission."""

    source = outcome if outcome is not None else job
    if not isinstance(source, Mapping) or str(source.get("status") or "") != "failed":
        return {"is_terminal": False, "message": None}

    fatal = source.get("fatal_system_error") or {}
    origin = str(fatal.get("origin") or "") if isinstance(fatal, Mapping) else ""
    if origin != "graph":
        return {
            "is_terminal": False,
            "message": "上次执行未完成，可以重新提交。" if origin == "runner" else None,
        }

    diagnostic_code = (
        str(fatal.get("diagnostic_code") or "")
        if isinstance(fatal, Mapping)
        else ""
    )
    summary = _SAFE_FATAL_SUMMARIES.get(diagnostic_code, "系统工作流无法继续")
    return {
        "is_terminal": True,
        "message": (
            "当前报告任务已停止（"
            f"{summary}）。请新建报告任务后继续。"
        ),
    }


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
        decision = delta.get("failure_decision") or {}
        failure_class = str(decision.get("failure_class") or "")
        subtype = str(decision.get("subtype") or "")
        if failure_class == "DEGRADABLE_QUALITY":
            return f"已记录非阻塞交付限制：{subtype or 'DEGRADABLE_QUALITY'}"
        if failure_class == "FATAL_SYSTEM":
            return f"系统错误：{subtype or 'FATAL_SYSTEM'}"
        action = str(delta.get("workflow_action") or "待定")
        return f"恢复动作：{action}"
    if node == "Summarizer":
        return "报告及评价已生成"
    if node == "Exit":
        return "流程结束"
    return ""

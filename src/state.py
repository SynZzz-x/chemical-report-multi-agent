from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langgraph.graph.message import BaseMessage, add_messages


def merge_docs(
    existing: List[Dict[str, Any]] | None,
    incoming: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """增量合并附件。

    合并键优先级：file_id -> resource_id -> path。
    相同附件再次写入时，使用新字段覆盖旧字段；不同附件保持原顺序。
    """
    merged: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}

    for doc in [*(existing or []), *(incoming or [])]:
        if not isinstance(doc, dict):
            continue

        key = str(
            doc.get("file_id")
            or doc.get("resource_id")
            or doc.get("path")
            or ""
        ).strip()

        if not key:
            continue

        normalized = dict(doc)
        # 过渡期同时保留 file_id/resource_id，兼容旧 Planner/Worker 代码。
        if normalized.get("file_id") and not normalized.get("resource_id"):
            normalized["resource_id"] = normalized["file_id"]
        if normalized.get("resource_id") and not normalized.get("file_id"):
            normalized["file_id"] = normalized["resource_id"]

        if key in index:
            position = index[key]
            merged[position] = {**merged[position], **normalized}
        else:
            index[key] = len(merged)
            merged.append(normalized)

    return merged


class State(TypedDict, total=False):
    """顶层 LangGraph 状态。

    当前阶段仍保留串行 cursor 工作流；后续引入 Scheduler 后，再将
    cursor/results 逐步替换为 task_status/candidate_results/committed_results。
    """

    # 身份与作用域
    user_id: str
    conversation_id: str
    job_id: str

    # 当前输入及对话
    current_user_input: str
    messages: Annotated[List[BaseMessage], add_messages]

    # 当前 Job 的附件；调用方只写入本轮新增附件，由 reducer 合并。
    docs: Annotated[List[Dict[str, Any]], merge_docs]

    # 当前串行工作流
    tasks: List[Dict[str, Any]]
    cursor: int
    current_result: Dict[str, Any]
    results: List[Dict[str, Any]]

    decision: Literal[
        "RETRY_WORKER",
        "REPLAN",
        "FULL_REPLAN",
        "NEXT",
        "DONE",
        "END",
    ]

    feedback: Dict[str, Any]
    final_result: Dict[str, Any]
    planner_action: str

    # 兼容现有节点，后续再逐步拆除或重构。
    metadata: Dict[str, Any]
    worker_state: Dict[str, Any]
    guidance: Dict[str, Any]
    task_retry_count: Dict[str, int]
    replan_count: int
    workflow_action: str
    plan_revision: int
    task_revisions: Dict[str, int]
    evidence_recovery_count: Dict[str, int]
    task_patch_count: Dict[str, int]
    job_patch_count: int
    pending_user_action: Dict[str, Any]
    plan_patch_history: List[Dict[str, Any]]
    verification_warnings: List[Dict[str, Any]]
    current_task: Dict[str, Any]
    all_results: List[Dict[str, Any]]
    tool_execution_history: List[Dict[str, Any]]
    full_replan_previous_task_ids: List[str]
    full_replan_reason: str
    full_replan_candidate_tasks: List[Dict[str, Any]]
    task_id_registry: List[str]


class ConfigSchema(TypedDict, total=False):
    """模型参数定义。"""

    base_url: str
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float

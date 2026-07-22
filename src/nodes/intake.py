from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from ..llm import get_llm
from ..state import State, merge_docs

# ============================================================
# Intake 节点
# 当前阶段只解析“本轮输入”，不再把当前 Job 的全部历史用户消息拼接后重解析。
# ============================================================

DECISION_ASK_USER = "END"
DECISION_NEXT = "NEXT"
DECISION_ERROR = "ERROR"

DECISION_REPLAN = "REPLAN"
DECISION_RETRY_VERIFIER = "RETRY_VERIFIER"


def extract_initial_request(
    messages: List[Dict[str, Any]],
    docs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """兼容旧测试和 CLI 的历史辅助函数。

    该函数仍会合并历史 user 消息，但主流程 ``intake`` 已不再调用它。
    """
    all_user_content: List[str] = []
    user_resources: List[Dict[str, Any]] = []

    def _add_resource(resource: Dict[str, Any]) -> None:
        resource_id = resource.get("resource_id")
        resource_key = str(resource.get("name", "")) + str(resource.get("path", ""))

        exists = any(
            (
                resource_id is not None
                and item.get("resource_id") == resource_id
            )
            or (
                str(item.get("name", "")) + str(item.get("path", ""))
                == resource_key
            )
            for item in user_resources
        )
        if exists:
            return

        user_resources.append(
            {
                "name": resource.get("name"),
                "type": resource.get("type", "unknown"),
                "path": resource.get("path"),
                "resource_id": resource.get("resource_id"),
            }
        )

    for message in messages or []:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
            resources = (
                message.get("resources", [])
                if isinstance(message.get("resources"), list)
                else []
            )
        else:
            role = getattr(message, "type", None) or getattr(message, "role", None)
            content = getattr(message, "content", None)
            resources = getattr(message, "additional_kwargs", {}).get("resources", [])

        if role not in ("user", "human"):
            continue

        if content:
            all_user_content.append(str(content))
        for resource in resources or []:
            if isinstance(resource, dict):
                _add_resource(resource)

    for resource in docs or []:
        if isinstance(resource, dict):
            _add_resource(resource)

    return {
        "raw_request": "\n".join(all_user_content),
        "resources": user_resources,
    }


def extract_current_request(state: State) -> Dict[str, Any]:
    """提取本轮用户输入和当前 Job 已登记的附件。

    ``current_user_input`` 是当前事件；``messages`` 只作为 LangGraph 的语义历史，
    Intake 不再遍历并无差别拼接全部历史用户消息。
    """
    raw_request = str(state.get("current_user_input") or "").strip()
    resources = merge_docs([], state.get("docs") or [])
    return {
        "raw_request": raw_request,
        "resources": resources,
    }


def _load_intake_refine_prompt() -> str:
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "../prompts/intake_refine.md")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError:
        return (
            "改善 Intake 节点的输出，确保 JSON 格式正确，包含必要字段"
            "（user_intent, task_type, title, doc_length）。"
        )


def _load_intake_prompt() -> str:
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "../prompts/intake_to_planner.md")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError:
        return "你是 Intake 节点。严格输出 JSON，并使用变量 {user_input} 作为输入。"


def _extract_first_json(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Intake 模型输出中未找到 JSON 对象")
    return match.group(0)


def llm_parse_user_need(raw_request: str, config: RunnableConfig) -> Dict[str, Any]:
    """调用 LLM，将本轮自然语言输入解析为结构化需求。"""
    model = get_llm(config)
    system_template = _load_intake_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "请生成 JSON 输出"),
        ]
    )
    messages = prompt.format_messages(user_input=raw_request)
    response = model.invoke(messages, config=config)
    initial_output = _extract_first_json(str(response.content))

    try:
        pre_parsed = json.loads(initial_output)
        if pre_parsed.get("is_chat"):
            return pre_parsed
    except (TypeError, json.JSONDecodeError):
        pass

    refine_template = _load_intake_refine_prompt()
    refine_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", refine_template),
            ("human", "{init_output}"),
        ]
    )
    refine_messages = refine_prompt.format_messages(init_output=initial_output)
    refined_response = model.invoke(refine_messages, config=config)

    refined_text = str(refined_response.content)
    tagged_json = re.search(r"<json>((.|\n)*?)</json>", refined_text)
    payload = tagged_json.group(1) if tagged_json else _extract_first_json(refined_text)

    parsed = json.loads(payload)
    parsed.setdefault("constraints", [])
    parsed.setdefault("missing_fields", [])
    return parsed


def build_task_spec(
    parsed_request: Dict[str, Any],
    resources: List[Dict[str, Any]],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """生成发送给 Planner 的结构化 INTAKE_SUMMARY。"""
    content = {
        "from": "Intake",
        "to": "Planner",
        "type": "INTAKE_SUMMARY",
        "user_intent": parsed_request.get("user_intent"),
        "task_type": parsed_request.get("task_type"),
        "title": parsed_request.get("title"),
        "doc_length": parsed_request.get("doc_length"),
        "constraints": parsed_request.get("constraints", []),
        "style": parsed_request.get("style"),
        "output_format": parsed_request.get("format"),
        "sections": parsed_request.get("sections", []),
        "core_content": parsed_request.get("core_content"),
        "resources": resources,
    }
    return {
        "decision": DECISION_NEXT,
        "messages": [AIMessage(content=json.dumps(content, ensure_ascii=False))],
        "metadata": metadata or {},
    }


def intake(state: State, config: RunnableConfig, **kwargs: Any) -> Dict[str, Any]:
    """解析本轮用户输入，并向 Planner 发送 INTAKE_SUMMARY。"""
    metadata = dict(state.get("metadata") or {})
    metadata.setdefault("user_id", state.get("user_id"))
    metadata.setdefault("conversation_id", state.get("conversation_id"))
    metadata.setdefault("job_id", state.get("job_id"))
    metadata.setdefault("session_id", state.get("conversation_id") or "unknown_session")
    metadata.setdefault("request_id", state.get("job_id") or "unknown_request")
    metadata.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))

    initial = extract_current_request(state)
    raw_request = initial.get("raw_request", "")
    if not raw_request:
        return {
            "decision": DECISION_ASK_USER,
            "messages": [AIMessage(content="请输入本轮报告需求或补充说明。")],
            "metadata": metadata,
        }

    parsed = llm_parse_user_need(raw_request, config)

    # 注意：当前 graph.py 仍是 Intake -> Planner 固定边。真正的闲聊分流应在
    # 下一阶段通过 IntentRouter/条件边解决；这里先保留兼容输出。
    if parsed.get("is_chat"):
        return {
            "decision": DECISION_ASK_USER,
            "messages": [AIMessage(content=parsed.get("response", ""))],
            "metadata": metadata,
        }

    return build_task_spec(
        parsed,
        initial.get("resources", []),
        metadata=metadata,
    )

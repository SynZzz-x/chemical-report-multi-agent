import os
import json
import re
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from typing import Dict, Any, List

from ..state import State, merge_docs
from ..llm import get_llm

logger = logging.getLogger(__name__)

"""
Planner 节点：
- 职责：读取最新一条发给 Planner 的指令（messages content 为 JSON，且 `to="Planner"`），
  根据指令类型（INTAKE_SUMMARY / PROCEED / FULL_REPLAN）选择流程，生成任务列表与下游消息。
- 输入：
  - 来自 Intake 的 INTAKE_SUMMARY：初始化任务规划
  - 来自 Verifier 的 PROCEED：根据决策推进 cursor
  - 来自人工 Verifier 的 FULL_REPLAN：暂存新计划，等待用户确认后才重置执行状态
- 输出：
  - 更新 `tasks` 与 `cursor`
  - 设置 `planner_action`
  - 若为 PROCEED，生成发给 Worker 的 `PLAN_RESULT` 消息
"""


def _latest_to_planner(messages):
    """获取 messages 中最新一条发给 Planner 的指令，解析其 JSON content。"""
    for m in reversed(messages or []):
        if isinstance(m, dict):
            role = str(m.get("role") or m.get("type") or "").lower()
        else:
            role = str(
                getattr(m, "type", None) or getattr(m, "role", None) or ""
            ).lower()
        if role not in {"ai", "assistant"}:
            continue
        try:
            c = (
                str(m.get("content") or "")
                if isinstance(m, dict)
                else str(getattr(m, "content", "") or "")
            ).strip()
            parsed = json.loads(c)
            if isinstance(parsed, dict) and parsed.get("to") == "Planner":
                return parsed
        except Exception:
            continue
    return None


def _is_user_full_replan(parsed: Dict[str, Any] | None) -> bool:
    """Accept full replanning only from the manual, user-feedback verifier."""
    return bool(
        isinstance(parsed, dict)
        and parsed.get("from") == "Verifier"
        and parsed.get("to") == "Planner"
        and str(parsed.get("type") or "").upper() == "FULL_REPLAN"
    )


def _task_ids(tasks: List[Dict[str, Any]]) -> list[str]:
    return [
        str(task.get("task_id") or "").strip()
        for task in tasks or []
        if isinstance(task, dict) and str(task.get("task_id") or "").strip()
    ]


def _next_task_number(task_ids: List[str]) -> int:
    numbers = [
        int(task_id[1:])
        for task_id in task_ids
        if re.fullmatch(r"T\d+", task_id)
    ]
    return max(numbers, default=0) + 1


def _normalize_replacement_tasks(
    candidate_tasks: List[Dict[str, Any]], previous_task_ids: List[str]
) -> List[Dict[str, Any]]:
    """Give a replacement plan stable IDs that cannot collide with prior work."""
    if not isinstance(candidate_tasks, list) or not candidate_tasks:
        raise ValueError("Replacement plan must contain at least one task")

    normalized: List[Dict[str, Any]] = []
    used_ids = {str(task_id).strip() for task_id in previous_task_ids if str(task_id).strip()}
    next_number = _next_task_number(list(used_ids))
    for candidate in candidate_tasks:
        if not isinstance(candidate, dict):
            raise ValueError("Replacement tasks must be objects")
        task = dict(candidate)
        task_id = str(task.get("task_id") or "").strip()
        if not task_id or task_id in used_ids:
            while f"T{next_number}" in used_ids:
                next_number += 1
            task_id = f"T{next_number}"
            next_number += 1
        task["task_id"] = task_id
        used_ids.add(task_id)
        normalized.append(task)
    return normalized


def _job_task_ids(state: State) -> list[str]:
    """Return all task IDs ever committed or currently active for this job."""
    task_ids = [str(task_id) for task_id in state.get("task_id_registry") or []]
    task_ids.extend(_task_ids(state.get("tasks") or []))

    # Older SQLite checkpoints predate the registry and full-replan audit.
    # Recover their history only from fields whose schema explicitly carries a
    # task_id (or mappings explicitly keyed by task ID), never from generic
    # record IDs such as tool invocation IDs.
    for key in ("current_task", "current_result", "pending_user_action"):
        record = state.get(key)
        if isinstance(record, dict) and str(record.get("task_id") or "").strip():
            task_ids.append(str(record["task_id"]))
    for key in (
        "results",
        "all_results",
        "tool_execution_history",
        "verification_warnings",
    ):
        task_ids.extend(_task_ids(state.get(key) or []))
    for key in ("feedback", "assessment"):
        payload = state.get(key)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("task_id") or "").strip():
            task_ids.append(str(payload["task_id"]))
        task_ids.extend(_task_ids(payload.get("issues") or []))
    for key in (
        "task_revisions",
        "task_retry_count",
        "evidence_recovery_count",
        "task_patch_count",
    ):
        counter = state.get(key)
        if isinstance(counter, dict):
            task_ids.extend(str(task_id) for task_id in counter)

    for event in state.get("plan_patch_history") or []:
        if not isinstance(event, dict):
            continue
        for key in ("old_task_ids", "new_task_ids"):
            task_ids.extend(str(task_id) for task_id in event.get(key) or [])
    return list(dict.fromkeys(task_id for task_id in task_ids if task_id))


def _clear_full_replan_staging() -> Dict[str, Any]:
    return {
        "full_replan_previous_task_ids": [],
        "full_replan_reason": "",
        "full_replan_candidate_tasks": [],
    }


def _error_resume_action(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("action") or value.get("text")
    else:
        raw = value
    normalized = str(raw or "").strip().upper()
    if normalized in {"RETRY_FULL_REPLAN", "RETRY", "REFINE", "重新生成", "重试"}:
        return "RETRY_FULL_REPLAN"
    return "RESUME_OLD_PLAN"


def _full_replan_reason(state: State, parsed: Dict[str, Any] | None = None) -> str:
    if state.get("full_replan_reason"):
        return str(state["full_replan_reason"])
    if parsed and parsed.get("reason"):
        return str(parsed["reason"])
    feedback = state.get("feedback") or {}
    issues = feedback.get("issues") if isinstance(feedback, dict) else []
    if issues and isinstance(issues[0], dict):
        return str(issues[0].get("description") or "")
    return ""


def _full_replan_error_guidance(error: Exception | str) -> Dict[str, Any]:
    return {
        "natural_language_guidance": (
            "无法安全生成替换计划，请提供新的整体目标或重试。"
        ),
        "resource_mapping": {},
        "error": str(error),
    }


def _read_prompt(rel_path: str) -> str:
    """读取统一的 Prompt 文件内容（相对路径，供 ChatPromptTemplate 使用）。"""
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, rel_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "你是 Planner 节点。严格输出 JSON 数组任务列表。"


def _clean_json_fences(s: str) -> str:
    """移除可能的 ```json ... ``` 包裹，返回清洗后的字符串。"""
    s2 = re.sub(r"^```(json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s2 = re.sub(r"\s*```$", "", s2)
    return s2


def _is_confirmation_feedback(user_feedback: str) -> bool:
    """判断用户是否直接确认当前计划。"""
    normalized = str(user_feedback or "").strip().lower()
    return normalized in {
        "",
        "ok",
        "okay",
        "确认",
        "通过",
        "继续",
        "开始",
        "开始执行",
        "按此执行",
        "没问题",
        "可以",
        "同意",
    }


def _normalize_resources(resources):
    result = []
    for r in resources or []:
        if not isinstance(r, dict):
            continue
        name = r.get("name") or (os.path.basename(r.get("path", "")) if r.get("path") else None)
        result.append({
            "name": name,
            "path": r.get("path"),
            "type": r.get("type", "unknown"),
        })
    return result


def _build_resource_index(resources):
    index = {}
    for r in resources or []:
        if not isinstance(r, dict):
            continue
        path = r.get("path") or r.get("file_path")
        name = r.get("name")
        if path:
            if name:
                index[name] = path
            basename = os.path.basename(path)
            index[basename] = path
            index[path] = path
    return index


def _ensure_use_resources_paths(tasks, resources):
    index = _build_resource_index(resources)
    for t in tasks or []:
        raw_list = t.get("use_resources") or []
        normalized = []
        for item in raw_list:
            if isinstance(item, dict):
                path = item.get("path") or item.get("file_path")
                name = item.get("name")
                if path:
                    normalized.append(path)
                    continue
                if name and name in index:
                    normalized.append(index[name])
                    continue
            elif isinstance(item, str):
                if item in index:
                    normalized.append(index[item])
                else:
                    normalized.append(item)
        t["use_resources"] = normalized
        t.setdefault("use_rag", False)
        t.setdefault("task_type", "analysis")
        t.setdefault("query", "")
    return tasks


def _build_tasks_with_llm(intake_obj, config):
    """初始规划：基于 Intake 摘要与统一 Prompt 生成任务列表。"""
    resource_objs = intake_obj.get("resources", []) or []
    resources = _normalize_resources(resource_objs)
    resource_paths = []
    for r in resource_objs:
        if isinstance(r, dict):
            p = r.get("path") or r.get("file_path") or r.get("name")
            if p:
                resource_paths.append(p)
    sections = intake_obj.get("sections") or []
    task_type = intake_obj.get("task_type") or "analysis"
    title = intake_obj.get("title") or "未知项目"
    intent = intake_obj.get("user_intent") or "无"
    doc_length = intake_obj.get("doc_length") or 3000
    constraints = intake_obj.get("constraints") or []
    sys_prompt = _read_prompt("../prompts/planner_to_worker.md")
    try:
        model = get_llm(config)
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "请基于以上输入生成严格的 JSON 任务数组")
        ])
        messages = prompt.format_messages(
            title=title,
            user_intent=intent,
            task_type=task_type,
            constraints=constraints,
            doc_length=doc_length,
            sections=sections,
            resources=resources,
        )
        resp = model.invoke(messages, config=config)
        content_str = _clean_json_fences(str(resp.content).strip())
        tasks = json.loads(content_str)
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("bad tasks")
        tasks = _ensure_use_resources_paths(tasks, resource_objs)
        return tasks
    except Exception as e:
        logger.error(f"Error in _build_tasks_with_llm: {e}")
        fallback = []
        for i, sec in enumerate(sections or ["摘要", "背景与意义"]):
            fallback.append({
                "task_id": f"T{i+1}",
                "task_name": sec,
                "task_description": f"围绕 {sec} 生成占位内容。",
                "generate_figure": False,
                "generate_table": False,
                "use_resources": resource_paths,
            })
        return fallback


def _get_intake_data(state):
    """从 messages 中寻找最新的 `type=INTAKE_SUMMARY` 作为背景数据。"""
    for msg in reversed(state.get("messages", []) or []):
        try:
            c = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            data = json.loads(c) if isinstance(c, str) else c
            if isinstance(data, dict) and data.get("type") == "INTAKE_SUMMARY":
                return data
        except Exception:
            continue
    return {"title": "未知项目", "user_intent": "无", "resources": []}


def _build_tasks_from_replan_feedback(state, config, current_tasks):
    """重做规划：依据 Verifier 反馈与统一 Prompt 生成任务列表。"""
    intake_data = _get_intake_data(state)
    resource_objs = intake_data.get("resources", []) or []
    feedback_obj = state.get("feedback", {}) or {}
    if isinstance(feedback_obj, str):
        try:
            feedback_obj = json.loads(feedback_obj)
        except Exception:
            feedback_obj = {"status": "BLOCKED", "issues": [{"description": feedback_obj}]}
    issues = feedback_obj.get("issues", []) or []
    reason = (issues[0].get("description") if issues else feedback_obj.get("status")) or "需要重新规划"
    suggestion = (issues[0].get("suggestion") if issues else "请检查并重新规划")
    sys_prompt = _read_prompt("../prompts/planner_replan.md")
    try:
        model = get_llm(config)
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "请根据反馈重新生成任务列表")
        ])
        messages = prompt.format_messages(
            title=intake_data.get("title"),
            user_intent=intake_data.get("user_intent"),
            task_type=intake_data.get("task_type", "通用"),
            constraints=intake_data.get("constraints"),
            doc_length=intake_data.get("doc_length"),
            blocked_reason=reason,
            suggestion=suggestion,
            prev_tasks=[t.get("task_name") for t in current_tasks or []],
            resources=_normalize_resources(resource_objs),
        )
        resp = model.invoke(messages, config=config)
        tasks = json.loads(_clean_json_fences(str(resp.content).strip()))
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("replacement plan must be a non-empty task list")
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                raise ValueError("replacement tasks must be objects")
            task.setdefault("task_id", f"T{i+1}")
            task.setdefault("generate_figure", False)
            task.setdefault("generate_table", False)
            task.setdefault("use_resources", [])
        return _ensure_use_resources_paths(tasks, resource_objs)
    except Exception as exc:
        # A full replan must never silently clone the active plan.  Planner
        # converts this explicit failure into FULL_REPLAN_ERROR, which gives
        # the user a retry/cancel path without mutating the old plan.
        raise ValueError(f"replacement plan generation failed: {exc}") from exc


def _resource_identity(resource: Dict[str, Any]) -> str:
    return str(
        resource.get("file_id")
        or resource.get("resource_id")
        or resource.get("path")
        or (str(resource.get("name", "")) + str(resource.get("path", "")))
    )


def _refine_tasks(
    state: State,
    current_tasks,
    user_feedback,
    state_docs,
    intake_data,
    config,
):
    """依据计划确认反馈和本轮新增附件优化任务列表。"""
    current_docs = merge_docs([], state_docs or [])
    initial_resources = merge_docs([], intake_data.get("resources", []) or [])
    initial_resource_ids = {
        _resource_identity(resource)
        for resource in initial_resources
        if isinstance(resource, dict)
    }

    new_resources = []
    for document in current_docs:
        if not isinstance(document, dict):
            continue
        if _resource_identity(document) in initial_resource_ids:
            continue

        normalized = dict(document)
        file_id = normalized.get("file_id") or normalized.get("resource_id")
        if file_id:
            normalized["file_id"] = file_id
            normalized["resource_id"] = file_id
        normalized.setdefault(
            "name",
            normalized.get("original_name")
            or os.path.basename(normalized.get("path", "")),
        )
        normalized.setdefault("type", "unknown")
        new_resources.append(normalized)

    # 纯确认且没有新附件时，保持原计划，避免无意义地再次调用 LLM。
    if _is_confirmation_feedback(user_feedback) and not new_resources:
        return current_tasks

    all_resources = merge_docs(initial_resources, new_resources)
    previous_tasks = json.dumps(current_tasks, ensure_ascii=False)
    system_prompt = _read_prompt("../prompts/planner_intake_replan.md")

    try:
        model = get_llm(config)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "请根据用户反馈和新增资源重新生成任务列表"),
            ]
        )
        messages = prompt.format_messages(
            title=intake_data.get("title"),
            user_intent=intake_data.get("user_intent"),
            task_type=intake_data.get("task_type", "通用"),
            resources=_normalize_resources(initial_resources),
            new_resources=_normalize_resources(new_resources),
            prev_tasks=previous_tasks,
            doc_length=intake_data.get("doc_length"),
            constraints=intake_data.get("constraints"),
            user_feedback=user_feedback,
        )
        response = model.invoke(messages, config=config)
        tasks = json.loads(_clean_json_fences(str(response.content).strip()))
        if not isinstance(tasks, list):
            raise ValueError("Planner refined tasks must be a list")

        for index, task in enumerate(tasks):
            task.setdefault("task_id", f"T{index + 1}")
            task.setdefault("generate_figure", False)
            task.setdefault("generate_table", False)
            task.setdefault("use_resources", [])

        return _ensure_use_resources_paths(tasks, all_resources)
    except Exception as exc:
        logger.exception("Failed to refine tasks: %s", exc)
        if current_tasks:
            return current_tasks

        resource_paths = []
        for resource in all_resources:
            if not isinstance(resource, dict):
                continue
            value = resource.get("path") or resource.get("file_path") or resource.get("name")
            if value:
                resource_paths.append(value)

        return [
            {
                "task_id": "T1",
                "task_name": "重做任务",
                "task_description": f"依据用户反馈重做：{user_feedback}",
                "generate_figure": False,
                "generate_table": False,
                "use_resources": resource_paths,
            }
        ]


def _generate_plan_guidance(tasks: List[Dict[str, Any]], initial_resources: List[str], config: RunnableConfig) -> Dict[str, Any]:
    """生成计划确认引导信息和资源映射关系"""
    try:
        model = get_llm(config)
        sys_prompt = _read_prompt("../prompts/planner_resource_guide.md")
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "请生成引导信息和资源映射")
        ])
        messages = prompt.format_messages(
            tasks=json.dumps(tasks, ensure_ascii=False),
            initial_resources=json.dumps(initial_resources, ensure_ascii=False)
        )
        resp = model.invoke(messages, config=config)
        
        content = str(resp.content)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
        
    return {
        "natural_language_guidance": "为您生成的任务计划如下，请您查阅。如果需要补充资料，请上传；如果您对计划满意，或者您也可以直接运行，我们将立即开始工作。",
        "resource_mapping": {}
    }


def planner(state: State, config: RunnableConfig, **kwargs):
    """Planner 主流程：
    - 确定 Action (INTAKE_SUMMARY / FULL_REPLAN / PROCEED)
    - 生成/更新任务列表
    - 若 PROCEED，生成消息
    - 若其他，推迟消息生成到 planner_confirm
    """
    parsed = _latest_to_planner(state.get("messages", []))
    tasks = state.get("tasks", []) or []
    cursor = state.get("cursor", 0)
    decision = state.get("decision", "NEXT")
    
    # 确定 Action
    if parsed and parsed.get("type") == "INTAKE_SUMMARY":
        planner_action = "INTAKE_SUMMARY"
    elif _is_user_full_replan(parsed) or state.get("planner_action") == "FULL_REPLAN_RETRY":
        planner_action = "FULL_REPLAN"
    else:
        planner_action = "PROCEED"

    # 执行逻辑
    overview = "保持既有任务列表。"
    
    if planner_action == "INTAKE_SUMMARY":
        tasks = _build_tasks_with_llm(parsed, config)
        cursor = 0
        overview = "初始规划已生成。"
    
    elif planner_action == "FULL_REPLAN":
        previous_tasks = list(tasks)
        try:
            tasks = _normalize_replacement_tasks(
                _build_tasks_from_replan_feedback(state, config, previous_tasks),
                _job_task_ids(state),
            )
        except ValueError as exc:
            planner_action = "FULL_REPLAN_ERROR"
            tasks = previous_tasks
            overview = "替换计划无效，等待用户输入。"
        else:
            overview = "已按用户请求生成替换计划，等待确认后执行。"
    
    elif planner_action == "PROCEED":
        # 如果是 PROCEED 且有明确的 PROCEED 指令，移动 cursor
        # 这里的逻辑可能需要根据实际 graph 流转调整，目前假设 PROCEED 时 Verifier 已决定 NEXT
        if parsed and parsed.get("type") == "PROCEED" and decision == "NEXT":
             cursor = min(cursor + 1, max(len(tasks) - 1, 0))
        overview = "继续执行下一任务。"
        
        # 兜底：如果 tasks 为空
        if not tasks:
            tasks = _build_tasks_with_llm({"sections": ["摘要", "背景与意义"], "resources": []}, config)
            cursor = 0
            overview = "使用默认任务列表。"

    # 返回结果
    result = {
        "cursor": cursor,
        "planner_action": planner_action,
        "decision": "NEXT" # 默认重置决策
    }
    if planner_action == "FULL_REPLAN":
        result["tasks"] = list(state.get("tasks") or [])
        result["full_replan_candidate_tasks"] = tasks
        result["full_replan_previous_task_ids"] = _task_ids(state.get("tasks") or [])
        result["full_replan_reason"] = _full_replan_reason(state, parsed)
    else:
        result["tasks"] = tasks
    if planner_action == "INTAKE_SUMMARY":
        result["task_id_registry"] = list(
            dict.fromkeys([*_job_task_ids(state), *_task_ids(tasks)])
        )
    
    # 如果是 PROCEED，生成消息
    if planner_action == "PROCEED":
        content_obj = {
            "from": "Planner",
            "to": "Worker",
            "type": "PLAN_RESULT",
            "content": {
                "tasks": tasks,
                "cursor": cursor
            },
        }
        msg = AIMessage(content=json.dumps(content_obj, ensure_ascii=False))
        result["messages"] = [msg]
    else:
        # 准备 Guidance 数据供 Confirm 节点使用
        intake_data = _get_intake_data(state)
        initial_resources_names = [r.get("name") for r in intake_data.get("resources", []) if isinstance(r, dict)]
        
        guidance_result = (
            _full_replan_error_guidance("empty or invalid replacement plan")
            if planner_action == "FULL_REPLAN_ERROR"
            else _generate_plan_guidance(tasks, initial_resources_names, config)
        )
        result["guidance"] = guidance_result
    
    return result


def planner_confirm(state: State, config: RunnableConfig, **kwargs):
    """等待用户确认计划，并按反馈或新增附件调整任务。"""
    active_tasks = state.get("tasks", []) or []
    planner_action = str(state.get("planner_action") or "")
    is_full_replan = planner_action in {"FULL_REPLAN", "FULL_REPLAN_REFINED"}
    tasks = (
        state.get("full_replan_candidate_tasks") or []
        if is_full_replan
        else active_tasks
    )
    guidance_result = state.get("guidance") or {}

    if not guidance_result:
        intake_data = _get_intake_data(state)
        initial_resource_names = [
            resource.get("name")
            for resource in intake_data.get("resources", [])
            if isinstance(resource, dict) and resource.get("name")
        ]
        guidance_result = _generate_plan_guidance(
            tasks,
            initial_resource_names,
            config,
        )

    if planner_action == "FULL_REPLAN_ERROR":
        resume_value = interrupt(
            {
                "type": "needs_user_input",
                "guidance_text": guidance_result.get("natural_language_guidance"),
                "error": guidance_result.get("error"),
                "accepted_choices": ["RETRY_FULL_REPLAN", "RESUME_OLD_PLAN", "CANCEL"],
            }
        )
        if _error_resume_action(resume_value) == "RETRY_FULL_REPLAN":
            if isinstance(resume_value, dict):
                retry_text = str(resume_value.get("text") or "").strip()
            else:
                retry_text = str(resume_value or "").strip()
            return {
                "planner_action": "FULL_REPLAN_RETRY",
                "decision": "NEXT",
                "feedback": {
                    "status": "BLOCKED",
                    "issues": [{"description": retry_text, "suggestion": retry_text}],
                },
                "full_replan_reason": retry_text,
                "guidance": {},
                "full_replan_candidate_tasks": [],
            }
        return {
            "planner_action": "PROCEED",
            "decision": "NEXT",
            "guidance": {},
            **_clear_full_replan_staging(),
        }

    structured_tasks = [
        {
            "task_name": task.get("task_name"),
            "task_description": task.get("task_description"),
        }
        for task in tasks
    ]
    payload = {
        "type": "confirm_plan_and_resources",
        "guidance_text": guidance_result.get("natural_language_guidance"),
        "structured_msg": {
            "tasks": structured_tasks,
            "resource_mapping": guidance_result.get("resource_mapping") or {},
        },
    }

    resume_value = interrupt(payload)

    if isinstance(resume_value, dict):
        feedback_text = str(resume_value.get("text") or "").strip()
        feedback_message_id = resume_value.get("message_id")
        resumed_docs = resume_value.get("docs") or []
    else:
        feedback_text = str(resume_value or "").strip()
        feedback_message_id = None
        resumed_docs = []

    if not feedback_text:
        feedback_text = "继续"

    feedback_message = HumanMessage(
        id=feedback_message_id,
        content=feedback_text,
        additional_kwargs={
            "message_type": "plan_confirmation",
            "user_id": state.get("user_id"),
            "conversation_id": state.get("conversation_id"),
            "job_id": state.get("job_id"),
            "attachment_ids": [
                document.get("file_id") or document.get("resource_id")
                for document in resumed_docs
                if isinstance(document, dict)
            ],
        },
    )

    # 当前节点内需要完整资源视图；写回 State 时仍只返回 resumed_docs 增量。
    combined_docs = merge_docs(state.get("docs") or [], resumed_docs)
    intake_data = _get_intake_data(state)
    is_confirmation = _is_confirmation_feedback(feedback_text) and not resumed_docs
    if is_full_replan and not is_confirmation:
        try:
            refined_tasks = _normalize_replacement_tasks(
                _refine_tasks(
                    state,
                    tasks,
                    feedback_text,
                    combined_docs,
                    intake_data,
                    config,
                ),
                _job_task_ids(state),
            )
        except ValueError as exc:
            return {
                "planner_action": "FULL_REPLAN_ERROR",
                "guidance": _full_replan_error_guidance(exc),
                "messages": [feedback_message],
                "docs": resumed_docs,
            }
        initial_resource_names = [
            resource.get("name")
            for resource in intake_data.get("resources", [])
            if isinstance(resource, dict) and resource.get("name")
        ]
        return {
            "messages": [feedback_message],
            "cursor": int(state.get("cursor", 0) or 0),
            "planner_action": "FULL_REPLAN_REFINED",
            "full_replan_candidate_tasks": refined_tasks,
            "guidance": _generate_plan_guidance(
                refined_tasks, initial_resource_names, config
            ),
            "docs": resumed_docs,
        }

    if not is_full_replan:
        tasks = _refine_tasks(
            state,
            tasks,
            feedback_text,
            combined_docs,
            intake_data,
            config,
        )
    else:
        try:
            tasks = _normalize_replacement_tasks(
                tasks,
                _job_task_ids(state),
            )
        except ValueError as exc:
            return {
                "planner_action": "FULL_REPLAN_ERROR",
                "guidance": _full_replan_error_guidance(exc),
                "messages": [feedback_message],
                "docs": resumed_docs,
            }

    content_obj = {
        "from": "Planner",
        "to": "Worker",
        "type": "PLAN_RESULT",
        "content": {
            "tasks": tasks,
            "cursor": 0,
        },
    }
    message = AIMessage(content=json.dumps(content_obj, ensure_ascii=False))

    update = {
        "messages": [feedback_message, message],
        "tasks": tasks,
        "cursor": 0,
        # 只提交本轮新增附件，State.merge_docs 会负责合并。
        "docs": resumed_docs,
    }
    if is_full_replan:
        # The replacement plan is staged above, but it becomes a new execution
        # revision only when this explicit confirmation resumes the node.
        task_revisions = {
            str(task["task_id"]): 1
            for task in tasks
            if isinstance(task, dict) and task.get("task_id") is not None
        }
        old_task_ids = list(state.get("full_replan_previous_task_ids") or [])
        new_task_ids = _task_ids(tasks)
        plan_revision = int(state.get("plan_revision", 1) or 1) + 1
        history = list(state.get("plan_patch_history") or [])
        history.append(
            {
                "type": "FULL_REPLAN",
                "previous_plan_revision": plan_revision - 1,
                "new_plan_revision": plan_revision,
                "reason": _full_replan_reason(state),
                "old_task_ids": old_task_ids,
                "new_task_ids": new_task_ids,
            }
        )
        update.update(
            {
                "planner_action": "PROCEED",
                "plan_revision": plan_revision,
                "task_revisions": task_revisions,
                "current_result": {},
                "results": [],
                "all_results": [],
                "current_task": {},
                "worker_state": {},
                "tool_execution_history": [],
                "feedback": {},
                "assessment": {},
                "workflow_action": "",
                "continuation_action": "",
                "verification_warning": {},
                "task_retry_count": {},
                "evidence_recovery_count": {},
                "task_patch_count": {},
                "job_patch_count": 0,
                "replan_count": 0,
                "pending_user_action": {},
                "plan_patch_history": history,
                "verification_warnings": [],
                "guidance": {},
                "final_result": {},
                "decision": "NEXT",
                "task_id_registry": list(
                    dict.fromkeys([*_job_task_ids(state), *new_task_ids])
                ),
                **_clear_full_replan_staging(),
            }
        )
    return update

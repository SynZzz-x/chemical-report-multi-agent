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
from ..limits import MAX_PLAN_TASKS
from ..rag.catalog import load_active_catalog
from ..task_contract import task_allows_web
from ..tool_names import canonical_tool_name
from .intake import web_authorization_directive

logger = logging.getLogger(__name__)


class PlannerGenerationError(ValueError):
    """Planner model output could not satisfy the executable task contract."""


_GENERATED_TASK_REQUIRED_FIELDS = {
    "task_id",
    "task_name",
    "task_description",
    "task_type",
    "use_rag",
    "use_web",
    "query",
    "use_resources",
    "generate_figure",
    "generate_table",
    "visualization",
}
_GENERATED_TASK_BOOLEAN_FIELDS = {
    "use_rag",
    "use_web",
    "generate_figure",
    "generate_table",
}
_VISUALIZATION_REQUIRED_FIELDS = {
    "kind",
    "title",
    "required_concepts",
    "web_queries",
    "allow_web_fallback",
}
_SUPPORTED_VISUALIZATION_KINDS = {"causal"}
_MAX_REQUIRED_CONCEPTS = 6
_CONCEPT_LIST_DELIMITERS = re.compile(r"[/／、,，;；]")
_DATA_ANALYSIS_MARKERS = (
    "pearson",
    "相关系数",
    "回归模型",
    "r²",
    "r2",
    "时间序列",
    "热力图",
    "定量操作窗口",
)
_DATA_RESOURCE_SUFFIXES = (".csv",)


def _is_atomic_concept(value: str) -> bool:
    concept = value.strip()
    if _CONCEPT_LIST_DELIMITERS.search(concept):
        return False
    for connector in ("以及", "与", "和", "及"):
        left, separator, right = concept.partition(connector)
        if separator and len(left.strip()) >= 2 and len(right.strip()) >= 2:
            return False
    return True

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


_REPLACEMENT_TASK_TYPES = {"analysis", "summary", "inference"}
_REPLACEMENT_BOOLEAN_FIELDS = {
    "generate_figure",
    "generate_table",
    "use_rag",
    "use_web",
    "allow_web_fallback",
}
_REPLACEMENT_TASK_FIELDS = {
    "task_id",
    "task_name",
    "task_description",
    "generate_figure",
    "generate_table",
    "use_rag",
    "use_web",
    "allow_web_fallback",
    "task_type",
    "query",
    "use_resources",
    "tool_requirements",
    "visualization",
}


def _validate_string_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")


def _validate_required_concepts(value: Any) -> None:
    field = "visualization.required_concepts"
    _validate_string_list(value, field)
    if not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > _MAX_REQUIRED_CONCEPTS:
        raise ValueError(f"{field} must contain at most 6 concepts")
    non_atomic = [concept for concept in value if not _is_atomic_concept(concept)]
    if non_atomic:
        raise ValueError(
            f"{field} must contain atomic concepts; split={non_atomic}"
        )


def _validate_visualization_contract(visualization: Any) -> None:
    if not isinstance(visualization, dict):
        raise ValueError("visualization must be an object or null")
    if set(visualization) != _VISUALIZATION_REQUIRED_FIELDS:
        missing = sorted(_VISUALIZATION_REQUIRED_FIELDS - set(visualization))
        extra = sorted(set(visualization) - _VISUALIZATION_REQUIRED_FIELDS)
        raise ValueError(
            "visualization fields must exactly match the concept-graph contract; "
            f"missing={missing}, extra={extra}"
        )
    if (
        not isinstance(visualization["kind"], str)
        or visualization["kind"] not in _SUPPORTED_VISUALIZATION_KINDS
    ):
        raise ValueError("visualization.kind must be causal")
    if (
        not isinstance(visualization["title"], str)
        or not visualization["title"].strip()
    ):
        raise ValueError("visualization.title must be a non-empty string")
    _validate_required_concepts(visualization["required_concepts"])
    _validate_string_list(
        visualization["web_queries"],
        "visualization.web_queries",
    )
    if not isinstance(visualization["allow_web_fallback"], bool):
        raise ValueError("visualization.allow_web_fallback must be a boolean")


def _validate_generated_task_schema(candidate_tasks: Any) -> None:
    """Validate the exact model-facing task contract without legacy fields."""
    if not isinstance(candidate_tasks, list) or not candidate_tasks:
        raise ValueError("Planner tasks must be a non-empty list")
    if len(candidate_tasks) > MAX_PLAN_TASKS:
        raise ValueError(f"Planner plan exceeds {MAX_PLAN_TASKS} tasks")

    for task in candidate_tasks:
        if not isinstance(task, dict):
            raise ValueError("Planner tasks must be objects")
        if set(task) != _GENERATED_TASK_REQUIRED_FIELDS:
            missing = sorted(_GENERATED_TASK_REQUIRED_FIELDS - set(task))
            extra = sorted(set(task) - _GENERATED_TASK_REQUIRED_FIELDS)
            raise ValueError(
                "Planner task fields must exactly match the generated contract; "
                f"missing={missing}, extra={extra}"
            )
        for field in ("task_id", "task_name", "task_description"):
            if not isinstance(task[field], str) or not task[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        if (
            not isinstance(task["task_type"], str)
            or task["task_type"] not in _REPLACEMENT_TASK_TYPES
        ):
            raise ValueError("task_type must be analysis, summary, or inference")
        for field in _GENERATED_TASK_BOOLEAN_FIELDS:
            if not isinstance(task[field], bool):
                raise ValueError(f"{field} must be a boolean")
        if not isinstance(task["query"], str):
            raise ValueError("query must be a string")
        _validate_string_list(task["use_resources"], "use_resources")

        visualization = task["visualization"]
        if visualization is None:
            continue
        _validate_visualization_contract(visualization)


def _validate_replacement_task_schema(candidate_tasks: Any) -> None:
    """Reject malformed replacement tasks before assigning stable IDs."""
    if not isinstance(candidate_tasks, list) or not candidate_tasks:
        raise ValueError("Replacement plan must contain at least one task")
    if len(candidate_tasks) > MAX_PLAN_TASKS:
        raise ValueError(f"Replacement plan exceeds {MAX_PLAN_TASKS} tasks")
    for task in candidate_tasks:
        if not isinstance(task, dict):
            raise ValueError("Replacement tasks must be objects")
        unknown = set(task) - _REPLACEMENT_TASK_FIELDS
        if unknown:
            raise ValueError(f"replacement task has unknown field: {sorted(unknown)[0]}")
        for field in ("task_name", "task_description"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        if "task_id" in task and not isinstance(task["task_id"], str):
            raise ValueError("task_id must be a string when provided")
        for field in _REPLACEMENT_BOOLEAN_FIELDS:
            if field in task and not isinstance(task[field], bool):
                raise ValueError(f"{field} must be a boolean")
        if "task_type" in task and task["task_type"] not in _REPLACEMENT_TASK_TYPES:
            raise ValueError("task_type must be analysis, summary, or inference")
        if "query" in task and not isinstance(task["query"], str):
            raise ValueError("query must be a string")
        for field in ("use_resources", "tool_requirements"):
            if field in task:
                _validate_string_list(task[field], field)
        if "visualization" in task and task["visualization"] is not None:
            visualization = task["visualization"]
            legacy_policy_only = (
                isinstance(visualization, dict)
                and set(visualization) == {"allow_web_fallback"}
                and task.get("generate_figure") is False
            )
            if legacy_policy_only:
                if not isinstance(visualization["allow_web_fallback"], bool):
                    raise ValueError(
                        "visualization.allow_web_fallback must be a boolean"
                    )
            else:
                _validate_visualization_contract(visualization)


def _normalize_replacement_tasks(
    candidate_tasks: List[Dict[str, Any]], previous_task_ids: List[str]
) -> List[Dict[str, Any]]:
    """Give a replacement plan stable IDs that cannot collide with prior work."""
    _validate_replacement_task_schema(candidate_tasks)

    normalized: List[Dict[str, Any]] = []
    used_ids = {str(task_id).strip() for task_id in previous_task_ids if str(task_id).strip()}
    next_number = _next_task_number(list(used_ids))
    for candidate in candidate_tasks:
        task = dict(candidate)
        if "tool_requirements" in task:
            canonical_requirements = [
                canonical_tool_name(requirement)
                for requirement in task["tool_requirements"]
            ]
            if any(requirement is None for requirement in canonical_requirements):
                raise ValueError("tool_requirements contains an invalid tool requirement")
            task["tool_requirements"] = canonical_requirements
            if "spider_tool" in canonical_requirements and not task_allows_web(task):
                raise ValueError("spider_tool requires explicit web permission")
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


def _counter_task_ids(counter: Any, tasks: List[Dict[str, Any]]) -> list[str]:
    """Return task IDs from a legacy task-keyed counter without cursor aliases."""
    task_ids: list[str] = []
    for key in (counter or {}):
        if isinstance(key, int) and not isinstance(key, bool):
            if 0 <= key < len(tasks):
                task = tasks[key]
                if isinstance(task, dict) and str(task.get("task_id") or "").strip():
                    task_ids.append(str(task["task_id"]))
            # Integer keys in old checkpoints were cursor positions.  If the
            # matching task is unavailable, ignoring them avoids reserving a
            # bogus literal ID such as "3".
            continue
        task_id = str(key).strip()
        if task_id:
            task_ids.append(task_id)
    return task_ids


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
        "verifier_retry_count",
    ):
        counter = state.get(key)
        if isinstance(counter, dict):
            task_ids.extend(_counter_task_ids(counter, state.get("tasks") or []))

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


def _initial_plan_error_action(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("action") or value.get("text")
    else:
        raw = value
    normalized = str(raw or "").strip().upper()
    if normalized in {
        "RETRY_INITIAL_PLAN",
        "RETRY",
        "重新生成",
        "重新规划",
        "重试",
    }:
        return "RETRY_INITIAL_PLAN"
    return "CANCEL"


def _full_replan_confirmation_action(value: Any, resumed_docs: List[Any]) -> str:
    """Return CONFIRM, REFINE, CANCEL, or RESUME_OLD_PLAN deterministically."""
    if isinstance(value, dict):
        explicit = str(value.get("action") or "").strip().upper()
        text = str(value.get("text") or "").strip()
    else:
        explicit = ""
        text = str(value or "").strip()
    aliases = {
        "CONFIRM": "CONFIRM",
        "REFINE": "REFINE",
        "CANCEL": "CANCEL",
        "RESUME_OLD_PLAN": "RESUME_OLD_PLAN",
    }
    if explicit in aliases:
        return aliases[explicit]

    normalized = text.upper()
    if normalized in {"CANCEL", "取消", "取消整体重规划", "取消重规划"}:
        return "CANCEL"
    if normalized in {
        "RESUME_OLD_PLAN",
        "恢复旧计划",
        "继续旧计划",
        "恢复原计划",
        "继续原计划",
    }:
        return "RESUME_OLD_PLAN"
    if resumed_docs:
        return "REFINE"
    return "CONFIRM" if _is_confirmation_feedback(text) else "REFINE"


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


def _initial_plan_error_guidance(error: Exception | str) -> Dict[str, Any]:
    return {
        "natural_language_guidance": (
            "任务规划生成失败，尚未创建任何可执行任务。请重试或取消本次任务。"
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
    except OSError as exc:
        raise PlannerGenerationError(f"Planner prompt is unavailable: {path}") from exc


def _clean_json_fences(s: str) -> str:
    """移除可能的 ```json ... ``` 包裹，返回清洗后的字符串。"""
    s2 = re.sub(r"^```(json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s2 = re.sub(r"\s*```$", "", s2)
    return s2


def _resource_aliases(resource: Dict[str, Any]) -> set[str]:
    aliases = {
        str(resource.get("name") or "").strip(),
        str(resource.get("path") or resource.get("file_path") or "").strip(),
        str(resource.get("file_id") or "").strip(),
        str(resource.get("resource_id") or "").strip(),
    }
    path = str(resource.get("path") or resource.get("file_path") or "").strip()
    if path:
        aliases.add(os.path.basename(path))
    return {alias for alias in aliases if alias}


def _resolve_assigned_resources(
    task: Dict[str, Any], resources: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    resource_index: Dict[str, List[Dict[str, Any]]] = {}
    for resource in resources or []:
        if not isinstance(resource, dict):
            continue
        for alias in _resource_aliases(resource):
            matches = resource_index.setdefault(alias, [])
            if resource not in matches:
                matches.append(resource)

    resolved: List[Dict[str, Any]] = []
    for raw_value in task.get("use_resources") or []:
        value = str(raw_value).strip()
        matches = resource_index.get(value) or []
        if not matches:
            raise ValueError(
                f"task {task.get('task_id')} references unknown resource: {value}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"task {task.get('task_id')} references ambiguous resource: {value}"
            )
        resolved.append(matches[0])
    return resolved


def _task_has_data_resource(assigned_resources: List[Dict[str, Any]]) -> bool:
    for resource in assigned_resources:
        path = str(resource.get("path") or resource.get("file_path") or "").strip()
        if path.lower().endswith(_DATA_RESOURCE_SUFFIXES):
            return True
    return False


def _requires_data_resource(text: Any) -> bool:
    normalized = str(text or "").casefold()
    if any(marker in normalized for marker in _DATA_ANALYSIS_MARKERS):
        return True
    quantitative_action = r"(?:计算|统计|测算|量化|定量评估)"
    metric = r"(?:转化率|能耗)"
    return bool(
        re.search(rf"{quantitative_action}.{{0,20}}{metric}", normalized)
        or re.search(rf"{metric}.{{0,12}}{quantitative_action}", normalized)
    )


def _validate_generated_task_semantics(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    policy_context: Dict[str, Any],
) -> None:
    web_authorized = policy_context.get("web_authorized") is True

    for task in tasks:
        task_text = " ".join(
            str(task.get(field) or "")
            for field in ("task_name", "task_description", "query")
        )
        normalized = task_text.casefold()
        if task.get("use_rag") is True and not str(task.get("query") or "").strip():
            raise ValueError(
                f"task {task.get('task_id')} sets use_rag=true but has an empty query"
            )
        if task.get("use_rag") is False and task.get("query") != "":
            raise ValueError(
                f"task {task.get('task_id')} sets use_rag=false but query is not empty"
            )

        assigned_resources = _resolve_assigned_resources(task, resources)
        requires_data = _requires_data_resource(normalized)
        if requires_data and not _task_has_data_resource(assigned_resources):
            raise ValueError(
                f"task {task.get('task_id')} requires a real assigned data resource"
            )

        visualization = task.get("visualization")
        if visualization is not None and task.get("generate_figure") is not True:
            raise ValueError(
                f"task {task.get('task_id')} has visualization but "
                "generate_figure is false"
            )
        if (
            visualization is not None
            and task.get("use_rag") is not True
            and not task_allows_web(task)
        ):
            raise ValueError(
                f"task {task.get('task_id')} concept visualization requires an "
                "active RAG or authorized Web evidence channel"
            )
        if (
            task.get("generate_figure") is True
            and visualization is None
            and not _task_has_data_resource(assigned_resources)
        ):
            raise ValueError(
                f"task {task.get('task_id')} ordinary figure requires a real "
                "assigned CSV data resource with a usable path"
            )
        has_web_queries = bool(
            isinstance(visualization, dict) and visualization.get("web_queries")
        )
        if (task_allows_web(task) or has_web_queries) and not web_authorized:
            raise ValueError(
                f"task {task.get('task_id')} requires explicit web authorization"
            )

def _is_abstract_section(section: Any) -> bool:
    value = str(section or "").strip()
    normalized = re.sub(r"[\s:：_-]+", "", value).casefold()
    return normalized.startswith(("摘要", "abstract")) or normalized.endswith(
        ("摘要", "abstract")
    )


def _validate_initial_section_coverage(
    tasks: List[Dict[str, Any]],
    sections: List[Any] | None,
) -> None:
    expected = [
        str(section).strip()
        for section in sections or []
        if str(section or "").strip() and not _is_abstract_section(section)
    ]
    if not expected:
        return
    actual = [str(task.get("task_name") or "").strip() for task in tasks]
    if actual != expected:
        raise ValueError(
            "initial plan tasks must match Intake sections one-to-one and in order: "
            f"expected={expected}, actual={actual}"
        )


def _parse_generated_plan_payload(
    content: str,
    resources: List[Dict[str, Any]],
    policy_context: Dict[str, Any],
    expected_sections: List[Any] | None = None,
) -> List[Dict[str, Any]]:
    payload = json.loads(_clean_json_fences(str(content).strip()))
    if not isinstance(payload, dict) or set(payload) != {"tasks"}:
        raise ValueError("Planner output must be a JSON object containing only tasks")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Planner tasks must be a non-empty list")

    normalized_tasks: List[Dict[str, Any]] = []
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            raise ValueError("Planner tasks must be objects")
        normalized_tasks.append(dict(raw_task))

    _validate_generated_task_schema(normalized_tasks)
    expected_ids = [f"T{index}" for index in range(1, len(normalized_tasks) + 1)]
    actual_ids = [task["task_id"] for task in normalized_tasks]
    if actual_ids != expected_ids:
        raise ValueError(
            "Planner task IDs must be sequential "
            f"{expected_ids}; actual={actual_ids}"
        )
    _validate_generated_task_semantics(
        normalized_tasks,
        resources,
        policy_context,
    )
    _validate_initial_section_coverage(normalized_tasks, expected_sections)
    return normalized_tasks


def _invoke_plan_generation(
    *,
    config: RunnableConfig,
    system_prompt: str,
    human_prompt: str,
    prompt_values: Dict[str, Any],
    resources: List[Dict[str, Any]],
    policy_context: Dict[str, Any],
    failure_label: str,
    expected_sections: List[Any] | None = None,
) -> List[Dict[str, Any]]:
    try:
        model = get_llm(config, json_mode=True)
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
        base_messages = prompt.format_messages(**prompt_values)
    except Exception as exc:
        raise PlannerGenerationError(f"{failure_label}: {exc}") from exc

    last_error: Exception | None = None
    for attempt in range(1, 3):
        messages = list(base_messages)
        if last_error is not None:
            messages.append(
                HumanMessage(
                    content=(
                        "上一次输出未通过 Plan JSON 校验。请仅重新生成符合约束的 "
                        '{"tasks": [...]} JSON 对象。校验错误：'
                        f"{last_error}"
                    )
                )
            )
        response_text = ""
        try:
            response = model.invoke(messages, config=config)
            response_text = str(response.content).strip()
            tasks = _parse_generated_plan_payload(
                response_text,
                resources,
                policy_context,
                expected_sections,
            )
            return _ensure_use_resources_paths(tasks, resources)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Planner generation validation failed: path=%s attempt=%s "
                "error_type=%s error=%s response=%r",
                failure_label,
                attempt,
                type(exc).__name__,
                str(exc),
                response_text[:2000],
            )

    raise PlannerGenerationError(f"{failure_label}: {last_error}") from last_error


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
            "file_id": r.get("file_id"),
            "resource_id": r.get("resource_id"),
        })
    return result


def _normalize_knowledge_catalog(entries):
    """Expose only compact planning metadata, never source paths or chunks."""

    allowed = (
        "resource_id",
        "file_name",
        "file_type",
        "indexed",
        "summary",
        "topics",
        "content_type",
        "has_structured_data",
        "supports",
        "catalog_version",
    )
    result = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        result.append({key: entry.get(key) for key in allowed})
    return result


def _knowledge_catalog_for_planner():
    try:
        return _normalize_knowledge_catalog(load_active_catalog())
    except Exception as exc:
        logger.warning("Planner could not read Knowledge Catalog: %s", exc)
        return []


def _build_resource_index(resources):
    index = {}
    for r in resources or []:
        if not isinstance(r, dict):
            continue
        path = r.get("path") or r.get("file_path")
        if path:
            for alias in _resource_aliases(r):
                index[alias] = path
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
                value = item.strip()
                if value in index:
                    normalized.append(index[value])
                else:
                    normalized.append(value)
        t["use_resources"] = normalized
    return tasks


def _build_tasks_with_llm(intake_obj, config):
    """初始规划：基于 Intake 摘要与统一 Prompt 生成任务列表。"""
    resource_objs = intake_obj.get("resources", []) or []
    resources = _normalize_resources(resource_objs)
    knowledge_catalog = _knowledge_catalog_for_planner()
    sections = intake_obj.get("sections") or []
    task_type = intake_obj.get("task_type") or "analysis"
    title = intake_obj.get("title") or "未知项目"
    intent = intake_obj.get("user_intent") or "无"
    doc_length = intake_obj.get("doc_length") or 3000
    constraints = intake_obj.get("constraints") or []
    sys_prompt = _read_prompt("../prompts/planner_to_worker.md")
    return _invoke_plan_generation(
        config=config,
        system_prompt=sys_prompt,
        human_prompt="请基于以上输入生成严格的 Plan JSON 对象",
        prompt_values={
            "title": title,
            "user_intent": intent,
            "task_type": task_type,
            "constraints": constraints,
            "doc_length": doc_length,
            "sections": sections,
            "resources": resources,
            "knowledge_catalog": knowledge_catalog,
            "core_content": intake_obj.get("core_content") or [],
            "style": intake_obj.get("style"),
            "output_format": intake_obj.get("output_format"),
            "web_authorized": intake_obj.get("web_authorized") is True,
        },
        resources=resource_objs,
        policy_context=intake_obj,
        failure_label="initial plan generation failed",
        expected_sections=sections,
    )


def _get_intake_data(state):
    """从 messages 中寻找最新的 `type=INTAKE_SUMMARY` 作为背景数据。"""
    for msg in reversed(state.get("messages", []) or []):
        if isinstance(msg, dict):
            role = str(msg.get("role") or msg.get("type") or "").casefold()
        else:
            role = str(
                getattr(msg, "type", None) or getattr(msg, "role", None) or ""
            ).casefold()
        if role not in {"ai", "assistant"}:
            continue
        try:
            c = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            data = json.loads(c) if isinstance(c, str) else c
            if (
                isinstance(data, dict)
                and data.get("type") == "INTAKE_SUMMARY"
                and data.get("from") == "Intake"
                and data.get("to") == "Planner"
            ):
                return data
        except Exception:
            continue
    return {}


def _require_intake_data(state: State) -> Dict[str, Any]:
    intake_data = _get_intake_data(state)
    if intake_data.get("type") != "INTAKE_SUMMARY":
        raise PlannerGenerationError(
            "original INTAKE_SUMMARY is unavailable; planning cannot continue"
        )
    return intake_data


def _effective_web_authorization(
    state: State,
    intake_data: Dict[str, Any],
    user_feedback: str | None = None,
) -> bool:
    directive = web_authorization_directive(user_feedback or "")
    if directive is not None:
        return directive
    if isinstance(state.get("web_authorized"), bool):
        return state["web_authorized"]
    return intake_data.get("web_authorized") is True


def _build_tasks_from_replan_feedback(state, config, current_tasks):
    """重做规划：依据 Verifier 反馈与统一 Prompt 生成任务列表。"""
    intake_data = _require_intake_data(state)
    policy_context = dict(intake_data)
    policy_context["web_authorized"] = _effective_web_authorization(
        state,
        intake_data,
    )
    resource_objs = intake_data.get("resources", []) or []
    knowledge_catalog = _knowledge_catalog_for_planner()
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
        return _invoke_plan_generation(
            config=config,
            system_prompt=sys_prompt,
            human_prompt="请根据反馈重新生成严格的 Plan JSON 对象",
            prompt_values={
                "title": intake_data.get("title"),
                "user_intent": intake_data.get("user_intent"),
                "task_type": intake_data.get("task_type", "通用"),
                "constraints": intake_data.get("constraints") or [],
                "doc_length": intake_data.get("doc_length"),
                "blocked_reason": reason,
                "suggestion": suggestion,
                "prev_tasks": [t.get("task_name") for t in current_tasks or []],
                "resources": _normalize_resources(resource_objs),
                "knowledge_catalog": knowledge_catalog,
                "core_content": intake_data.get("core_content") or [],
                "style": intake_data.get("style"),
                "output_format": intake_data.get("output_format"),
                "web_authorized": policy_context["web_authorized"],
            },
            resources=resource_objs,
            policy_context=policy_context,
            failure_label="replacement plan generation failed",
        )
    except Exception as exc:
        # A full replan must never silently clone the active plan.  Planner
        # converts this explicit failure into FULL_REPLAN_ERROR, which gives
        # the user a retry/cancel path without mutating the old plan.
        if isinstance(exc, PlannerGenerationError):
            raise
        raise PlannerGenerationError(
            f"replacement plan generation failed: {exc}"
        ) from exc


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
    if intake_data.get("type") != "INTAKE_SUMMARY":
        raise PlannerGenerationError(
            "original INTAKE_SUMMARY is unavailable; refinement cannot continue"
        )
    policy_context = dict(intake_data)
    policy_context["web_authorized"] = _effective_web_authorization(
        state,
        intake_data,
        str(user_feedback or ""),
    )
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
    knowledge_catalog = _knowledge_catalog_for_planner()
    previous_tasks = json.dumps(current_tasks, ensure_ascii=False)
    system_prompt = _read_prompt("../prompts/planner_intake_replan.md")

    try:
        return _invoke_plan_generation(
            config=config,
            system_prompt=system_prompt,
            human_prompt="请根据用户反馈和新增资源生成严格的 Plan JSON 对象",
            prompt_values={
                "title": intake_data.get("title"),
                "user_intent": intake_data.get("user_intent"),
                "task_type": intake_data.get("task_type", "通用"),
                "resources": _normalize_resources(initial_resources),
                "new_resources": _normalize_resources(new_resources),
                "knowledge_catalog": knowledge_catalog,
                "prev_tasks": previous_tasks,
                "doc_length": intake_data.get("doc_length"),
                "constraints": intake_data.get("constraints") or [],
                "user_feedback": user_feedback,
                "core_content": intake_data.get("core_content") or [],
                "style": intake_data.get("style"),
                "output_format": intake_data.get("output_format"),
                "web_authorized": policy_context["web_authorized"],
            },
            resources=all_resources,
            policy_context=policy_context,
            failure_label="replacement plan refinement failed",
        )
    except Exception as exc:
        logger.exception("Failed to refine tasks: %s", exc)
        if isinstance(exc, PlannerGenerationError):
            raise
        raise PlannerGenerationError(
            f"replacement plan refinement failed: {exc}"
        ) from exc


def _generate_plan_guidance(tasks: List[Dict[str, Any]], initial_resources: List[str], config: RunnableConfig) -> Dict[str, Any]:
    """生成计划确认引导信息和资源映射关系"""
    try:
        model = get_llm(config, json_mode=True)
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
    retrying_initial_plan = state.get("planner_action") == "INITIAL_PLAN_RETRY"
    initial_retry_error: Exception | None = None
    if retrying_initial_plan:
        try:
            parsed = _require_intake_data(state)
        except PlannerGenerationError as exc:
            parsed = None
            initial_retry_error = exc
    
    # 确定 Action
    if initial_retry_error is not None:
        planner_action = "INITIAL_PLAN_ERROR"
    elif retrying_initial_plan or (
        parsed and parsed.get("type") == "INTAKE_SUMMARY"
    ):
        planner_action = "INTAKE_SUMMARY"
    elif _is_user_full_replan(parsed) or state.get("planner_action") == "FULL_REPLAN_RETRY":
        planner_action = "FULL_REPLAN"
    else:
        planner_action = "PROCEED"

    # 执行逻辑
    overview = "保持既有任务列表。"
    full_replan_error: Exception | None = None
    
    if planner_action == "INITIAL_PLAN_ERROR":
        tasks = []
        cursor = 0
        overview = "缺少原始需求上下文，等待用户取消。"
        initial_plan_error = initial_retry_error

    elif planner_action == "INTAKE_SUMMARY":
        try:
            tasks = _build_tasks_with_llm(parsed, config)
        except ValueError as exc:
            planner_action = "INITIAL_PLAN_ERROR"
            tasks = []
            cursor = 0
            overview = "初始规划无效，等待用户重试或取消。"
            initial_plan_error = exc
        else:
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
            full_replan_error = exc
        else:
            overview = "已按用户请求生成替换计划，等待确认后执行。"
    
    elif planner_action == "PROCEED":
        # 如果是 PROCEED 且有明确的 PROCEED 指令，移动 cursor
        # 这里的逻辑可能需要根据实际 graph 流转调整，目前假设 PROCEED 时 Verifier 已决定 NEXT
        if parsed and parsed.get("type") == "PROCEED" and decision == "NEXT":
             cursor = min(cursor + 1, max(len(tasks) - 1, 0))
        overview = "继续执行下一任务。"
        
        # 无有效计划时必须停止，不能生成默认占位计划并进入 Worker。
        if not tasks:
            planner_action = "INITIAL_PLAN_ERROR"
            initial_plan_error = PlannerGenerationError(
                "cannot proceed without a validated initial plan"
            )
            cursor = 0
            overview = "缺少有效初始计划，等待用户重试或取消。"

    # 返回结果
    result = {
        "cursor": cursor,
        "planner_action": planner_action,
        "decision": "NEXT" # 默认重置决策
    }
    if planner_action in {"INTAKE_SUMMARY", "INITIAL_PLAN_ERROR"} and isinstance(
        parsed, dict
    ):
        result["web_authorized"] = parsed.get("web_authorized") is True
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
        
        if planner_action == "FULL_REPLAN_ERROR":
            guidance_result = _full_replan_error_guidance(
                full_replan_error or "empty or invalid replacement plan"
            )
        elif planner_action == "INITIAL_PLAN_ERROR":
            guidance_result = _initial_plan_error_guidance(initial_plan_error)
        else:
            guidance_result = _generate_plan_guidance(
                tasks, initial_resources_names, config
            )
        result["guidance"] = guidance_result
    
    return result


def planner_confirm(state: State, config: RunnableConfig, **kwargs):
    """等待用户确认计划，并按反馈或新增附件调整任务。"""
    active_tasks = state.get("tasks", []) or []
    planner_action = str(state.get("planner_action") or "")
    is_full_replan = planner_action in {"FULL_REPLAN", "FULL_REPLAN_REFINED"}
    is_initial_plan = planner_action in {
        "INTAKE_SUMMARY",
        "INTAKE_SUMMARY_REFINED",
    }
    tasks = (
        state.get("full_replan_candidate_tasks") or []
        if is_full_replan
        else active_tasks
    )
    guidance_result = state.get("guidance") or {}

    if planner_action == "INITIAL_PLAN_ERROR":
        resume_value = interrupt(
            {
                "type": "needs_user_input",
                "guidance_text": guidance_result.get("natural_language_guidance")
                or "任务规划生成失败，请重试或取消。",
                "error": guidance_result.get("error"),
                "accepted_choices": ["RETRY_INITIAL_PLAN", "CANCEL"],
            }
        )
        if _initial_plan_error_action(resume_value) == "RETRY_INITIAL_PLAN":
            return {
                "planner_action": "INITIAL_PLAN_RETRY",
                "decision": "NEXT",
                "tasks": [],
                "guidance": {},
            }
        return {
            "planner_action": "INITIAL_PLAN_CANCELLED",
            "decision": "END",
            "tasks": [],
            "guidance": {},
            "final_result": {
                "success": False,
                "status": "cancelled",
                "error": guidance_result.get("error")
                or "Initial plan generation was cancelled.",
            },
        }

    if is_full_replan:
        try:
            tasks = _normalize_replacement_tasks(tasks, _job_task_ids(state))
        except ValueError as exc:
            return {
                "planner_action": "FULL_REPLAN_ERROR",
                "guidance": _full_replan_error_guidance(exc),
            }

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
            "task_id": task.get("task_id"),
            "task_name": task.get("task_name"),
            "task_description": task.get("task_description"),
            "tool_requirements": task.get("tool_requirements") or [],
            "use_web": task.get("use_web") is True,
            "allow_web_fallback": task.get("allow_web_fallback") is True,
            "visualization": task.get("visualization"),
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

    full_replan_confirmation_action = (
        _full_replan_confirmation_action(resume_value, resumed_docs)
        if is_full_replan
        else ""
    )

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
    effective_web_authorized = _effective_web_authorization(
        state,
        intake_data,
        feedback_text,
    )
    if is_full_replan and full_replan_confirmation_action in {
        "CANCEL",
        "RESUME_OLD_PLAN",
    }:
        return {
            "messages": [feedback_message],
            "docs": resumed_docs,
            "planner_action": "PROCEED",
            "decision": "NEXT",
            "guidance": {},
            **_clear_full_replan_staging(),
        }
    is_confirmation = (
        full_replan_confirmation_action == "CONFIRM"
        if is_full_replan
        else _is_confirmation_feedback(feedback_text) and not resumed_docs
    )
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
                "web_authorized": effective_web_authorized,
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
            "web_authorized": effective_web_authorized,
        }

    if is_initial_plan and not is_confirmation:
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
                [],
            )
        except ValueError as exc:
            return {
                "planner_action": "INTAKE_SUMMARY_REFINED",
                "guidance": _initial_plan_error_guidance(exc),
                "messages": [feedback_message],
                "docs": resumed_docs,
                "web_authorized": effective_web_authorized,
            }
        initial_resource_names = [
            resource.get("name")
            for resource in intake_data.get("resources", [])
            if isinstance(resource, dict) and resource.get("name")
        ]
        return {
            "messages": [feedback_message],
            "tasks": refined_tasks,
            "cursor": 0,
            "planner_action": "INTAKE_SUMMARY_REFINED",
            "guidance": _generate_plan_guidance(
                refined_tasks, initial_resource_names, config
            ),
            "docs": resumed_docs,
            "web_authorized": effective_web_authorized,
            "task_id_registry": list(
                dict.fromkeys([*_job_task_ids(state), *_task_ids(refined_tasks)])
            ),
        }

    if is_full_replan:
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
    if is_initial_plan:
        update.update(
            {
                "planner_action": "PROCEED",
                "task_id_registry": list(
                    dict.fromkeys([*_job_task_ids(state), *_task_ids(tasks)])
                ),
            }
        )
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
                "verifier_retry_count": {},
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

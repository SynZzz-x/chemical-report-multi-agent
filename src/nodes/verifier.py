import json
import os
import datetime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.state import State
from src.config import get_app_config
from src.llm import get_llm


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "verifier_logs.jsonl")

MAX_AUTO_RETRIES_PER_TASK = 2
MAX_AUTO_REPLANS_PER_JOB = 1
PLAN_DEFECT_CODES = {
    "BAD_PLAN",
    "CONTRADICTORY_REQUIREMENTS",
    "INVALID_PLAN",
    "MISSING_RESOURCE",
    "RESOURCE_UNAVAILABLE",
    "UNEXECUTABLE_TASK",
}


def _task_name(tasks, idx):
    if 0 <= idx < len(tasks):
        t = tasks[idx]
        return t.get("task_name") or t.get("task_id") or f"Task_{idx}"
    return f"Task_{idx}"


def _append_result_once(previous_results, current_result):
    results = list(previous_results or [])
    if not current_result:
        return results
    task_id = current_result.get("task_id")
    if task_id is not None and any(item.get("task_id") == task_id for item in results):
        return results
    if task_id is None and current_result in results:
        return results
    results.append(current_result)
    return results


def _is_explicit_plan_defect(assessment: dict) -> bool:
    issue_codes = {
        str(issue.get("code") or "").strip().upper()
        for issue in assessment.get("issues", [])
        if isinstance(issue, dict)
    }
    return bool(issue_codes & PLAN_DEFECT_CODES)


def _advance_control(tasks, cursor):
    if (cursor + 1) >= len(tasks):
        return "DONE", {
            "from": "Verifier",
            "to": "Summarizer",
            "type": "SUMMARIZE",
        }
    return "NEXT", {
        "from": "Verifier",
        "to": "Planner",
        "type": "PROCEED",
        "current_section": _task_name(tasks, cursor),
    }


def verifier(state: State, config: RunnableConfig, **kwargs):
    """对当前 state 进行评估并返回结构化 assessment 与 decision。

    简化版：直接使用 get_llm + prompt 进行评估，移除复杂的 _llm_assess 流程。
    """
    current_result = state.get("current_result", {}) or {}
    tasks = state.get("tasks", []) or []
    cursor = state.get("cursor", 0)
    
    # 暂存旧的 results，稍后根据决策决定是否追加 current_result
    previous_results = state.get("results", []) or []
    results = previous_results
    
    # 获取当前任务的重试次数
    task_retry_count = state.get("task_retry_count", {}) or {}
    current_retry_count = task_retry_count.get(cursor, 0)

    decision_code, content_obj = _advance_control(tasks, cursor)
    replan_count = int(state.get("replan_count", 0) or 0)

    # 检查是否启用 LLM
    use_llm = False
    try:
        conf = config.get("configurable", {}) if config else {}
        use_llm = bool(conf.get("use_llm")) or bool(
            get_app_config().deepseek_api_key
        )
    except Exception:
        use_llm = bool(get_app_config().deepseek_api_key)

    assessment = None
    llm_record = {}

    if use_llm:
        try:
            # 1. 准备上下文
            current_task = tasks[cursor] if 0 <= cursor < len(tasks) else {}
            content = (current_result.get("content") or current_result.get("text_output") or "")
            task_name = _task_name(tasks, cursor)
            task_requirements = json.dumps(current_task, ensure_ascii=False, indent=2)
            worker_assets = json.dumps(
                {
                    "status": current_result.get("status"),
                    "tables": current_result.get("tables", []),
                    "figures": current_result.get("figures", []),
                    "citations": current_result.get("citations", []),
                    "sources_used": current_result.get("sources_used", []),
                    "evidence_coverage": current_result.get("evidence_coverage", {}),
                },
                ensure_ascii=False,
            )
            
            # 2. 加载 Prompt
            prompts_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "verifier.md")
            with open(prompts_path, "r", encoding="utf-8") as f:
                template = f.read()
            
            format_instructions = (
                "请仅输出严格的 JSON 对象，格式示例："
                " {\n  \"status\": \"PASS|FAILED|BLOCKED\",\n  \"current_section\": \"任务名称\",\n  \"issues\": [{\"code\": \"ERR_CODE\", \"description\": \"...\", \"suggestion\": \"...\"}],\n  \"recommended_decision\": \"NEXT|RETRY_WORKER|REPLAN\"\n}"
            )
            
            # 3. 构建 Chain
            # 直接使用 system prompt 模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", template),
            ])
            
            # 使用 get_llm 获取模型实例（启用 json_mode）
            model = get_llm(config, json_mode=True)
            chain = prompt | model
            
            # 4. 执行调用
            res = chain.invoke({
                "task_name": task_name,
                "task_requirements": task_requirements,
                "worker_result": content,
                "worker_assets": worker_assets,
                "format_instructions": format_instructions
            })
            
            # 5. 解析结果
            # get_llm 返回 ChatOpenAI 实例，invoke 返回 AIMessage
            raw_content = res.content
            
            # 清洗逻辑：去除 Markdown 代码块包裹
            cleaned_content = raw_content.strip()
            if cleaned_content.startswith("```"):
                import re
                # 去除开头的 ```json (不区分大小写) 或 ```，以及可能的换行
                cleaned_content = re.sub(r'^```[a-zA-Z]*\s*', '', cleaned_content)
                # 去除结尾的 ``` 以及可能的空白
                cleaned_content = re.sub(r'\s*```$', '', cleaned_content)

            # json_mode=True 应该返回有效的 JSON 字符串
            assessment = json.loads(cleaned_content)
            
            llm_record["response_snippet"] = str(assessment)[:500]
            
        except Exception as e:
            # 简化异常处理：任何错误都视为 FAILED
            assessment = {
                "status": "FAILED",
                "current_section": _task_name(tasks, cursor),
                "issues": [{"code": "LLM_ERROR", "description": f"LLM error: {str(e)}", "suggestion": "Retry worker"}],
                "recommended_decision": "RETRY_WORKER"
            }
            llm_record["error"] = str(e)
    else:
        # 如果没有启用 LLM，返回明确的标识
        assessment = {
            "status": "FAILED",
            "current_section": _task_name(tasks, cursor),
            "issues": [{"code": "LLM_NOT_ENABLED", "description": "LLM is not enabled for verification", "suggestion": "Enable LLM or skip verification"}],
            "recommended_decision": "RETRY_WORKER"
        }

    # 后处理：清理无效 issues
    assessment = _sanitize_assessment(assessment, state)

    # 自动审核只允许明确的计划结构问题触发一次 REPLAN。普通内容问题
    # 有限返工，达到上限后保留审核警告并继续，避免无限回到 T1。
    status = assessment.get("status", "PASS")
    if status in ["FAILED", "BLOCKED"]:
        if (
            _is_explicit_plan_defect(assessment)
            and replan_count < MAX_AUTO_REPLANS_PER_JOB
        ):
            decision_code = "REPLAN"
            replan_count += 1
            task_retry_count = {}
            issues_desc = "; ".join(
                f"{item.get('code', 'ISSUE')}: {item.get('description', '')}"
                for item in assessment.get("issues", [])
            )
            content_obj = {
                "from": "Verifier",
                "to": "Planner",
                "type": "REPLAN",
                "reason": f"任务存在明确规划阻断：{issues_desc}",
                "current_section": _task_name(tasks, cursor),
            }
            assessment["recommended_decision"] = "REPLAN"
        else:
            new_retry_count = current_retry_count + 1
            task_retry_count = dict(task_retry_count)
            if new_retry_count <= MAX_AUTO_RETRIES_PER_TASK:
                task_retry_count[cursor] = new_retry_count
                decision_code = "RETRY_WORKER"
                content_obj = {
                    "from": "Verifier",
                    "to": "Worker",
                    "type": "REWORK",
                    "reason": (
                        f"任务质量不达标，自动返工 {new_retry_count}/"
                        f"{MAX_AUTO_RETRIES_PER_TASK}"
                    ),
                }
                assessment["recommended_decision"] = "RETRY_WORKER"
            else:
                task_retry_count.pop(cursor, None)
                decision_code, content_obj = _advance_control(tasks, cursor)
                assessment["issues"] = assessment.get("issues", []) + [{
                    "code": "AUTO_RETRY_LIMIT_REACHED",
                    "description": (
                        f"自动返工已达到 {MAX_AUTO_RETRIES_PER_TASK} 次上限，"
                        "为避免工作流循环，保留当前结果并继续。"
                    ),
                    "suggestion": "请在最终报告中人工复核本章节。",
                }]
                assessment["recommended_decision"] = decision_code
    else:
        task_retry_count = dict(task_retry_count)
        task_retry_count.pop(cursor, None)
    
    # 只有当决策是 NEXT 或 DONE 时，才将当前结果追加到 results 中
    # 如果是 RETRY_WORKER 或 REPLAN，说明当前结果不合格，应被丢弃
    if decision_code in ["NEXT", "DONE"]:
        results = _append_result_once(previous_results, current_result)
    else:
        results = previous_results

    # 记录日志
    print(
        "🔍 AutoVerifier: "
        f"task={_task_name(tasks, cursor)} status={assessment.get('status')} "
        f"decision={decision_code} retries={task_retry_count.get(cursor, 0)} "
        f"replans={replan_count}/{MAX_AUTO_REPLANS_PER_JOB}"
    )
    _log_verifier_output(
        state,
        assessment,
        llm_record,
        decision_code=decision_code,
        task_retry_count=task_retry_count,
        replan_count=replan_count,
    )

    msg = AIMessage(content=json.dumps(content_obj, ensure_ascii=False))
    return {
        "messages": [msg],
        "results": results,
        "decision": decision_code,
        "assessment": assessment,
        "feedback": assessment,
        "task_retry_count": task_retry_count,
        "replan_count": replan_count,
    }


def decision(state: State, config: RunnableConfig, **kwargs):
    return state.get("decision", "DONE")


def _sanitize_assessment(assessment: dict, state: State) -> dict:
    """后处理：移除显然与任务不符的 issue（比如未要求表格却返回 missing_table）。"""
    if not assessment:
        return assessment
    
    # 兼容新的 issues 结构（列表包含字典）
    raw_issues = assessment.get("issues", [])
    sanitized_issues = []
    
    tasks = state.get("tasks", []) or []
    requires_table = False
    requires_image = False
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks):
        task = tasks[cursor]
        desc = task.get("task_description") or ""
        requires_table = bool(task.get("generate_table")) or any(
            value in desc for value in ("表格", "数据表", "生成表")
        )
        requires_image = bool(task.get("generate_figure")) or any(
            value in desc for value in ("趋势图", "因果图", "流程图", "生成图")
        )

    for issue in raw_issues:
        # 如果是旧格式的字符串，尝试转换为新格式
        if isinstance(issue, str):
            code = issue.upper()
            description = issue
            suggestion = "请修正此问题"
            issue_obj = {"code": code, "description": description, "suggestion": suggestion}
        elif isinstance(issue, dict):
            issue_obj = issue
        else:
            continue
            
        code = issue_obj.get("code", "").upper()
        
        if code == "MISSING_TABLE" and not requires_table:
            continue
        if code == "MISSING_IMAGE" and not requires_image:
            continue
            
        sanitized_issues.append(issue_obj)

    # If issues removed and no other issues, bump status
    status = assessment.get("status")
    if not sanitized_issues and status in ["FAILED", "BLOCKED"]:
        status = "PASS"

    return {
        "status": status,
        "current_section": assessment.get("current_section"),
        "issues": sanitized_issues,
        "recommended_decision": assessment.get("recommended_decision")
    }


def _log_verifier_output(
    state: State,
    assessment: dict,
    llm_record: dict = None,
    *,
    decision_code: str = "",
    task_retry_count: dict | None = None,
    replan_count: int = 0,
):
    def _safe(obj):
        try:
            json.dumps(obj, ensure_ascii=False)
            return obj
        except Exception:
            try:
                return str(obj)
            except Exception:
                return None

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tasks": _safe(state.get("tasks")),
        "cursor": _safe(state.get("cursor")),
        "assessment": _safe(assessment),
        "decision": decision_code,
        "task_retry_count": _safe(task_retry_count or {}),
        "replan_count": replan_count,
    }
    if llm_record:
        entry["llm"] = llm_record
        
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

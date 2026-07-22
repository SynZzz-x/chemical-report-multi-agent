import json
import os
import datetime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.state import State
from src.llm import get_llm


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "verifier_logs.jsonl")


def _task_name(tasks, idx):
    if 0 <= idx < len(tasks):
        t = tasks[idx]
        return t.get("task_name") or t.get("task_id") or f"Task_{idx}"
    return f"Task_{idx}"


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

    # 是否为最后一个任务
    done = (cursor + 1) >= len(tasks)
    if done:
        decision_code = "DONE"
        content_obj = {
            "from": "Verifier",
            "to": "Summarizer",
            "type": "SUMMARIZE",
        }
    else:
        decision_code = "NEXT"
        content_obj = {
            "from": "Verifier",
            "to": "Planner",
            "type": "PROCEED",
            "current_section": _task_name(tasks, cursor),
        }

    # 检查是否启用 LLM
    use_llm = False
    try:
        conf = config.get("configurable", {}) if config else {}
        use_llm = bool(conf.get("use_llm")) or bool(os.environ.get("OPENAI_API_KEY"))
    except Exception:
        use_llm = bool(os.environ.get("OPENAI_API_KEY"))

    assessment = None
    llm_record = {}

    if use_llm:
        try:
            # 1. 准备上下文
            content = (current_result.get("content") or current_result.get("text_output") or "")
            task_name = _task_name(tasks, cursor)
            
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
                "worker_result": content,
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

    # 检查任务质量：如果 status 不是 PASS，增加重试计数并决定下一步
    status = assessment.get("status", "PASS")
    if status in ["FAILED", "BLOCKED"]:
        # 任务失败，增加重试计数
        new_retry_count = current_retry_count + 1
        task_retry_count = dict(task_retry_count)  # 创建副本
        task_retry_count[cursor] = new_retry_count
        
        # 如果连续失败2次，强制 REPLAN
        if new_retry_count >= 2:
            decision_code = "REPLAN"
            task_retry_count[cursor] = 0  # 重置计数，重新规划后从头开始
            content_obj = {
                "from": "Verifier",
                "to": "Planner",
                "type": "REPLAN",
                "reason": f"任务 {_task_name(tasks, cursor)} 连续失败 {new_retry_count} 次，需要重新规划",
                "current_section": _task_name(tasks, cursor),
            }
            # Add a generic issue for consecutive failures if not present
            if not any(i.get("code") == "CONSECUTIVE_FAILURES" for i in assessment.get("issues", [])):
                assessment["issues"] = assessment.get("issues", []) + [{
                    "code": "CONSECUTIVE_FAILURES", 
                    "description": f"Failed {new_retry_count} times", 
                    "suggestion": "Replan"
                }]
            assessment["recommended_decision"] = "REPLAN"
        else:
            # 第一次失败，使用 LLM 评估的建议决策
            recommended = assessment.get("recommended_decision", "RETRY_WORKER")
            decision_code = recommended
            
            if recommended in ["REPLAN", "BAD_PLAN"]:
                decision_code = "REPLAN"  # Normalize BAD_PLAN to REPLAN for graph routing
                task_retry_count[cursor] = 0  # 重置计数，重新规划后从头开始
                
                # Format issues for the reason message
                issues_desc = "; ".join([f"{i.get('code', 'ISSUE')}: {i.get('description', '')}" for i in assessment.get("issues", [])])
                reason_msg = f"任务质量不达标，建议重新规划（第 {new_retry_count} 次失败）。问题：{issues_desc}"

                content_obj = {
                    "from": "Verifier",
                    "to": "Planner",
                    "type": "REPLAN",
                    "reason": reason_msg,
                    "current_section": _task_name(tasks, cursor),
                }
            else:  # RETRY_WORKER
                content_obj = {
                    "from": "Verifier",
                    "to": "Worker",
                    "type": "REWORK",
                    "reason": f"任务质量不达标，第 {new_retry_count} 次重试",
                }
    # 任务成功时，不需要重置计数（cursor 会移动到下一个任务）
    
    # 只有当决策是 NEXT 或 DONE 时，才将当前结果追加到 results 中
    # 如果是 RETRY_WORKER 或 REPLAN，说明当前结果不合格，应被丢弃
    if decision_code in ["NEXT", "DONE"]:
        results = previous_results + ([current_result] if current_result else [])
    else:
        results = previous_results

    # 记录日志
    _log_verifier_output(state, assessment, llm_record)

    msg = AIMessage(content=json.dumps(content_obj, ensure_ascii=False))
    return {
        "messages": [msg],
        "results": results,
        "decision": decision_code,
        "assessment": assessment,
        "task_retry_count": task_retry_count,
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
    if tasks:
        desc = (tasks[0].get("task_description") or "")
        requires_table = ("表" in desc or "表格" in desc or "数据表" in desc)
        requires_image = ("图" in desc or "趋势图" in desc or "画图" in desc)

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


def _log_verifier_output(state: State, assessment: dict, llm_record: dict = None):
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
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "tasks": _safe(state.get("tasks")),
        "cursor": _safe(state.get("cursor")),
        "assessment": _safe(assessment),
    }
    if llm_record:
        entry["llm"] = llm_record
        
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

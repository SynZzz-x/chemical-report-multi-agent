import json
import os
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from src.state import State
from src.llm import get_llm
from src.nodes.intake import web_authorization_directive
from src.recovery.policy import commit_current_result
from src.report_acceptance import (
    VERIFIED_PASS,
    derive_report_status,
    record_section_status,
)


_DIRECT_APPROVAL_FEEDBACK = {
    "",
    "ok",
    "okay",
    "确认",
    "通过",
    "继续",
    "继续工作",
    "继续执行",
    "继续下一项",
    "下一步",
    "开始",
    "开始执行",
    "按此执行",
    "没问题",
    "可以",
    "同意",
}

_DECISION_ALIASES = {
    "PASS": "PASS",
    "APPROVE": "PASS",
    "APPROVED": "PASS",
    "CONTINUE": "PASS",
    "NEXT": "PASS",
    "通过": "PASS",
    "继续": "PASS",
    "REWORK": "REWORK",
    "RETRY": "REWORK",
    "RETRY_WORKER": "REWORK",
    "修改": "REWORK",
    "返工": "REWORK",
    "FULL_REPLAN": "FULL_REPLAN",
    "REPLAN": "FULL_REPLAN",
    "BAD_PLAN": "FULL_REPLAN",
    "重新规划": "FULL_REPLAN",
}


def _normalized_feedback(value: str) -> str:
    return str(value or "").strip().rstrip("。.!！?？").strip().lower()


def _is_direct_approval(value: str) -> bool:
    return _normalized_feedback(value) in _DIRECT_APPROVAL_FEEDBACK


def _normalize_decision(value: object) -> str:
    normalized = str(value or "PASS").strip().upper()
    return _DECISION_ALIASES.get(normalized, "PASS")


def _task_name(tasks, idx):
    if 0 <= idx < len(tasks):
        t = tasks[idx]
        return t.get("task_name") or t.get("task_id") or f"Task_{idx}"
    return f"Task_{idx}"


import re

def _read_prompt(rel_path: str) -> str:
    """读取统一的 Prompt 文件内容（相对路径，供 ChatPromptTemplate 使用）。"""
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, rel_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _clean_json_fences(s: str) -> str:
    """移除可能的 ```json ... ``` 包裹，返回清洗后的字符串。"""
    s2 = re.sub(r"^```(json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s2 = re.sub(r"\s*```$", "", s2)
    return s2

def _analyze_feedback(user_feedback: str, task_name: str, current_result_content: str, config: RunnableConfig):
    """使用 LLM 分析用户反馈，决定后续动作"""
    try:
        model = get_llm(config, json_mode=True)
        template = _read_prompt("../prompts/verifier_manual.md")
        if not template:
            # Fallback if file read fails (though it shouldn't)
            template = """你是一个任务审核助手。请分析用户反馈：{user_feedback}"""
            
        prompt = ChatPromptTemplate.from_messages([("system", template)])
        chain = prompt | model
        
        # 截取一部分结果以防 prompt 过长
        snippet = current_result_content[:1000] + "..." if len(current_result_content) > 1000 else current_result_content
        
        res = chain.invoke({
            "task_name": task_name,
            "current_result_snippet": snippet,
            "user_feedback": user_feedback
        })
        
        content_str = _clean_json_fences(str(res.content).strip())
        return json.loads(content_str)
    except Exception as e:
        # 兜底逻辑：默认视为通过
        return {
            "decision": "PASS",
            "reason": f"反馈分析失败，默认通过: {str(e)}",
            "suggestions": ""
        }


def verifier_manual(state: State, config: RunnableConfig, **kwargs):
    """
    人工审核节点：
    1. 展示当前结果摘要
    2. Interrupt 等待用户反馈
    3. Resume 后 LLM 分析反馈
    4. 路由：PASS -> NEXT/DONE; REWORK -> RETRY_WORKER;
       FULL_REPLAN -> Planner（仅用户触发）
    """
    current_result = state.get("current_result", {}) or {}
    tasks = state.get("tasks", []) or []
    cursor = state.get("cursor", 0)
    previous_results = state.get("results", []) or []
    
    task_name = _task_name(tasks, cursor)
    content_text = current_result.get("text_output") or current_result.get("content") or "无内容"
    
    # 1. 构造 Interrupt Payload
    payload = {
        "type": "verify_result",
        "task_name": task_name,
        "content_summary": content_text[:2000], # 限制长度供前端展示
        "full_content": content_text, # 前端可按需获取
        "guidance_text": f"任务 [{task_name}] 已完成，请审核结果。您可以命令我继续工作，或提供修改意见进行返工。"
    }
    
    # 2. 中断并等待反馈
    # user_feedback 可能是字符串（"通过"）或 结构化数据
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
            "message_type": "verifier_feedback",
            "task_id": current_result.get("task_id"),
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
    
    # 3. 分析反馈
    if _is_direct_approval(feedback_text):
        analysis = {
            "decision": "PASS",
            "reason": "用户明确确认当前任务结果",
            "suggestions": "",
        }
    else:
        analysis = _analyze_feedback(feedback_text, task_name, content_text, config)

    decision_code = _normalize_decision(analysis.get("decision", "PASS"))
    suggestions = analysis.get("suggestions", "")
    reason = analysis.get("reason", "")
    
    output_updates = {}
    content_obj = {}
    web_directive = web_authorization_directive(feedback_text)
    if web_directive is not None:
        output_updates["web_authorized"] = web_directive
    
    # 4. 路由逻辑
    if decision_code == "PASS":
        # 检查是否全部完成
        done = (cursor + 1) >= len(tasks)
        
        if done:
            final_decision = "DONE"
            content_obj = {
                "from": "Verifier",
                "to": "Summarizer",
                "type": "SUMMARIZE",
            }
        else:
            final_decision = "NEXT"
            content_obj = {
                "from": "Verifier",
                "to": "Planner",
                "type": "PROCEED",
                "current_section": task_name,
            }
            
        # 追加结果
        results = commit_current_result(state)
        output_updates["results"] = results
        statuses = record_section_status(
            state,
            VERIFIED_PASS,
            accepted_by="user",
            issues=[],
        )
        output_updates["section_status"] = statuses
        output_updates["report_status"] = derive_report_status(tasks, statuses)
        
    elif decision_code == "REWORK":
        final_decision = "RETRY_WORKER"
        content_obj = {
            "from": "Verifier",
            "to": "Worker",
            "type": "REWORK",
            "reason": reason,
            "suggestions": suggestions
        }
        # 将用户反馈传递给 Worker
        # 注意：需要确保 WorkerState 定义了 verifier_feedback 字段
        worker_updates = {
            "verifier_feedback": {
                "feedback": suggestions,
                "original_user_feedback": feedback_text
            }
        }
        # 使用 update 语义，LangGraph 会合并 worker_state
        output_updates["worker_state"] = worker_updates
        output_updates["results"] = previous_results # 不追加当前结果
        
    elif decision_code == "FULL_REPLAN":
        final_decision = "FULL_REPLAN"
        content_obj = {
            "from": "Verifier",
            "to": "Planner",
            "type": "FULL_REPLAN",
            "reason": reason,
            "current_section": task_name,
        }
        # 将反馈传递给 Planner
        output_updates["feedback"] = {
            "status": "BLOCKED",
            "issues": [{"description": reason, "suggestion": suggestions}]
        }
        output_updates["results"] = previous_results
        
    else:
        # _normalize_decision 保证只返回上面三个值；保留防御性兜底。
        final_decision = "NEXT"
        content_obj = {
            "from": "Verifier",
            "to": "Planner",
            "type": "PROCEED",
            "current_section": task_name,
        }
        output_updates["results"] = commit_current_result(state)
        statuses = record_section_status(
            state,
            VERIFIED_PASS,
            accepted_by="user",
            issues=[],
        )
        output_updates["section_status"] = statuses
        output_updates["report_status"] = derive_report_status(tasks, statuses)
    
    msg = AIMessage(content=json.dumps(content_obj, ensure_ascii=False))
    
    output_updates["decision"] = final_decision

    output_updates["messages"] = [
        feedback_message,
        msg,
    ]

    output_updates["docs"] = resumed_docs
    
    return output_updates

def decision(state: State, config: RunnableConfig, **kwargs):
    return state.get("decision", "DONE")

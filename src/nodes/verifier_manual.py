import json
import os
import hashlib
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from src.state import State
from src.llm import get_llm
from src.quality.models import QualityDimensions, ReviewIssue, ReviewRecord
from src.workflow_records import ensure_task_records, set_task_status
from src.workflow_store import WorkflowRecordStore

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
else:
    BaseStore = Any


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


def _append_result_once(previous_results, current_result):
    results = list(previous_results or [])
    if not current_result:
        return results

    current_task_id = current_result.get("task_id")
    if current_task_id is not None:
        existing = next(
            (
                index
                for index, result in enumerate(results)
                if isinstance(result, dict)
                and result.get("task_id") == current_task_id
            ),
            None,
        )
        if existing is not None:
            if results[existing] != current_result:
                results[existing] = current_result
            return results
    elif current_result in results:
        return results

    results.append(current_result)
    return results


def _human_review_record(
    *,
    state: State,
    task_id: str,
    artifact_id: str,
    decision_code: str,
    reason: str,
    suggestions: str,
    feedback_text: str,
    feedback_message_id: str | None,
) -> dict[str, Any]:
    issues = []
    status = "PASS"
    if decision_code == "REWORK":
        status = "REVISE"
        issues = [
            ReviewIssue(
                code="HUMAN_REVISION_REQUEST",
                category="CONTENT_DEFECT",
                severity="major",
                description=reason or feedback_text or "Human reviewer requested revision.",
                responsible_handler="worker_agent",
                revision_instruction=suggestions or feedback_text or "Revise the current section.",
            )
        ]
    elif decision_code == "FULL_REPLAN":
        status = "HUMAN_REVIEW"
        issues = [
            ReviewIssue(
                code="HUMAN_FULL_REPLAN_REQUEST",
                category="LOCAL_PLAN_DEFECT",
                severity="major",
                description=reason or feedback_text or "Human reviewer requested full replanning.",
                responsible_handler="planner",
                revision_instruction=suggestions or feedback_text or "Replan the remaining report tasks.",
            )
        ]

    stable = json.dumps(
        {
            "task_id": task_id,
            "artifact_id": artifact_id,
            "decision": decision_code,
            "feedback": feedback_text,
            "message_id": feedback_message_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    review_id = "review_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
    score = 5 if status == "PASS" else 0
    return ReviewRecord(
        review_id=review_id,
        task_id=task_id,
        artifact_id=artifact_id,
        reviewer="human",
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        status=status,
        issues=issues,
        quality_dimensions=QualityDimensions(
            completeness=score,
            evidence=score,
            logic=score,
            actionability=score,
            safety=score,
        ),
    ).model_dump(mode="json")


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


def verifier_manual(
    state: State,
    config: RunnableConfig,
    store: BaseStore | None = None,
    **kwargs,
):
    """
    人工审核节点：
    1. 展示当前结果摘要
    2. Interrupt 等待用户反馈
    3. Resume 后 LLM 分析反馈
    4. 路由：PASS/REWORK -> TaskController;
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
    
    task_id = str(current_result.get("task_id") or tasks[cursor].get("task_id"))
    artifact_id = str(
        current_result.get("artifact_id")
        or (state.get("active_artifact_ids") or {}).get(task_id)
        or f"legacy-{task_id}"
    )
    review_record = _human_review_record(
        state=state,
        task_id=task_id,
        artifact_id=artifact_id,
        decision_code=decision_code,
        reason=str(reason),
        suggestions=str(suggestions),
        feedback_text=feedback_text,
        feedback_message_id=feedback_message_id,
    )
    review_records = list(state.get("review_records") or [])
    previous_review = next(
        (
            item
            for item in review_records
            if isinstance(item, dict)
            and item.get("review_id") == review_record["review_id"]
        ),
        None,
    )
    if previous_review is not None:
        review_record = dict(previous_review)
    else:
        review_records.append(review_record)
    if store is not None:
        WorkflowRecordStore(
            store,
            str(state.get("user_id") or ""),
            str(state.get("job_id") or ""),
        ).put_review(review_record)

    output_updates = {
        "review_record": review_record,
        "review_records": review_records,
    }
    content_obj = {}
    records = ensure_task_records(state)
    
    # 4. 路由逻辑
    if decision_code == "PASS":
        final_decision = "NEXT"
        content_obj = {
            "from": "Verifier",
            "to": "TaskController",
            "type": "PROCEED",
            "current_section": task_name,
        }
        output_updates["results"] = _append_result_once(
            previous_results, current_result
        )
        output_updates["task_records"] = set_task_status(
            records,
            task_id,
            "PASSED",
            active_artifact_id=artifact_id,
        )
        
    elif decision_code == "REWORK":
        final_decision = "REWORK"
        content_obj = {
            "from": "Verifier",
            "to": "TaskController",
            "type": "REWORK",
            "reason": reason,
            "suggestions": suggestions
        }
        worker_updates = deepcopy(state.get("worker_state") or {})
        worker_updates.update(
            {
                "verifier_feedback": {
                    "feedback": suggestions,
                    "original_user_feedback": feedback_text,
                },
                "execution_feedback": {
                    "mode": "human_rework",
                    "issues": review_record["issues"],
                    "instructions": suggestions or feedback_text,
                    "responsible_handlers": ["worker_agent"],
                },
            }
        )
        output_updates["worker_state"] = worker_updates
        output_updates["task_records"] = set_task_status(
            records, task_id, "REVISE_REQUIRED"
        )
        output_updates["results"] = previous_results

    elif decision_code == "FULL_REPLAN":
        final_decision = "FULL_REPLAN"
        content_obj = {
            "from": "Verifier",
            "to": "Planner",
            "type": "FULL_REPLAN",
            "reason": reason,
            "current_section": task_name,
        }
        output_updates["feedback"] = {
            "status": "BLOCKED",
            "issues": [{"description": reason, "suggestion": suggestions}],
        }
        output_updates["results"] = previous_results

    else:
        final_decision = "NEXT"
        content_obj = {
            "from": "Verifier",
            "to": "TaskController",
            "type": "PROCEED",
            "current_section": task_name,
        }
        output_updates["results"] = _append_result_once(
            previous_results,
            current_result,
        )
        output_updates["task_records"] = set_task_status(
            records,
            task_id,
            "PASSED",
            active_artifact_id=artifact_id,
        )
    
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

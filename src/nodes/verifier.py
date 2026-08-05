"""Automatic task assessment without workflow-routing decisions."""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.config import get_app_config
from src.llm import get_llm
from src.state import State


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "verifier_logs.jsonl")

_CATEGORY_BY_CODE = {
    "EVIDENCE_GAP": "EVIDENCE_GAP",
    "INSUFFICIENT_EVIDENCE": "EVIDENCE_GAP",
    "MISSING_CITATION": "EVIDENCE_GAP",
    "MISSING_EVIDENCE": "EVIDENCE_GAP",
    "RAG_COVERAGE_GAP": "EVIDENCE_GAP",
    "RAG_INSUFFICIENT": "EVIDENCE_GAP",
    "SOURCE_UNSUPPORTED": "EVIDENCE_GAP",
    "INVALID_TASK_ORDER": "LOCAL_PLAN_DEFECT",
    "MISSING_DEPENDENCY": "LOCAL_PLAN_DEFECT",
    "RESOURCE_NOT_ASSIGNED": "LOCAL_PLAN_DEFECT",
    "TASK_GRANULARITY": "LOCAL_PLAN_DEFECT",
    "BAD_PLAN": "EXTERNAL_BLOCKER",
    "CONTRADICTORY_REQUIREMENTS": "EXTERNAL_BLOCKER",
    "EXTERNAL_BLOCKER": "EXTERNAL_BLOCKER",
    "INVALID_PLAN": "EXTERNAL_BLOCKER",
    "LLM_ERROR": "EXTERNAL_BLOCKER",
    "LLM_NOT_ENABLED": "EXTERNAL_BLOCKER",
    "PERMISSION_DENIED": "EXTERNAL_BLOCKER",
    "REQUIREMENTS_CONFLICT": "EXTERNAL_BLOCKER",
    "RESOURCE_UNAVAILABLE": "EXTERNAL_BLOCKER",
    "UNEXECUTABLE_TASK": "EXTERNAL_BLOCKER",
}
_VALID_CATEGORIES = {
    "CONTENT_DEFECT",
    "EVIDENCE_GAP",
    "LOCAL_PLAN_DEFECT",
    "EXTERNAL_BLOCKER",
}


def _task_name(tasks: list[dict[str, Any]], index: int) -> str:
    if 0 <= index < len(tasks):
        task = tasks[index]
        return task.get("task_name") or task.get("task_id") or f"Task_{index}"
    return f"Task_{index}"


def _clean_json_fences(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def verifier(state: State, config: RunnableConfig, **kwargs) -> dict[str, Any]:
    """Return only a sanitized assessment of the current Worker result."""
    current_result = state.get("current_result", {}) or {}
    tasks = state.get("tasks", []) or []
    cursor = int(state.get("cursor", 0) or 0)

    try:
        configurable = config.get("configurable", {}) if config else {}
        use_llm = bool(configurable.get("use_llm")) or bool(
            get_app_config().deepseek_api_key
        )
    except Exception:
        use_llm = bool(get_app_config().deepseek_api_key)

    llm_record: dict[str, Any] = {}
    if use_llm:
        try:
            current_task = tasks[cursor] if 0 <= cursor < len(tasks) else {}
            content = current_result.get("content") or current_result.get("text_output") or ""
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
            prompt_path = os.path.join(
                os.path.dirname(__file__), "..", "prompts", "verifier.md"
            )
            with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                template = prompt_file.read()
            format_instructions = (
                "仅输出 JSON 对象：status 为 PASS|FAILED|BLOCKED；issues 每项必须含 "
                "code、category、description、suggestion、severity，可选 resource_name；"
                "另含 current_section、requirements_met、requirements_missing。不得输出路由建议。"
            )
            chain = ChatPromptTemplate.from_messages([("system", template)]) | get_llm(
                config, json_mode=True
            )
            response = chain.invoke(
                {
                    "task_name": _task_name(tasks, cursor),
                    "task_requirements": json.dumps(
                        current_task, ensure_ascii=False, indent=2
                    ),
                    "worker_result": content,
                    "worker_assets": worker_assets,
                    "format_instructions": format_instructions,
                }
            )
            assessment = json.loads(_clean_json_fences(str(response.content)))
            llm_record["response_snippet"] = str(assessment)[:500]
        except Exception as exc:
            assessment = _service_error_assessment(tasks, cursor, "LLM_ERROR", str(exc))
            llm_record["error"] = str(exc)
    else:
        assessment = _service_error_assessment(
            tasks,
            cursor,
            "LLM_NOT_ENABLED",
            "LLM is not enabled for verification",
        )

    sanitized = _sanitize_assessment(assessment, state)
    plan_revision = int(state.get("plan_revision", 1) or 1)
    print(
        "🔍 AutoVerifier assessment: "
        f"task={_task_name(tasks, cursor)} status={sanitized.get('status')} "
        f"plan_revision={plan_revision}"
    )
    _log_verifier_output(state, sanitized, llm_record)
    return {"assessment": sanitized}


def _service_error_assessment(
    tasks: list[dict[str, Any]], cursor: int, code: str, description: str
) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "current_section": _task_name(tasks, cursor),
        "issues": [
            {
                "code": code,
                "category": "EXTERNAL_BLOCKER",
                "description": description,
                "suggestion": "Retry automatic verification after checking model configuration.",
                "severity": "error",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["automatic quality assessment"],
    }


def _sanitize_assessment(assessment: dict[str, Any], state: State) -> dict[str, Any]:
    """Normalize the assessment contract and discard inapplicable asset issues."""
    assessment = assessment if isinstance(assessment, dict) else {}
    tasks = state.get("tasks", []) or []
    cursor = int(state.get("cursor", 0) or 0)
    current_task = tasks[cursor] if 0 <= cursor < len(tasks) else {}
    description = str(current_task.get("task_description") or "")
    requires_table = bool(current_task.get("generate_table")) or any(
        marker in description for marker in ("表格", "数据表", "生成表")
    )
    requires_image = bool(current_task.get("generate_figure")) or any(
        marker in description for marker in ("趋势图", "因果图", "流程图", "生成图")
    )

    issues: list[dict[str, Any]] = []
    valid_issue_seen = False
    for raw_issue in assessment.get("issues") or []:
        if isinstance(raw_issue, str):
            valid_issue_seen = True
            issue = {"code": raw_issue.upper(), "description": raw_issue}
        elif isinstance(raw_issue, dict):
            valid_issue_seen = True
            issue = dict(raw_issue)
        else:
            continue
        code = str(issue.get("code") or "UNSPECIFIED_ISSUE").strip().upper()
        if code == "MISSING_TABLE" and not requires_table:
            continue
        if code in {"MISSING_IMAGE", "MISSING_FIGURE"} and not requires_image:
            continue
        supplied_category = str(issue.get("category") or "").strip().upper()
        category = _CATEGORY_BY_CODE.get(code)
        if category is None:
            category = (
                supplied_category
                if supplied_category in _VALID_CATEGORIES
                else "CONTENT_DEFECT"
            )
        normalized = {
            "code": code,
            "category": category,
            "description": str(issue.get("description") or code).strip(),
            "suggestion": str(issue.get("suggestion") or "Correct this issue.").strip(),
            "severity": str(issue.get("severity") or "major").strip().lower(),
        }
        resource_name = str(issue.get("resource_name") or "").strip()
        if resource_name:
            normalized["resource_name"] = resource_name
        issues.append(normalized)

    requirements_missing = list(assessment.get("requirements_missing") or [])
    status = str(assessment.get("status") or "FAILED").strip().upper()
    if status not in {"PASS", "FAILED", "BLOCKED"}:
        status = "FAILED"
    if issues and status == "PASS":
        status = "FAILED"
    if not issues and status in {"FAILED", "BLOCKED"}:
        if valid_issue_seen and not requirements_missing:
            status = "PASS"
        else:
            status = "FAILED"
            issues = [
                {
                    "code": "ASSESSMENT_CONTRACT_ERROR",
                    "category": "CONTENT_DEFECT",
                    "description": "Verifier returned a failure without usable issue details.",
                    "suggestion": "Retry automatic verification with a valid structured assessment.",
                    "severity": "error",
                }
            ]
    return {
        "status": status,
        "current_section": assessment.get("current_section") or _task_name(tasks, cursor),
        "issues": issues,
        "requirements_met": list(assessment.get("requirements_met") or []),
        "requirements_missing": requirements_missing,
    }


def _log_verifier_output(
    state: State,
    assessment: dict[str, Any],
    llm_record: dict[str, Any] | None = None,
) -> None:
    def safe(value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            return str(value)

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task_id": safe((state.get("current_result") or {}).get("task_id")),
        "cursor": safe(state.get("cursor")),
        "plan_revision": int(state.get("plan_revision", 1) or 1),
        "assessment": safe(assessment),
    }
    if llm_record:
        entry["llm"] = safe(llm_record)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

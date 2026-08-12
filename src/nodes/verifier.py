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
    "INVALID_CITATION_ID": "EVIDENCE_GAP",
    "MISSING_INLINE_CITATION": "EVIDENCE_GAP",
    "INVALID_TASK_ORDER": "LOCAL_PLAN_DEFECT",
    "MISSING_DEPENDENCY": "LOCAL_PLAN_DEFECT",
    "RESOURCE_NOT_ASSIGNED": "LOCAL_PLAN_DEFECT",
    "TASK_GRANULARITY": "LOCAL_PLAN_DEFECT",
    "BAD_PLAN": "EXTERNAL_BLOCKER",
    "CONTRADICTORY_REQUIREMENTS": "EXTERNAL_BLOCKER",
    "EXTERNAL_BLOCKER": "EXTERNAL_BLOCKER",
    "INVALID_PLAN": "EXTERNAL_BLOCKER",
    "LLM_ERROR": "VERIFIER_FAILURE",
    "LLM_NOT_ENABLED": "VERIFIER_FAILURE",
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
    "VERIFIER_FAILURE",
}
_ISSUE_REQUIRED_STRING_FIELDS = {
    "code",
    "category",
    "description",
    "suggestion",
    "severity",
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
    sanitized = _apply_citation_integrity(sanitized, current_result)
    plan_revision = int(state.get("plan_revision", 1) or 1)
    current_task = tasks[cursor] if 0 <= cursor < len(tasks) else {}
    issues = list(sanitized.get("issues") or [])
    print(
        "🔍 AutoVerifier assessment: "
        f"task={current_task.get('task_id') or cursor} "
        f"name={_task_name(tasks, cursor)} status={sanitized.get('status')} "
        f"issue_count={len(issues)} plan_revision={plan_revision}"
    )
    for issue in issues[:5]:
        description = re.sub(r"\s+", " ", str(issue.get("description") or "")).strip()
        print(f"  - {issue.get('code')}: {description}")
    _log_verifier_output(state, sanitized, llm_record)
    return {"assessment": sanitized}


def _apply_citation_integrity(
    assessment: dict[str, Any],
    current_result: dict[str, Any],
) -> dict[str, Any]:
    """Add deterministic failures for missing or invented inline evidence IDs."""
    citations = current_result.get("citations") or []
    known_ids = {
        str(citation.get("evidence_id") or "").strip().upper()
        for citation in citations
        if isinstance(citation, dict) and citation.get("evidence_id")
    }
    content = str(
        current_result.get("content") or current_result.get("text_output") or ""
    )
    cited_ids = {
        value.upper() for value in re.findall(r"\[(E\d+)\]", content, re.IGNORECASE)
    }
    unknown_ids = cited_ids - known_ids
    issue = None
    missing_requirement = "正文中的证据编号绑定"
    if unknown_ids:
        issue = {
            "code": "INVALID_CITATION_ID",
            "category": "EVIDENCE_GAP",
            "description": "正文引用了不存在的证据编号："
            + ", ".join(sorted(unknown_ids)),
            "suggestion": "仅使用 current_result.citations 中存在的 [E编号]。",
            "severity": "major",
        }
    elif known_ids and not cited_ids:
        issue = {
            "code": "MISSING_INLINE_CITATION",
            "category": "EVIDENCE_GAP",
            "description": "正文已有结构化证据，但没有把具体论断绑定到 [E编号]。",
            "suggestion": "在对应论断或段落后添加真实的 [E编号]。",
            "severity": "major",
        }
    if issue is None:
        return assessment

    updated = dict(assessment)
    issues = list(updated.get("issues") or [])
    if not any(item.get("code") == issue["code"] for item in issues):
        issues.append(issue)
    requirements_missing = list(updated.get("requirements_missing") or [])
    if missing_requirement not in requirements_missing:
        requirements_missing.append(missing_requirement)
    updated.update(
        {
            "status": "FAILED",
            "issues": issues,
            "requirements_missing": requirements_missing,
        }
    )
    return updated


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


def _contract_error_assessment(
    assessment: dict[str, Any], tasks: list[dict[str, Any]], cursor: int
) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "current_section": assessment.get("current_section")
        or _task_name(tasks, cursor),
        "issues": [
            {
                "code": "ASSESSMENT_CONTRACT_ERROR",
                "category": "VERIFIER_FAILURE",
                "description": "Verifier returned malformed collection fields.",
                "suggestion": "Retry automatic verification with a valid structured assessment.",
                "severity": "error",
            }
        ],
        "requirements_met": [],
        "requirements_missing": [],
    }


def _assessment_elements_are_valid(assessment: dict[str, Any]) -> bool:
    for field in ("requirements_met", "requirements_missing"):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in assessment[field]
        ):
            return False
    for issue in assessment["issues"]:
        if not isinstance(issue, dict):
            return False
        if any(
            not isinstance(issue.get(field), str) or not issue[field].strip()
            for field in _ISSUE_REQUIRED_STRING_FIELDS
        ):
            return False
        if str(issue["category"]).strip().upper() not in _VALID_CATEGORIES:
            return False
        if "resource_name" in issue and (
            not isinstance(issue["resource_name"], str)
            or not issue["resource_name"].strip()
        ):
            return False
    return True


def _sanitize_assessment(assessment: dict[str, Any], state: State) -> dict[str, Any]:
    """Normalize the assessment contract and discard inapplicable asset issues."""
    assessment = assessment if isinstance(assessment, dict) else {}
    tasks = state.get("tasks", []) or []
    cursor = int(state.get("cursor", 0) or 0)
    collection_fields = ("issues", "requirements_met", "requirements_missing")
    if any(not isinstance(assessment.get(field), list) for field in collection_fields):
        return _contract_error_assessment(assessment, tasks, cursor)
    if not _assessment_elements_are_valid(assessment):
        return _contract_error_assessment(assessment, tasks, cursor)
    current_task = tasks[cursor] if 0 <= cursor < len(tasks) else {}
    description = str(current_task.get("task_description") or "")
    requires_table = bool(current_task.get("generate_table")) or any(
        marker in description for marker in ("表格", "数据表", "生成表")
    )
    requires_image = bool(current_task.get("generate_figure")) or any(
        marker in description for marker in ("趋势图", "因果图", "流程图", "生成图")
    )

    issues: list[dict[str, Any]] = []
    valid_issue_seen = bool(assessment["issues"])
    for raw_issue in assessment["issues"]:
        issue = dict(raw_issue)
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

    requirements_missing = list(assessment["requirements_missing"])
    status = str(assessment.get("status") or "FAILED").strip().upper()
    if status not in {"PASS", "FAILED", "BLOCKED"}:
        status = "FAILED"
    if requirements_missing and status == "PASS":
        status = "FAILED"
        if not issues:
            issues = [
                {
                    "code": "REQUIREMENT_MISSING",
                    "category": "CONTENT_DEFECT",
                    "description": f"Required outcome is missing: {requirement}",
                    "suggestion": f"Address the missing requirement: {requirement}",
                    "severity": "major",
                }
                for requirement in requirements_missing
                if str(requirement).strip()
            ]
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
                    "category": "VERIFIER_FAILURE",
                    "description": "Verifier returned a failure without usable issue details.",
                    "suggestion": "Retry automatic verification with a valid structured assessment.",
                    "severity": "error",
                }
            ]
    return {
        "status": status,
        "current_section": assessment.get("current_section") or _task_name(tasks, cursor),
        "issues": issues,
        "requirements_met": list(assessment["requirements_met"]),
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

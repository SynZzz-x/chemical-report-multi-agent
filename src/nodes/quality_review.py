"""Independent Artifact quality assessment without workflow routing."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from src.config import get_app_config
from src.llm import get_llm
from src.quality.models import (
    QualityDimensions,
    ReviewAssessment,
    ReviewIssue,
    ReviewRecord,
)
from src.quality.validators import validate_artifact
from src.workflow_store import WorkflowRecordStore

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
else:
    BaseStore = Any


def _current_task(state: dict[str, Any]) -> dict[str, Any]:
    current = state.get("current_task")
    if isinstance(current, dict) and current:
        return current
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    return tasks[cursor] if 0 <= cursor < len(tasks) else {}


def _clean_json_fences(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _read_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "quality_review.md")
    with open(path, "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


def _review_failure(code: str, description: str) -> ReviewAssessment:
    return ReviewAssessment(
        status="BLOCKED",
        issues=[
            ReviewIssue(
                code=code,
                category="REVIEW_FAILURE",
                severity="error",
                description=description,
                responsible_handler="quality_review",
                revision_instruction="Retry quality review without rerunning Worker.",
            )
        ],
        quality_dimensions=QualityDimensions(
            completeness=0,
            evidence=0,
            logic=0,
            actionability=0,
            safety=0,
        ),
    )


def _semantic_assessment(state, task, artifact, config) -> ReviewAssessment:
    configurable = config.get("configurable", {}) if config else {}
    try:
        use_llm = bool(configurable.get("use_llm")) or bool(
            get_app_config().deepseek_api_key
        )
    except Exception:
        use_llm = bool(configurable.get("use_llm"))
    if not use_llm:
        return _review_failure("REVIEW_NOT_CONFIGURED", "Quality-review model is not configured.")

    payload = {
        "task": task,
        "artifact": {
            "artifact_id": artifact.get("artifact_id"),
            "content": artifact.get("content") or artifact.get("text_output"),
            "tables": artifact.get("tables") or [],
            "figures": artifact.get("figures") or [],
            "citations": artifact.get("citations") or [],
            "evidence_coverage": artifact.get("evidence_coverage") or {},
        },
    }
    try:
        model = get_llm(config, json_mode=True)
        message = HumanMessage(
            content=_read_prompt()
            + "\n\nINPUT JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        response = model.invoke([message], config=config)
        return ReviewAssessment.model_validate_json(
            _clean_json_fences(str(response.content))
        )
    except Exception as exc:
        return _review_failure("REVIEW_SERVICE_ERROR", str(exc))


def _merge_issues(
    deterministic: list[ReviewIssue], semantic: ReviewAssessment
) -> ReviewAssessment:
    issues: list[ReviewIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in [*deterministic, *semantic.issues]:
        key = (issue.code, issue.description)
        if key not in seen:
            seen.add(key)
            issues.append(issue)
    if not issues:
        status = "PASS"
    elif any(issue.category == "SAFETY_BOUNDARY" for issue in issues):
        status = "HUMAN_REVIEW"
    elif any(issue.category in {"EXTERNAL_BLOCKER", "REVIEW_FAILURE"} for issue in issues):
        status = "BLOCKED"
    else:
        status = "REVISE"
    return ReviewAssessment(
        status=status,
        issues=issues,
        quality_dimensions=semantic.quality_dimensions,
    )


def _review_id(task_id: str, artifact_id: str, assessment: ReviewAssessment) -> str:
    stable = json.dumps(
        {
            "task_id": task_id,
            "artifact_id": artifact_id,
            **assessment.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "review_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]


def _legacy_assessment(record: dict[str, Any]) -> dict[str, Any]:
    issues = [
        {
            "code": issue["code"],
            "category": issue["category"],
            "description": issue["description"],
            "suggestion": issue["revision_instruction"],
            "severity": issue["severity"],
            "evidence_refs": issue.get("evidence_refs") or [],
            "responsible_handler": issue["responsible_handler"],
            **(
                {"resource_name": issue["resource_name"]}
                if issue.get("resource_name")
                else {}
            ),
        }
        for issue in record["issues"]
    ]
    return {
        "status": "PASS" if record["status"] == "PASS" else "FAILED",
        "current_section": record["task_id"],
        "artifact_id": record["artifact_id"],
        "issues": issues,
        "requirements_met": [],
        "requirements_missing": [issue["description"] for issue in issues],
    }


def quality_review(
    state,
    config,
    store: BaseStore | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Assess only the active Artifact and emit no routing decision."""
    task = _current_task(state)
    artifact = state.get("current_result") or {}
    task_id = str(task.get("task_id") or artifact.get("task_id") or "")
    artifact_id = str(artifact.get("artifact_id") or "")
    deterministic = validate_artifact(
        task,
        artifact,
        active_artifact_id=(state.get("active_artifact_ids") or {}).get(task_id),
    )
    semantic = _semantic_assessment(state, task, artifact, config)
    assessment = _merge_issues(deterministic, semantic)
    record = ReviewRecord(
        **assessment.model_dump(mode="json"),
        review_id=_review_id(task_id, artifact_id, assessment),
        task_id=task_id,
        artifact_id=artifact_id,
        reviewer="quality_review_agent",
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    ).model_dump(mode="json")

    existing = list(state.get("review_records") or [])
    prior = next(
        (
            item
            for item in existing
            if isinstance(item, dict) and item.get("review_id") == record["review_id"]
        ),
        None,
    )
    if prior is not None:
        record = dict(prior)
    else:
        existing.append(record)

    if store is not None:
        WorkflowRecordStore(
            store,
            str(state.get("user_id") or ""),
            str(state.get("job_id") or ""),
        ).put_review(record)

    return {
        "review_record": record,
        "review_records": existing,
        "assessment": _legacy_assessment(record),
    }

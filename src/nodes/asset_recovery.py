"""Asset-only recovery for accepted report body results."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Mapping, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from ..evidence.coverage import assess_coverage
from ..evidence.models import EvidenceBundle, EvidenceRecord
from ..evidence.normalizer import normalize_rag_tool_calls
from ..recovery.policy import WorkflowAction
from ..report_validation import extract_markdown_tables
from ..state import State
from ..utils.path_manager import get_job_subdir
from .worker.tools.concept_graph_tool import ConceptGraphTool


logger = logging.getLogger(__name__)


def _current_task(state: State) -> dict[str, Any]:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], Mapping):
        return dict(tasks[cursor])
    current = state.get("current_task") or {}
    return dict(current) if isinstance(current, Mapping) else {}


def _issue_codes(state: State) -> set[str]:
    return {
        str(issue.get("code") or "").strip().upper()
        for issue in (state.get("assessment") or {}).get("issues") or []
        if isinstance(issue, Mapping)
    }


def _focused_worker_rework(
    state: State,
    current_result: dict[str, Any],
    instructions: str,
) -> dict[str, Any]:
    worker_state = deepcopy(state.get("worker_state") or {})
    worker_state["execution_feedback"] = {
        "mode": "asset_content_rework",
        "issues": deepcopy((state.get("assessment") or {}).get("issues") or []),
        "instructions": instructions,
    }
    return {
        "workflow_action": WorkflowAction.REWORK.value,
        "current_result": current_result,
        "worker_state": worker_state,
        "asset_recovery_error": instructions,
    }


def _evidence_bundle(result: Mapping[str, Any]) -> EvidenceBundle:
    records: list[EvidenceRecord] = []
    for citation in result.get("citations") or []:
        if not isinstance(citation, Mapping):
            continue
        try:
            records.append(EvidenceRecord.model_validate(dict(citation)))
        except ValidationError:
            continue
    if records:
        return EvidenceBundle(records=tuple(records))
    return normalize_rag_tool_calls(
        call
        for call in result.get("tool_calls") or []
        if isinstance(call, Mapping)
    )


def _recover_causal_figure(
    state: State,
    config: Optional[RunnableConfig],
    task: dict[str, Any],
    current_result: dict[str, Any],
) -> dict[str, Any]:
    visualization = task.get("visualization") or {}
    if not isinstance(visualization, Mapping) or str(
        visualization.get("kind") or ""
    ).strip().lower() != "causal":
        return _focused_worker_rework(
            state,
            current_result,
            "当前图形不是可由既有证据独立恢复的因果概念图，请只补充要求的正式图形资产。",
        )

    evidence = _evidence_bundle(current_result)
    required_concepts = [
        str(value).strip()
        for value in visualization.get("required_concepts") or []
        if str(value).strip()
    ]
    coverage = assess_coverage(evidence, required_concepts)
    if coverage.status != "sufficient":
        error = (
            "概念图证据覆盖不足："
            + ("、".join(coverage.uncovered_concepts) or coverage.status)
        )
        logger.warning("Asset recovery failed: task=%s error=%s", task.get("task_id"), error)
        return {
            "workflow_action": WorkflowAction.RETRY_VERIFIER.value,
            "current_result": current_result,
            "asset_recovery_error": error,
        }

    try:
        result = ConceptGraphTool().execute(
            task,
            evidence,
            get_job_subdir(state, "charts", config),
        )
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
    if not result.get("success") or not isinstance(result.get("figure"), Mapping):
        error = str(result.get("error") or "concept graph generation failed")
        logger.warning("Asset recovery failed: task=%s error=%s", task.get("task_id"), error)
        return {
            "workflow_action": WorkflowAction.RETRY_VERIFIER.value,
            "current_result": current_result,
            "asset_recovery_error": error,
        }

    candidate = deepcopy(current_result)
    candidate["figures"] = [dict(result["figure"])]
    candidate["graph_spec"] = deepcopy(result.get("graph_spec") or {})
    candidate["figures_generated"] = len(candidate["figures"])
    logger.info("Asset recovery succeeded: task=%s asset=figure", task.get("task_id"))
    return {
        "workflow_action": WorkflowAction.RETRY_VERIFIER.value,
        "current_result": candidate,
        "asset_recovery_error": "",
    }


def asset_recovery(
    state: State, config: Optional[RunnableConfig] = None, **kwargs
) -> dict[str, Any]:
    """Repair only independently recoverable assets from the current result."""

    current_result = deepcopy(state.get("current_result") or {})
    if not current_result:
        return _focused_worker_rework(
            state,
            current_result,
            "当前任务没有可复用的正文结果，必须重新执行正文任务。",
        )

    original_result = deepcopy(current_result)
    task = _current_task(state)
    task_id = str(task.get("task_id") or "").strip()
    result_task_id = str(current_result.get("task_id") or "").strip()
    if not task_id or result_task_id != task_id:
        return _focused_worker_rework(
            state,
            current_result,
            (
                "当前结果所属任务与活动任务不一致："
                f"result={result_task_id or '<missing>'}, "
                f"task={task_id or '<missing>'}，必须重新执行正文任务。"
            ),
        )
    codes = _issue_codes(state)
    if "MISSING_TABLE" in codes:
        tables = extract_markdown_tables(
            str(current_result.get("text_output") or current_result.get("content") or "")
        )
        if not tables:
            return _focused_worker_rework(
                state,
                current_result,
                "当前正文没有可确定性物化的 Markdown 表格，请保留正文事实并补写要求的表格。",
            )
        current_result["tables"] = tables
        logger.info("Asset recovery succeeded: task=%s asset=table", task.get("task_id"))

    if "MISSING_FIGURE" in codes:
        update = _recover_causal_figure(state, config, task, current_result)
        if update.get("asset_recovery_error") or update.get("workflow_action") == (
            WorkflowAction.REWORK.value
        ):
            update["current_result"] = original_result
        return update

    if "MISSING_TABLE" in codes:
        return {
            "workflow_action": WorkflowAction.RETRY_VERIFIER.value,
            "current_result": current_result,
            "asset_recovery_error": "",
        }

    return _focused_worker_rework(
        state,
        current_result,
        "当前验收结果不包含可独立恢复的资产缺陷。",
    )


def route_after_asset_recovery(state: State, **kwargs) -> str:
    return str(state.get("workflow_action") or WorkflowAction.RETRY_VERIFIER.value)

"""Constrained aggregation for conclusion/summary execution tasks."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Mapping, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..evidence.identity import normalize_sections_evidence
from ..llm import get_llm
from ..report_acceptance import is_admitted_section_entry
from ..report_acceptance import (
    USER_ACCEPTED_GAP,
    USER_ACCEPTED_WARNING,
    VERIFIED_PASS,
)
from ..report_validation import count_report_length
from ..state import State


logger = logging.getLogger(__name__)

_EVIDENCE_ID = re.compile(r"\[(E\d+)\]", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z\d])\d+(?:\.\d+)?(?:\s*%|‰)?")
_TECHNICAL_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])(?:pH|[A-Za-z]{2,}(?:[-_/][A-Za-z0-9]+)*|[A-Za-z]+\d+[A-Za-z0-9-]*)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_PHANTOM_ACTIONS = (
    "补充检索",
    "本次检索",
    "网络检索",
    "查阅外部",
    "历史批次数据",
    "回归分析",
    "相关性分析",
    "正交试验",
    "现场试验",
    "现场验证",
)
_IDENTIFIER_ALLOWLIST = {
    "abstract",
    "markdown",
    "pdf",
    "docx",
    "worker",
    "verifier",
    "rag",
    "web",
}


def _current_task(state: Mapping[str, Any]) -> dict[str, Any]:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks) and isinstance(tasks[cursor], Mapping):
        return dict(tasks[cursor])
    return {}


def _revision(value: Any, default: int = 1) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _citation_id(citation: Mapping[str, Any]) -> str:
    return str(citation.get("evidence_id") or "").strip().upper()


def build_synthesis_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build a revision-safe, read-only context from admitted prior sections."""

    tasks = [task for task in (state.get("tasks") or []) if isinstance(task, Mapping)]
    cursor = int(state.get("cursor", 0) or 0)
    prior_tasks = tasks[: max(cursor, 0)]
    task_by_id = {
        str(task.get("task_id")): task
        for task in prior_tasks
        if task.get("task_id") is not None
    }
    result_by_id = {
        str(result.get("task_id")): result
        for result in (state.get("results") or [])
        if isinstance(result, Mapping) and result.get("task_id") is not None
    }
    statuses = state.get("section_status") or {}
    verified_sections: list[dict[str, Any]] = []
    accepted_gap_sections: list[dict[str, Any]] = []
    warning_sections: list[dict[str, Any]] = []
    accepted_citations: list[dict[str, Any]] = []
    known_gaps: list[dict[str, Any]] = []

    for task_id, task in task_by_id.items():
        status = statuses.get(task_id)
        result = result_by_id.get(task_id)
        if not is_admitted_section_entry(status) or not isinstance(result, Mapping):
            continue
        expected_plan_revision = _revision(status.get("plan_revision"))
        expected_task_revision = _revision(status.get("task_revision"))
        current_task_revision = _revision(
            (state.get("task_revisions") or {}).get(task_id)
        )
        if (
            _revision(result.get("plan_revision")) != expected_plan_revision
            or _revision(result.get("task_revision")) != expected_task_revision
            or expected_task_revision != current_task_revision
        ):
            continue
        text = str(result.get("content") or result.get("text_output") or "").strip()
        if not text:
            continue
        section = {
            "task_id": task_id,
            "title": str(task.get("task_name") or task_id),
            "covers_sections": list(task.get("covers_sections") or []),
            "status": str(status.get("status") or ""),
            "content": text,
            "citations": [
                dict(citation)
                for citation in (result.get("citations") or [])
                if isinstance(citation, Mapping) and _citation_id(citation)
            ],
        }
        section_status = str(status.get("status") or "")
        if section_status == VERIFIED_PASS:
            verified_sections.append(section)
        elif section_status == USER_ACCEPTED_GAP:
            accepted_gap_sections.append(section)
        elif section_status == USER_ACCEPTED_WARNING:
            warning_sections.append(section)
        for issue in status.get("issues") or []:
            if isinstance(issue, Mapping) and "EVIDENCE" in str(issue.get("code") or "").upper():
                known_gaps.append({"task_id": task_id, **dict(issue)})

        waiver = (state.get("accepted_evidence_gaps") or {}).get(task_id)
        if not isinstance(waiver, Mapping):
            continue
        if (
            _revision(waiver.get("plan_revision"), 0) != expected_plan_revision
            or _revision(waiver.get("task_revision"), 0) != expected_task_revision
        ):
            continue
        known_gaps.extend(
            {"task_id": task_id, **dict(issue)}
            for issue in (waiver.get("issues") or [])
            if isinstance(issue, Mapping)
        )

    verified_sections, evidence_display_map = normalize_sections_evidence(
        verified_sections
    )
    accepted_citations = [
        dict(citation)
        for section in verified_sections
        for citation in section.get("citations") or []
        if isinstance(citation, Mapping) and _citation_id(citation)
    ]
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for citation in accepted_citations
            if (evidence_id := _citation_id(citation))
        )
    )
    unique_gaps: list[dict[str, Any]] = []
    seen_gaps: set[str] = set()
    for gap in known_gaps:
        signature = json.dumps(gap, ensure_ascii=False, sort_keys=True, default=str)
        if signature not in seen_gaps:
            seen_gaps.add(signature)
            unique_gaps.append(gap)
    return {
        # Compatibility name now deliberately means verified factual inputs only.
        "accepted_sections": verified_sections,
        "verified_sections": verified_sections,
        "accepted_gap_sections": accepted_gap_sections,
        "warning_sections": warning_sections,
        "accepted_citations": accepted_citations,
        "accepted_evidence_ids": evidence_ids,
        "evidence_display_map": evidence_display_map,
        "known_gaps": unique_gaps,
    }


def _comparable_text(text: str) -> str:
    value = _EVIDENCE_ID.sub("", str(text or ""))
    value = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", value)
    value = re.sub(r"(?m)^\s*(?:[-*+] |\d+[.)、．]\s*)", "", value)
    return value


def _sentences(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])|\n+", str(text or ""))
        if part.strip()
    ]
    sentences: list[str] = []
    for part in parts:
        citation_prefix = re.match(r"^((?:\[E\d+\]\s*)+)(.*)$", part, re.IGNORECASE)
        if citation_prefix and sentences:
            sentences[-1] = f"{sentences[-1]}{citation_prefix.group(1).strip()}"
            remainder = citation_prefix.group(2).strip()
            if remainder:
                sentences.append(remainder)
            continue
        sentences.append(part)
    return sentences


def _normalized_claim(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u3400-\u9fff]", "", _comparable_text(text)).casefold()


def _sentence_record(sentence: str) -> tuple[str, frozenset[str]]:
    return (
        _normalized_claim(sentence),
        frozenset(value.upper() for value in _EVIDENCE_ID.findall(sentence)),
    )


def _grounding_findings(
    candidate: str, source: str
) -> tuple[list[str], list[str]]:
    source_records: dict[str, list[frozenset[str]]] = {}
    for sentence in _sentences(source):
        core, evidence_ids = _sentence_record(sentence)
        if core:
            source_records.setdefault(core, []).append(evidence_ids)

    ungrounded: list[str] = []
    rebound: list[str] = []
    for sentence in _sentences(candidate):
        core, evidence_ids = _sentence_record(sentence)
        if not core:
            continue
        source_evidence_sets = source_records.get(core)
        if not source_evidence_sets:
            ungrounded.append(sentence)
            continue
        if evidence_ids and not any(
            evidence_ids.issubset(source_ids)
            for source_ids in source_evidence_sets
        ):
            rebound.append(sentence)
    return ungrounded, rebound


def _technical_identifiers(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _TECHNICAL_IDENTIFIER.finditer(_comparable_text(text))
        if match.group(0).casefold() not in _IDENTIFIER_ALLOWLIST
    }


def check_synthesis_consistency(
    candidate: str, context: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Return high-confidence drift findings without semantic guesswork."""

    source = "\n".join(
        str(section.get("content") or "")
        for section in context.get("accepted_sections") or []
        if isinstance(section, Mapping)
    )
    gap_source = "\n".join(
        str(gap.get("description") or gap.get("suggestion") or "")
        for gap in context.get("known_gaps") or []
        if isinstance(gap, Mapping)
    )
    if gap_source:
        source = f"{source}\n{gap_source}"
    issues: list[dict[str, str]] = []
    allowed_ids = {
        str(value).upper() for value in context.get("accepted_evidence_ids") or []
    }
    new_ids = sorted(
        {value.upper() for value in _EVIDENCE_ID.findall(candidate)} - allowed_ids
    )
    if new_ids:
        issues.append(
            {
                "code": "NEW_EVIDENCE_ID",
                "description": "结论使用了未被前文验收的证据编号：" + ", ".join(new_ids),
            }
        )

    source_numbers = set(_NUMBER.findall(_comparable_text(source)))
    new_numbers = sorted(set(_NUMBER.findall(_comparable_text(candidate))) - source_numbers)
    if new_numbers:
        issues.append(
            {
                "code": "NEW_NUMBER",
                "description": "结论新增了前文未出现的数字：" + ", ".join(new_numbers),
            }
        )

    new_identifiers = sorted(_technical_identifiers(candidate) - _technical_identifiers(source))
    if new_identifiers:
        issues.append(
            {
                "code": "NEW_TECHNICAL_IDENTIFIER",
                "description": "结论新增了前文未出现的技术标识：" + ", ".join(new_identifiers),
            }
        )

    phantom_actions = [
        marker for marker in _PHANTOM_ACTIONS if marker in candidate and marker not in source
    ]
    if phantom_actions:
        issues.append(
            {
                "code": "PHANTOM_ACTION",
                "description": "结论声称执行了前文未记录的操作：" + ", ".join(phantom_actions),
            }
        )
    ungrounded, rebound = _grounding_findings(candidate, source)
    if ungrounded:
        issues.append(
            {
                "code": "UNGROUNDED_CLAIM",
                "description": "结论包含无法映射到已验收前文的新增表述："
                + " | ".join(ungrounded[:3]),
            }
        )
    if rebound:
        issues.append(
            {
                "code": "CITATION_REBIND",
                "description": "结论把已验收证据编号绑定到了其他原句："
                + " | ".join(rebound[:3]),
            }
        )
    return issues


def _prompt_messages(
    task: Mapping[str, Any],
    context: Mapping[str, Any],
    previous_issues: list[dict[str, str]],
) -> list[Any]:
    system = """你是报告结论抽取器，不是普通写作 Worker。你只能从 accepted_sections 中逐句完整复制原文，允许删除整句和调整完整句子的顺序，不得截断、改写或添加任何句子；known_gaps 也只能完整复制输入中已有的描述。
accepted_gap_sections 与 warning_sections 仅供风险审计，不得从其中抽取事实或正文；证据缺口只能使用 known_gaps。
禁止新增或改写参数、质量指标、数字、因果方向、实验、统计分析、操作建议、控制策略、数据来源或工具调用；禁止声称执行检索、计算、试验或现场验证。只能保留原句已有且属于 accepted_evidence_ids 的 [E编号]。遵守任务描述，只输出抽取后的结论正文，不输出标题、JSON、代码块或解释。"""
    payload = {
        "task": {
            "task_id": task.get("task_id"),
            "task_name": task.get("task_name"),
            "task_description": task.get("task_description"),
            "covers_sections": task.get("covers_sections") or [],
        },
        **context,
        "previous_consistency_issues": previous_issues,
    }
    return [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]


def synthesis(
    state: State, config: Optional[RunnableConfig] = None, **kwargs
) -> dict[str, Any]:
    """Generate one tool-free synthesis result and enforce bounded safety."""

    task = _current_task(state)
    context = build_synthesis_context(state)
    task_id = str(task.get("task_id") or state.get("cursor", 0))
    logger.info(
        "Synthesis context: task=%s accepted_sections=%d accepted_evidence_ids=%d known_gaps=%d",
        task_id,
        len(context["accepted_sections"]),
        len(context["accepted_evidence_ids"]),
        len(context["known_gaps"]),
    )
    findings: list[dict[str, str]] = []
    finding_history: list[list[dict[str, str]]] = []
    content = ""
    attempts = 0
    model_error = ""
    model = None
    if context["accepted_sections"]:
        try:
            model = get_llm(config or {}, json_mode=False)
        except Exception as exc:
            model_error = str(exc)
            logger.exception("Synthesis model initialization failed: task=%s", task_id)
    if model is not None:
        for attempts in range(1, 3):
            try:
                response = model.invoke(
                    _prompt_messages(task, context, findings),
                    config=config or {},
                )
            except Exception as exc:
                model_error = str(exc)
                logger.exception(
                    "Synthesis model invocation failed: task=%s attempt=%d",
                    task_id,
                    attempts,
                )
                break
            content = str(getattr(response, "content", response) or "").strip()
            findings = check_synthesis_consistency(content, context)
            finding_history.append([dict(finding) for finding in findings])
            if content and not findings:
                break
            logger.warning(
                "Synthesis consistency failed: task=%s attempt=%d codes=%s",
                task_id,
                attempts,
                [finding["code"] for finding in findings],
            )
    generation_failed = not content or bool(findings) or bool(model_error)
    if generation_failed:
        content = ""

    cited_ids = {value.upper() for value in _EVIDENCE_ID.findall(content)}
    citations = [
        dict(citation)
        for citation in context["accepted_citations"]
        if _citation_id(citation) in cited_ids
    ]
    sources_used = list(
        dict.fromkeys(
            str(citation.get("file_path") or citation.get("url") or "").strip()
            for citation in citations
            if str(citation.get("file_path") or citation.get("url") or "").strip()
        )
    )
    revision = _revision((state.get("task_revisions") or {}).get(task_id))
    current_result = {
        "task_id": task_id,
        "section_name": str(task.get("task_name") or task_id),
        "text_output": content,
        "status": "COMPLETED" if content and context["accepted_sections"] else "FAILED",
        "tables": [],
        "figures": [],
        "sources_used": sources_used,
        "figures_generated": 0,
        "word_count": count_report_length(content),
        "plan_revision": _revision(state.get("plan_revision")),
        "task_revision": revision,
        "generated_at": datetime.now().isoformat(),
        "execution_time": 0,
        "tool_calls": [],
        "tool_usage_stats": {},
        "knowledge_base_used": False,
        "spider_results_used": False,
        "citations": citations,
        "graph_spec": {},
        "evidence_coverage": {},
        "synthesis_audit": {
            "accepted_task_ids": [
                section["task_id"] for section in context["accepted_sections"]
            ],
            "attempts": attempts,
            "fallback_used": False,
            "model_error": model_error,
            "generation_findings": finding_history,
            "final_consistency_issues": findings,
        },
        "error": (
            None
            if content and context["accepted_sections"]
            else (
                "NO_ACCEPTED_SECTIONS"
                if not context["accepted_sections"]
                else "SYNTHESIS_CONSISTENCY_FAILED"
            )
        ),
    }
    return {
        "current_task": task,
        "current_result": current_result,
        "tool_execution_history": [],
    }

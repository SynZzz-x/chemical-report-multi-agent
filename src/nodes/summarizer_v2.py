"""Deterministically assemble admitted task results into a report."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image as PILImage

from ..evidence.identity import normalize_sections_evidence
from ..evidence.integrity import (
    CitationIntegrityIssue,
    CitationIntegrityValidation,
    project_lossless_used_citations,
    validate_final_citation_integrity,
    validate_pre_remap_citation_integrity,
)
from ..evidence.projection import canonical_source_identity, project_used_citations
from ..evidence.reporting import (
    append_missing_figures,
    format_grouped_evidence_appendix,
    format_knowledge_base_file_table,
)
from ..report_acceptance import (
    BLOCKED,
    DRAFT_WITH_GAPS,
    READY_FOR_FINAL,
    USER_ACCEPTED_GAP,
    USER_ACCEPTED_WARNING,
    derive_report_status,
    eligible_task_ids,
    is_admitted_section_entry,
)
from ..report_outline import (
    classify_outline,
    content_container_paths,
    is_knowledge_base_file_list_section,
    is_reference_section,
    section_container_paths,
    section_markdown_level,
)
from ..state import State
from ..utils import md_to_docx, md_to_pdf
from ..utils.path_manager import get_session_cache_dir


logger = logging.getLogger(__name__)

_ACCEPTED_MISSING_ASSET_CODES = {
    "figure": frozenset(
        {
            "MISSING_FIGURE",
            "MISSING_REQUIRED_FIGURE",
            "MISSING_IMAGE",
            "MISSING_FIGURE_ASSET",
        }
    ),
    "table": frozenset({"MISSING_TABLE", "MISSING_TABLE_ASSET"}),
}


def _user_accepted_missing_asset(
    status_entry: Mapping[str, Any], asset_kind: str
) -> bool:
    """Return whether the user explicitly accepted this asset absence family."""

    if (
        str(status_entry.get("status") or "") != USER_ACCEPTED_WARNING
        or str(status_entry.get("accepted_by") or "") != "user"
    ):
        return False
    accepted_codes = _ACCEPTED_MISSING_ASSET_CODES.get(asset_kind, frozenset())
    return any(
        str(issue.get("code") or "").strip().upper() in accepted_codes
        for issue in status_entry.get("issues") or []
        if isinstance(issue, Mapping)
    )


_DANGLING_FIGURE_SENTENCE = re.compile(
    r"[^。！？!?\n]*(?:(?:见|参见)\s*图\s*\d+|如\s*图\s*\d+\s*所示)"
    r"[^。！？!?\n]*[。！？!?]?"
)
_ORPHAN_FIGURE_CAPTION = re.compile(
    r"^\s*(?:图\s*\d+|Figure\s*\d+)\s*[：:、.．-]?\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _degrade_accepted_missing_figure(
    text: str, *, accepted_by_system: bool = False
) -> str:
    """Remove references to an accepted-but-absent figure in this section only."""

    cleaned = _DANGLING_FIGURE_SENTENCE.sub("", str(text or ""))
    cleaned = _ORPHAN_FIGURE_CAPTION.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    note = (
        "> 图形缺口：系统已记录可选图形资产缺失，相关图号引用已移除。"
        if accepted_by_system
        else "> 图形缺口：用户已接受本节缺少正式图形资产，相关图号引用已移除。"
    )
    return f"{cleaned}\n\n{note}" if cleaned else note


def _active_asset_degradation(
    state: State, task_id: str, asset_kind: str
) -> bool:
    accepted_codes = _ACCEPTED_MISSING_ASSET_CODES.get(asset_kind, frozenset())
    return any(
        isinstance(issue, Mapping)
        and str(issue.get("task_id") or "") == task_id
        and str(issue.get("status") or "") == "active"
        and str(issue.get("subtype") or "").upper() in accepted_codes
        for issue in state.get("degraded_issue_registry") or []
    )


def find_title(state: State) -> str:
    for message in state.get("messages", []) or []:
        try:
            content = json.loads(str(message.content))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if content.get("from") == "Intake" and content.get("to") == "Planner":
            return str(content.get("title") or "自动生成报告")
    return "自动生成报告"


def _intake_sections(state: State) -> list[str]:
    for message in reversed(state.get("messages", []) or []):
        raw_content = (
            message.get("content")
            if isinstance(message, Mapping)
            else getattr(message, "content", None)
        )
        try:
            content = (
                json.loads(raw_content)
                if isinstance(raw_content, str)
                else raw_content
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(content, Mapping):
            continue
        if content.get("from") != "Intake" or content.get("to") != "Planner":
            continue
        sections = content.get("sections")
        if isinstance(sections, list):
            return [str(section).strip() for section in sections if str(section).strip()]
    return []


def _normalize_title(value: str) -> str:
    text = re.sub(r"^\s*#+\s*", "", str(value or "").strip())
    text = re.sub(r"^\s*(?:第?[一二三四五六七八九十百千万\d]+[章节、.．:]|[（(]?[一二三四五六七八九十\d]+[）).、．])\s*", "", text)
    text = re.sub(r"[\s\-—_：:，,。.!！?？（）()\[\]【】]", "", text)
    return text.casefold()


def _titles_match(heading: str, task_name: str) -> bool:
    left = _normalize_title(heading)
    right = _normalize_title(task_name)
    if not left or not right:
        return False
    return left == right


def _strip_duplicate_leading_heading(text: str, task_name: str) -> str:
    """Remove only a matching first Markdown heading and preserve subheadings."""

    lines = str(text or "").splitlines()
    first_content = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_content is None:
        return ""
    match = re.match(r"^\s{0,3}#{1,2}\s+(.+?)\s*$", lines[first_content])
    if not match or not _titles_match(match.group(1), task_name):
        return str(text or "").strip()
    del lines[first_content]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _starts_with_matching_heading(text: str, title: str) -> bool:
    first = next(
        (line.strip() for line in str(text or "").splitlines() if line.strip()),
        "",
    )
    match = re.match(r"^#{1,6}\s+(.+?)\s*$", first)
    return bool(match and _titles_match(match.group(1), title))


def _escape_table_cell(value: Any) -> str:
    return str(value or "").replace("|", "｜").replace("\n", " ").strip()


def _is_renderable_local_image(path: str) -> bool:
    expected_formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
    expected_format = expected_formats.get(os.path.splitext(path)[1].lower())
    if expected_format is None:
        return False
    try:
        with PILImage.open(path) as image:
            if image.format != expected_format:
                return False
            image.verify()
        with PILImage.open(path) as image:
            if image.format != expected_format:
                return False
            image.load()
        return True
    except Exception:
        return False


def _table_markdown(table: Mapping[str, Any]) -> str:
    data = table.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)) or not data:
        headers = table.get("headers") or table.get("columns")
        rows = table.get("rows")
        if not isinstance(headers, Sequence) or isinstance(headers, (str, bytes)):
            return ""
        data = [list(headers), *(rows or [])]
    normalized_rows = [
        list(row)
        for row in data
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
    ]
    if not normalized_rows or not normalized_rows[0]:
        return ""
    width = len(normalized_rows[0])
    rows = [(row + [""] * width)[:width] for row in normalized_rows]
    title = str(table.get("title") or table.get("description") or "结构化表格").strip()
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(_escape_table_cell(cell) for cell in rows[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |"
        for row in rows[1:]
    )
    return "\n".join(lines)


def _append_missing_tables(markdown: str, tables: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for table in tables:
        if str(table.get("source") or "").strip() == "worker_markdown":
            continue
        block = _table_markdown(table)
        if block and block not in markdown:
            blocks.append(block)
    if not blocks:
        return markdown
    return f"{markdown.rstrip()}\n\n" + "\n\n".join(blocks)


def _ordered_sections(
    state: State, admitted_task_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = {
        str(task.get("task_id")): task
        for task in state.get("tasks", []) or []
        if isinstance(task, dict) and task.get("task_id") is not None
    }
    results = {
        str(result.get("task_id")): result
        for result in state.get("results", []) or []
        if isinstance(result, dict) and result.get("task_id") is not None
    }
    statuses = state.get("section_status") or {}
    container_path_by_section = content_container_paths(_intake_sections(state))
    sections: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for task_id in admitted_task_ids:
        task = tasks.get(task_id) or {}
        result = results.get(task_id)
        if result is None:
            missing.append(
                {
                    "task_id": task_id,
                    "task_name": task.get("task_name") or task_id,
                    "status": BLOCKED,
                    "issues": [
                        {
                            "code": "MISSING_RESULT",
                            "description": "章节已标记为可交付，但 results 中没有对应结果。",
                        }
                    ],
                }
            )
            continue
        status_entry = statuses.get(task_id) or {}
        expected_plan_revision = int(status_entry.get("plan_revision", 0) or 0)
        expected_task_revision = int(status_entry.get("task_revision", 0) or 0)
        actual_plan_revision = int(result.get("plan_revision", 0) or 0)
        actual_task_revision = int(result.get("task_revision", 0) or 0)
        if (
            expected_plan_revision != actual_plan_revision
            or expected_task_revision != actual_task_revision
        ):
            missing.append(
                {
                    "task_id": task_id,
                    "task_name": task.get("task_name") or task_id,
                    "status": BLOCKED,
                    "issues": [
                        {
                            "code": "REVISION_MISMATCH",
                            "description": (
                                "章节结果版本与验收状态不一致："
                                f"expected=p{expected_plan_revision}/t{expected_task_revision}, "
                                f"actual=p{actual_plan_revision}/t{actual_task_revision}。"
                            ),
                        }
                    ],
                }
            )
            continue
        raw_tables = list(result.get("tables") or [])
        valid_tables: list[dict[str, Any]] = []
        invalid_tables = 0
        for raw_table in raw_tables:
            if not isinstance(raw_table, Mapping):
                invalid_tables += 1
                continue
            table = dict(raw_table)
            if not _table_markdown(table):
                invalid_tables += 1
                continue
            valid_tables.append(table)
        missing_required_table = bool(task.get("generate_table") and not valid_tables)
        accepted_missing_table = (
            missing_required_table
            and not invalid_tables
            and (
                _user_accepted_missing_asset(status_entry, "table")
                or _active_asset_degradation(state, task_id, "table")
            )
        )
        if accepted_missing_table:
            logger.info(
                "Summarizer accepted asset degradation: task=%s asset=table",
                task_id,
            )
        elif invalid_tables or missing_required_table:
            code = "INVALID_TABLE_ASSET" if invalid_tables else "MISSING_TABLE_ASSET"
            missing.append(
                {
                    "task_id": task_id,
                    "task_name": task.get("task_name") or task_id,
                    "status": BLOCKED,
                    "issues": [
                        {
                            "code": code,
                            "description": "结构化表格资产不存在或无法确定性物化。",
                        }
                    ],
                }
            )
            continue
        figures: list[dict[str, Any]] = []
        raw_figures = list(result.get("figures") or [])
        if not raw_figures:
            raw_figures.extend(
                {"path": path, "description": f"图像：{os.path.basename(str(path))}"}
                for path in result.get("outputs") or []
                if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
            )
        missing_figure_paths: list[str] = []
        invalid_figure_content: list[str] = []
        invalid_figures = 0
        for raw_figure in raw_figures:
            if not isinstance(raw_figure, Mapping):
                invalid_figures += 1
                continue
            figure = dict(raw_figure)
            path = str(figure.get("path") or "").strip()
            if not path or path.startswith(("http://", "https://")):
                invalid_figures += 1
                continue
            if not os.path.isfile(path):
                missing_figure_paths.append(path)
                continue
            if not _is_renderable_local_image(path):
                invalid_figure_content.append(path)
                continue
            evidence_ids = [
                str(value).strip()
                for value in figure.get("evidence_ids") or []
                if str(value).strip()
            ]
            if evidence_ids:
                markers = "、".join(f"[{value}]" for value in evidence_ids)
                description = str(figure.get("description") or "图像").strip()
                if markers not in description:
                    figure["description"] = f"{description}（关系证据：{markers}）"
            figures.append(figure)
        missing_required_figure = bool(task.get("generate_figure") and not figures)
        accepted_missing_figure = (
            missing_required_figure
            and not invalid_figures
            and not missing_figure_paths
            and not invalid_figure_content
            and (
                _user_accepted_missing_asset(status_entry, "figure")
                or _active_asset_degradation(state, task_id, "figure")
            )
        )
        if accepted_missing_figure:
            logger.info(
                "Summarizer accepted asset degradation: task=%s asset=figure",
                task_id,
            )
        elif (
            invalid_figures
            or missing_figure_paths
            or invalid_figure_content
            or missing_required_figure
        ):
            if missing_figure_paths:
                code = "MISSING_FIGURE_FILE"
                detail = "正式图形资产不存在：" + "、".join(missing_figure_paths)
            elif invalid_figure_content:
                code = "INVALID_FIGURE_CONTENT"
                detail = "图形资产不是当前渲染器可解码的 PNG/JPEG：" + "、".join(
                    invalid_figure_content
                )
            elif invalid_figures:
                code = "INVALID_FIGURE_ASSET"
                detail = "图形资产缺少本地可渲染路径或结构非法。"
            else:
                code = "MISSING_FIGURE_ASSET"
                detail = "任务要求正式图形资产，但结果中没有可渲染图形。"
            missing.append(
                {
                    "task_id": task_id,
                    "task_name": task.get("task_name") or task_id,
                    "status": BLOCKED,
                    "issues": [
                        {
                            "code": code,
                            "description": detail,
                        }
                    ],
                }
            )
            continue
        section_text = str(result.get("text_output") or result.get("content") or "")
        if accepted_missing_figure:
            section_text = _degrade_accepted_missing_figure(
                section_text,
                accepted_by_system=_active_asset_degradation(
                    state, task_id, "figure"
                ),
            )
        sections.append(
            {
                "task_id": task_id,
                "title": str(task.get("task_name") or task_id),
                "text": section_text,
                "tables": [
                    dict(table) for table in valid_tables
                ],
                "figures": figures,
                "citations": list(result.get("citations") or []),
                "covers_sections": list(task.get("covers_sections") or []),
                "container_path": next(
                    (
                        list(container_path_by_section[covered])
                        for covered in task.get("covers_sections") or []
                        if covered in container_path_by_section
                    ),
                    [],
                ),
            }
        )
    return sections, missing


def _blocking_sections(state: State) -> list[dict[str, Any]]:
    statuses = state.get("section_status") or {}
    blocking: list[dict[str, Any]] = []
    for task in state.get("tasks", []) or []:
        if not isinstance(task, dict) or task.get("task_id") is None:
            continue
        task_id = str(task["task_id"])
        entry = statuses.get(task_id)
        if is_admitted_section_entry(entry):
            continue
        if isinstance(entry, Mapping):
            status = str(entry.get("status") or BLOCKED)
            issues = list(entry.get("issues") or [])
        else:
            status = BLOCKED
            issues = [
                {
                    "code": "MISSING_SECTION_STATUS",
                    "description": "章节没有可审计的验收状态。",
                }
            ]
        blocking.append(
            {
                "task_id": task_id,
                "task_name": task.get("task_name") or task_id,
                "status": status,
                "issues": issues,
            }
        )
    return blocking


def _blocked_update(blocking_sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    logger.warning(
        "Summarizer admission blocked: blocking_sections=%s",
        list(blocking_sections),
    )
    final_result = {
        "summary": "报告未满足正式交付准入条件。",
        "evaluation": "请先处理阻塞章节，或明确接受相应缺口后生成带风险草稿。",
        "report_status": BLOCKED,
        "blocking_sections": list(blocking_sections),
        "attachments": [],
        "path": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return {
        "messages": [AIMessage(content=final_result["evaluation"])],
        "final_result": final_result,
        "report_status": BLOCKED,
    }


def _citation_integrity_blocked_update(
    validation: CitationIntegrityValidation,
) -> dict[str, Any]:
    """Adapt pure citation-validation failures to the existing blocked result."""

    detail = "；".join(
        f"{issue.code}: {issue.description}" for issue in validation.issues
    )
    return _blocked_update(
        [
            {
                "task_id": "REPORT",
                "task_name": "报告引用完整性",
                "status": BLOCKED,
                "issues": [
                    {
                        "code": "FINAL_CITATION_INTEGRITY",
                        "description": detail or "最终引用完整性校验失败。",
                    }
                ],
            }
        ]
    )


def _draft_warning(state: State) -> str:
    statuses = state.get("section_status") or {}
    task_names = {
        str(task.get("task_id")): str(task.get("task_name") or task.get("task_id"))
        for task in state.get("tasks", []) or []
        if isinstance(task, dict) and task.get("task_id") is not None
    }
    lines = [
        "> **未完成草稿：已接受的证据缺口或内容风险**",
        ">",
        "> 以下章节由用户明确接受缺口后纳入，本文件不得作为无保留正式报告使用：",
    ]
    for task_id, entry in statuses.items():
        if not isinstance(entry, Mapping) or entry.get("status") not in {
            USER_ACCEPTED_GAP,
            USER_ACCEPTED_WARNING,
        }:
            continue
        descriptions = [
            str(issue.get("description") or issue.get("code") or "未说明的缺口")
            for issue in entry.get("issues") or []
            if isinstance(issue, Mapping)
        ]
        detail = "；".join(descriptions) or "用户接受该章节当前缺口"
        lines.append(f"> - {task_names.get(str(task_id), str(task_id))}：{detail}")
    blocks = ["\n".join(lines)] if len(lines) > 3 else []
    active_degradations = [
        issue
        for issue in state.get("degraded_issue_registry") or []
        if isinstance(issue, Mapping) and issue.get("status") == "active"
    ]
    if active_degradations:
        degradation_lines = [
            "> **系统记录的交付限制**",
            ">",
            "> 以下限制已按软要求降级处理，并未被描述为用户批准或问题已解决：",
        ]
        for issue in active_degradations:
            task_id = str(issue.get("task_id") or "")
            subtype = str(issue.get("subtype") or "DEGRADABLE_QUALITY")
            degradation_lines.append(
                f"> - {task_names.get(task_id, task_id)}：{subtype}"
            )
        blocks.append("\n".join(degradation_lines))
    return "\n\n".join(blocks)


def _deduplicate_citations(sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in sections:
        used_citations = project_used_citations(
            str(section.get("text") or section.get("content") or ""),
            section.get("citations") or [],
        )
        for citation in used_citations:
            if not isinstance(citation, Mapping):
                continue
            evidence_id = str(citation.get("evidence_id") or "").strip()
            task_id = str(section.get("task_id") or "").strip()
            key = (
                f"{task_id}:{evidence_id}"
                if evidence_id
                else f"{task_id}:" + json.dumps(
                    dict(citation), ensure_ascii=False, sort_keys=True
                )
            )
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    **dict(citation),
                    "section_title": str(section.get("title") or task_id),
                }
            )
    return citations


def _assemble_markdown(
    state: State,
    sections: Sequence[Mapping[str, Any]],
    report_status: str,
    *,
    body_spans: list[tuple[int, int]] | None = None,
) -> str:
    blocks = [f"# {find_title(state)}"]
    body_block_indices: set[int] = set()
    if report_status == DRAFT_WITH_GAPS:
        blocks.append(_draft_warning(state))
    outline = classify_outline(_intake_sections(state))
    outline_ordinals = {item.raw: item.ordinal for item in outline}
    reference_item = next(
        (
            item
            for item in outline
            if item.kind == "system_generated" and is_reference_section(item.title)
        ),
        None,
    )
    reference_citations = _deduplicate_citations(sections)
    reference_heading_level = (
        section_markdown_level(reference_item.raw) if reference_item else 2
    )
    if reference_item and is_knowledge_base_file_list_section(reference_item.title):
        file_list_markdown = format_knowledge_base_file_table(
            reference_citations,
            heading_level=reference_heading_level,
            heading_title=reference_item.raw,
        )
        evidence_index_markdown = format_grouped_evidence_appendix(
            reference_citations,
            heading_level=min(reference_heading_level + 1, 6),
            heading_title="证据索引",
        )
        reference_markdown = "\n\n".join(
            block for block in (file_list_markdown, evidence_index_markdown) if block
        )
    else:
        reference_markdown = format_grouped_evidence_appendix(
            reference_citations,
            heading_level=reference_heading_level,
            heading_title=(reference_item.raw if reference_item else "证据来源"),
        )
    all_container_paths = section_container_paths(_intake_sections(state))
    reference_container_path = tuple(
        all_container_paths.get(reference_item.raw, ()) if reference_item else ()
    )
    reference_inserted = False
    active_container_path: tuple[str, ...] = ()

    def append_container_headings(container_path: tuple[str, ...]) -> None:
        nonlocal active_container_path
        common_length = 0
        for current, previous in zip(container_path, active_container_path):
            if current != previous:
                break
            common_length += 1
        for depth, container_title in enumerate(
            container_path[common_length:],
            start=common_length,
        ):
            heading_level = min(depth + 2, 6)
            blocks.append(f"{'#' * heading_level} {container_title}")
        active_container_path = container_path

    for section in sections:
        covered_ordinals = [
            outline_ordinals[covered]
            for covered in section.get("covers_sections") or []
            if covered in outline_ordinals
        ]
        if (
            reference_markdown
            and reference_item is not None
            and not reference_inserted
            and covered_ordinals
            and reference_item.ordinal < min(covered_ordinals)
        ):
            append_container_headings(reference_container_path)
            blocks.append(reference_markdown)
            reference_inserted = True
        body = _strip_duplicate_leading_heading(
            str(section.get("text") or ""), str(section.get("title") or "章节")
        )
        container_path = tuple(section.get("container_path") or [])
        if container_path:
            append_container_headings(container_path)
            section_markdown = body.rstrip()
        elif section.get("covers_sections"):
            active_container_path = ()
            covered_sections = list(section.get("covers_sections") or [])
            if len(covered_sections) == 1 and not _starts_with_matching_heading(
                body, covered_sections[0]
            ):
                heading_level = section_markdown_level(covered_sections[0])
                section_markdown = (
                    f"{'#' * heading_level} {covered_sections[0]}\n\n{body}"
                ).rstrip()
            else:
                section_markdown = body.rstrip()
        else:
            active_container_path = ()
            section_markdown = f"## {section.get('title') or '章节'}\n\n{body}".rstrip()
        section_markdown = _append_missing_tables(
            section_markdown, section.get("tables") or []
        )
        section_markdown = append_missing_figures(
            section_markdown, section.get("figures") or []
        )
        body_block_indices.add(len(blocks))
        blocks.append(section_markdown)
    if reference_markdown and not reference_inserted:
        append_container_headings(reference_container_path)
        blocks.append(reference_markdown)
    final_markdown = "\n\n".join(block for block in blocks if block).rstrip() + "\n"
    if body_spans is not None:
        offset = 0
        for index, block in enumerate(blocks):
            if not block:
                continue
            end = offset + len(block)
            if index in body_block_indices:
                body_spans.append((offset, min(end, len(final_markdown))))
            offset = end + 2  # The exact separator used by the single join above.
    return final_markdown


def summarizer(state: State, config: RunnableConfig, **kwargs) -> dict[str, Any]:
    """Generate only report artifacts admitted by deterministic acceptance state."""

    tasks = state.get("tasks") or []
    statuses = state.get("section_status") or {}
    report_status = derive_report_status(tasks, statuses)
    logger.info(
        "Summarizer admission: report_status=%s section_status=%s",
        report_status,
        {
            str(task_id): str(entry.get("status") or "")
            for task_id, entry in statuses.items()
            if isinstance(entry, Mapping)
        },
    )
    if report_status == BLOCKED:
        return _blocked_update(_blocking_sections(state))

    admitted = eligible_task_ids(tasks, statuses, report_status)
    sections, missing = _ordered_sections(state, admitted)
    if missing:
        return _blocked_update(missing)

    preflight = validate_pre_remap_citation_integrity(sections)
    if not preflight.is_valid:
        return _citation_integrity_blocked_update(preflight)

    sections, evidence_display_map = normalize_sections_evidence(sections)
    body_spans: list[tuple[int, int]] = []
    try:
        final_markdown = _assemble_markdown(
            state, sections, report_status, body_spans=body_spans
        )
    except ValueError as exc:
        if str(exc) != "FINAL_DISPLAY_IDENTITY_CONFLICT":
            raise
        return _citation_integrity_blocked_update(
            CitationIntegrityValidation(
                issues=(
                    CitationIntegrityIssue(
                        code="FINAL_DISPLAY_IDENTITY_CONFLICT",
                        description="同一最终证据编号对应多个不同证据来源或定位。",
                    ),
                )
            )
        )
    final_citations = project_lossless_used_citations(sections)
    final_validation = validate_final_citation_integrity(
        sections, final_markdown, final_citations, body_spans=body_spans
    )
    if not final_validation.is_valid:
        return _citation_integrity_blocked_update(final_validation)

    report_sources: list[str] = []
    seen_sources: set[str] = set()
    for citation in _deduplicate_citations(sections):
        source = canonical_source_identity(citation)
        if source and source.casefold() not in seen_sources:
            seen_sources.add(source.casefold())
            report_sources.append(source)
    report_dir = os.path.join(get_session_cache_dir(state, config), "report")
    os.makedirs(report_dir, exist_ok=True)
    stem = "report_draft_with_gaps" if report_status == DRAFT_WITH_GAPS else "report"
    md_path = os.path.abspath(os.path.join(report_dir, f"{stem}.md"))
    pdf_path = os.path.abspath(os.path.join(report_dir, f"{stem}.pdf"))
    docx_path = os.path.abspath(os.path.join(report_dir, f"{stem}.docx"))

    attachments: list[str] = []
    artifact_errors: list[dict[str, str]] = []
    try:
        with open(md_path, "w", encoding="utf-8") as report_file:
            report_file.write(final_markdown)
        attachments.append(md_path)
    except OSError as exc:
        logger.error("Failed to write Markdown report: %s", exc)
        artifact_errors.append({"format": "markdown", "error": str(exc)})

    try:
        math_img_dir = os.path.join(get_session_cache_dir(state, config), "math_imgs")
        md_to_pdf.md_to_pdf(final_markdown, pdf_path, math_img_dir=math_img_dir)
        if os.path.exists(pdf_path):
            attachments.append(pdf_path)
    except Exception as exc:
        logger.error("Failed to generate PDF: %s", exc)
        artifact_errors.append({"format": "pdf", "error": str(exc)})

    try:
        md_to_docx.md_to_docx(final_markdown, docx_path)
        if os.path.exists(docx_path):
            attachments.append(docx_path)
    except Exception as exc:
        logger.error("Failed to generate DOCX: %s", exc)
        artifact_errors.append({"format": "docx", "error": str(exc)})

    delivery_status = (
        "COMPLETE" if len(attachments) == 3 else "PARTIAL" if attachments else "FAILED"
    )
    if delivery_status == "COMPLETE":
        summary = (
            "已生成带证据缺口说明的草稿。"
            if report_status == DRAFT_WITH_GAPS
            else "已生成通过验收的正式报告。"
        )
    elif delivery_status == "PARTIAL":
        summary = "报告内容已完成验收，但仅成功生成部分交付文件。"
    else:
        summary = "报告内容已完成验收，但交付文件生成失败。"
    preferred_path = next(
        (
            path
            for path in (docx_path, pdf_path, md_path)
            if path in attachments
        ),
        None,
    )
    final_result = {
        "summary": summary,
        "evaluation": summary,
        "report_status": report_status,
        "delivery_status": delivery_status,
        "artifact_errors": artifact_errors,
        "evidence_display_map": evidence_display_map,
        "report_sources": report_sources,
        "blocking_sections": [],
        "attachments": attachments,
        "path": preferred_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return {
        "messages": [AIMessage(content=summary)],
        "final_result": final_result,
        "report_status": report_status,
    }

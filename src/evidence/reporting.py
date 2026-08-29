from __future__ import annotations

from collections.abc import Mapping, Sequence
import ntpath
import os
import re
from typing import Any

from .projection import canonical_source_identity
from .text_projection import presentation_evidence_excerpt


_NATURAL_EVIDENCE_ID = re.compile(r"^E(\d+)$", re.IGNORECASE)


def _escape_cell(value: Any) -> str:
    return str(value or "").replace("|", "｜").replace("\n", " ").strip()


def _path_like(value: str) -> bool:
    text = str(value or "").strip()
    return "/" in text or "\\" in text or (len(text) > 1 and text[1] == ":")


def _safe_source_name(citation: Mapping[str, Any]) -> str:
    if str(citation.get("source_type") or "").strip().casefold() == "web":
        title = str(citation.get("title") or "").strip()
        if title:
            return title
    return canonical_source_identity(citation) or "未命名知识库文件"


def _safe_group_label(citation: Mapping[str, Any]) -> str:
    for field in ("title", "file_name"):
        candidate = str(citation.get(field) or "").strip()
        if not candidate:
            continue
        if _path_like(candidate):
            candidate = ntpath.basename(candidate.rstrip("/\\"))
        if candidate:
            return candidate
    return canonical_source_identity(citation) or "未命名知识库文件"


def _internal_reference(value: str) -> bool:
    lowered = value.casefold().replace("\\", "/")
    return bool(
        value.startswith(("/", "\\"))
        or (len(value) > 1 and value[1] == ":")
        or "chunk_id=" in lowered
        or re.search(r"(?:^|[/_-])rag[_-]?\d", lowered)
    )


def _safe_locator(citation: Mapping[str, Any]) -> str:
    locator = str(citation.get("locator") or citation.get("url") or "").strip()
    if not locator or _internal_reference(locator):
        return "—"
    return locator


def _safe_section_title(citation: Mapping[str, Any]) -> str:
    section_title = str(citation.get("section_title") or "").strip()
    if not section_title or _internal_reference(section_title):
        return "—"
    return section_title


def _evidence_sort_key(citation: Mapping[str, Any]) -> tuple[int, int, str]:
    evidence_id = str(citation.get("evidence_id") or "").strip()
    match = _NATURAL_EVIDENCE_ID.fullmatch(evidence_id)
    if match:
        return (0, int(match.group(1)), evidence_id.casefold())
    return (1, 0, evidence_id.casefold())


def format_grouped_evidence_appendix(
    citations: Sequence[Mapping[str, Any]],
    *,
    heading_level: int = 3,
    heading_title: str = "证据来源",
) -> str:
    """Render a stable presentation-only evidence projection by source."""

    if not citations:
        return ""
    level = max(1, min(int(heading_level), 6))
    groups: dict[str, dict[str, Any]] = {}
    for citation in citations:
        identity = canonical_source_identity(citation)
        key = identity.casefold()
        group = groups.setdefault(
            key,
            {
                "label": _safe_group_label(citation),
                "citations": [],
            },
        )
        group["citations"].append(citation)

    lines = [
        f"{'#' * level} {str(heading_title).strip() or '证据来源'}",
    ]
    source_level = min(level + 1, 6)
    for group in groups.values():
        lines.extend(
            [
                "",
                f"{'#' * source_level} {_escape_cell(group['label'])}",
                "",
                "| 证据编号 | 定位 | 支撑章节 | 摘要 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for citation in sorted(group["citations"], key=_evidence_sort_key):
            lines.append(
                "| [{evidence_id}] | {locator} | {section_title} | {summary} |".format(
                    evidence_id=_escape_cell(citation.get("evidence_id")),
                    locator=_escape_cell(_safe_locator(citation)),
                    section_title=_escape_cell(_safe_section_title(citation)),
                    summary=_escape_cell(presentation_evidence_excerpt(citation)) or "—",
                )
            )
    return "\n".join(lines)


def append_missing_figures(markdown: str, figures: Sequence[Mapping[str, Any]]) -> str:
    """Ensure report assets are embedded even if the formatting LLM omits one."""
    blocks: list[str] = []
    for figure in figures:
        path = str(figure.get("path") or "").strip()
        if not path or path in markdown:
            continue
        description = str(figure.get("description") or "图像").strip()
        alt = str(figure.get("alt") or description.split("（关系证据", 1)[0] or "图像")
        alt = alt.replace("[", "").replace("]", "").replace("\n", " ")
        blocks.append(f"![{alt}]({path})\n<description>{description}</description>")
    if not blocks:
        return markdown
    return f"{markdown.rstrip()}\n\n" + "\n\n".join(blocks)


def format_evidence_table(
    citations: Sequence[Mapping[str, Any]],
    *,
    heading_level: int = 3,
    include_section: bool = False,
    heading_title: str = "证据来源",
) -> str:
    if not citations:
        return ""
    lines = [
        f"{'#' * max(1, min(int(heading_level), 6))} {str(heading_title).strip() or '证据来源'}",
        "",
    ]
    if include_section:
        lines.extend(
            [
                "| 证据编号 | 来源与支撑章节 | 定位 | 摘要 |",
                "| --- | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| 证据编号 | 来源类型 | 标题 | 定位或链接 | 支持内容 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
    for citation in citations:
        evidence_id = _escape_cell(citation.get("evidence_id"))
        locator = citation.get("locator") or citation.get("url")
        if not locator:
            locator = citation.get("file_name") or os.path.basename(
                str(citation.get("file_path") or "")
            )
        values = {
            "evidence_id": evidence_id,
            "source_type": _escape_cell(citation.get("source_type")),
            "title": _escape_cell(_safe_source_name(citation)),
            "locator": _escape_cell(locator),
            "supporting_text": _escape_cell(citation.get("supporting_text"))[:240],
            "section_title": _escape_cell(citation.get("section_title")),
        }
        if include_section:
            source = " / ".join(
                value for value in (values["source_type"], values["title"]) if value
            )
            if values["section_title"]:
                source = f"{source or '未命名来源'}（支撑章节：{values['section_title']}）"
            lines.append(
                "| [{evidence_id}] | {source} | {locator} | {supporting_text} |".format(
                    source=source,
                    **values,
                )
            )
        else:
            lines.append(
                "| [{evidence_id}] | {source_type} | {title} | {locator} | {supporting_text} |".format(
                    **values,
                )
            )
    return "\n".join(lines)


def format_knowledge_base_file_table(
    citations: Sequence[Mapping[str, Any]],
    *,
    heading_level: int = 3,
    heading_title: str = "知识库文件清单",
) -> str:
    """Project accepted RAG citations into one deterministic file-level table."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for citation in citations:
        source_type = str(citation.get("source_type") or "").strip()
        if source_type.casefold() != "rag":
            continue
        file_path = str(citation.get("file_path") or "").strip()
        file_name = _safe_source_name(citation)
        key = (source_type.casefold(), file_name.casefold())
        entry = grouped.setdefault(
            key,
            {
                "file_name": file_name,
                "source_type": source_type or "rag",
                "sections": [],
                "count": 0,
            },
        )
        section = str(citation.get("section_title") or "").strip()
        if section and section not in entry["sections"]:
            entry["sections"].append(section)
        entry["count"] += 1

    lines = [
        f"{'#' * max(1, min(int(heading_level), 6))} {str(heading_title).strip() or '知识库文件清单'}",
        "",
        "| 文件名称 | 来源类型 | 支撑章节 | 证据条数 |",
        "| --- | --- | --- | --- |",
    ]
    if not grouped:
        lines.append("| 未记录可追溯的知识库文件 | rag | — | 0 |")
    else:
        for entry in grouped.values():
            sections = "、".join(entry["sections"]) or "未绑定章节"
            lines.append(
                "| {file_name} | {source_type} | {sections} | {count} |".format(
                    file_name=_escape_cell(entry["file_name"]),
                    source_type=_escape_cell(entry["source_type"]),
                    sections=_escape_cell(sections),
                    count=int(entry["count"]),
                )
            )
    return "\n".join(lines)

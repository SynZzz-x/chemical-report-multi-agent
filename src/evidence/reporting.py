from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any

from .projection import canonical_source_identity


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

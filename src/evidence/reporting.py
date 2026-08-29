from __future__ import annotations

from collections.abc import Mapping, Sequence
import ipaddress
import ntpath
import os
import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from .projection import canonical_source_identity
from .text_projection import normalize_evidence_text, presentation_evidence_excerpt


_NATURAL_EVIDENCE_ID = re.compile(r"^E(\d+)$", re.IGNORECASE)
_URL_REFERENCE = re.compile(r"(?:https?|file)://[^\s|，。；、]+", re.IGNORECASE)
_PATH_REFERENCE = re.compile(
    r"(?:"
    r"[A-Za-z]:[\\/][^\s|，。；、]+"
    r"|(?<![\w/])/(?![\s/])[^\s|，。；、/]+(?:/[^\s|，。；、/]+)+"
    r"|(?<![\w.-])(?:cache|users?|conversations?|jobs?|chunks?)[\\/][^\s|，。；、]+"
    r")",
    re.IGNORECASE,
)
_IDENTIFIER_REFERENCE = re.compile(
    r"\b(?:user|conversation|job|chunk|cache)[_-]?id\s*[:=]\s*[^\s|，。；、;&]+",
    re.IGNORECASE,
)
_RAG_REFERENCE = re.compile(r"\brag_[\w.-]+", re.IGNORECASE)
_GENERATED_LABEL = re.compile(
    r"^(?:"
    r"(?:user|conversation|job|chunk|cache)(?:_id)?[-_=][\w.-]+"
    r"|\d{6,}(?:\.[A-Za-z0-9]{1,12})?"
    r"|[0-9a-f]{16,}(?:\.[A-Za-z0-9]{1,12})?"
    r"|[0-9a-f]{8}-[0-9a-f-]{13,}(?:\.[A-Za-z0-9]{1,12})?"
    r"|(?:u|c|j)[-_](?:\d{4,}|[0-9a-f]{8,})(?:\.[A-Za-z0-9]{1,12})?"
    r")$",
    re.IGNORECASE,
)
_SAFE_FILE_NAME = re.compile(r"[^./\\]+\.[A-Za-z0-9]{1,12}$")
_INTERNAL_URL_HOST = re.compile(r"(?:^|[.-])(?:internal|localhost|private)(?:[.-]|$)")
_SENSITIVE_QUERY_KEYS = {
    "user",
    "userid",
    "conversation",
    "conversationid",
    "job",
    "jobid",
    "chunk",
    "chunkid",
    "cache",
    "cacheid",
}
_INTERNAL_PATH_PARTS = {
    "cache",
    "user",
    "users",
    "conversation",
    "conversations",
    "job",
    "jobs",
    "chunk",
    "chunks",
}
_REDACTION = "[内部引用已隐藏]"


def _escape_cell(value: Any) -> str:
    return str(value or "").replace("|", "｜").replace("\n", " ").strip()


def _path_like(value: str) -> bool:
    text = str(value or "").strip()
    return "/" in text or "\\" in text or (len(text) > 1 and text[1] == ":")


def _safe_source_name(citation: Mapping[str, Any]) -> str:
    file_path = str(citation.get("file_path") or "").strip()
    file_name = ntpath.basename(file_path.rstrip("/\\"))
    if (
        file_name
        and _SAFE_FILE_NAME.fullmatch(file_name)
        and not _GENERATED_LABEL.fullmatch(file_name)
        and _redact_internal_references(file_name) == file_name
    ):
        return file_name
    return _safe_group_label(citation)


def _safe_group_label(citation: Mapping[str, Any]) -> str:
    for field in ("title", "file_name", "file_path"):
        candidate = str(citation.get(field) or "").strip()
        if not candidate:
            continue
        parsed = urlsplit(candidate)
        if parsed.scheme or parsed.netloc or "?" in candidate or "#" in candidate:
            continue
        path_like = _path_like(candidate)
        if path_like:
            candidate = ntpath.basename(candidate.rstrip("/\\"))
        if (
            candidate
            and (not path_like or _SAFE_FILE_NAME.fullmatch(candidate))
            and not _GENERATED_LABEL.fullmatch(candidate)
            and _redact_internal_references(candidate) == candidate
        ):
            return candidate
    source_type = str(citation.get("source_type") or "").strip().casefold()
    return {
        "rag": "知识库文档",
        "web": "网页来源",
        "file": "文件来源",
        "local": "文件来源",
        "tool": "工具来源",
    }.get(source_type, "证据来源")


def _normalized_identifier_key(value: str) -> str:
    return re.sub(r"[_-]", "", str(value or "").casefold())


def _literal_internal_host(hostname: str) -> bool:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
    )


def _internal_payload(value: str) -> bool:
    decoded = unquote(str(value or ""))
    if (
        _PATH_REFERENCE.search(decoded)
        or _IDENTIFIER_REFERENCE.search(decoded)
        or _RAG_REFERENCE.search(decoded)
    ):
        return True
    parsed = urlsplit(decoded)
    path_parts = {
        part.casefold() for part in unquote(parsed.path).split("/") if part
    }
    return bool(
        parsed.scheme.casefold() == "file"
        or _INTERNAL_URL_HOST.search(parsed.hostname or "")
        or _literal_internal_host(parsed.hostname or "")
        or path_parts.intersection(_INTERNAL_PATH_PARTS)
    )


def _redact_url(match: re.Match[str]) -> str:
    value = match.group(0)
    parsed = urlsplit(value)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query_keys = {_normalized_identifier_key(key) for key, _ in query}
    path_parts = {
        part.casefold() for part in unquote(parsed.path).split("/") if part
    }
    if (
        parsed.scheme.casefold() == "file"
        or _INTERNAL_URL_HOST.search(parsed.hostname or "")
        or _literal_internal_host(parsed.hostname or "")
        or query_keys.intersection(_SENSITIVE_QUERY_KEYS)
        or path_parts.intersection(_INTERNAL_PATH_PARTS)
        or any(_internal_payload(query_value) for _, query_value in query)
        or _internal_payload(parsed.fragment)
    ):
        return _REDACTION
    return value


def _redact_internal_references(value: str) -> str:
    protected_urls: dict[str, str] = {}

    def classify_url(match: re.Match[str]) -> str:
        redacted = _redact_url(match)
        if redacted != match.group(0):
            return redacted
        placeholder = f"PUBLICURLTOKEN{len(protected_urls)}END"
        protected_urls[placeholder] = match.group(0)
        return placeholder

    text = _URL_REFERENCE.sub(classify_url, str(value or ""))
    text = _PATH_REFERENCE.sub(_REDACTION, text)
    text = _IDENTIFIER_REFERENCE.sub(_REDACTION, text)
    text = _RAG_REFERENCE.sub(_REDACTION, text)
    text = re.sub(rf"(?:{re.escape(_REDACTION)}\s*)+", _REDACTION, text).strip()
    for placeholder, url in protected_urls.items():
        text = text.replace(placeholder, url)
    return text


def _presentation_text(value: Any, *, limit: int = 240) -> str:
    normalized = normalize_evidence_text(value)
    return presentation_evidence_excerpt(
        _redact_internal_references(normalized),
        limit=limit,
    )


def _safe_locator(citation: Mapping[str, Any]) -> str:
    locator = str(citation.get("locator") or citation.get("url") or "").strip()
    return _presentation_text(locator) or "—"


def _safe_section_title(citation: Mapping[str, Any]) -> str:
    section_title = str(citation.get("section_title") or "").strip()
    return _presentation_text(section_title) or "—"


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
                    summary=_escape_cell(_presentation_text(citation)) or "—",
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
        identity = canonical_source_identity(citation)
        file_name = _safe_group_label(citation)
        key = (source_type.casefold(), identity.casefold())
        entry = grouped.setdefault(
            key,
            {
                "file_name": file_name,
                "source_type": source_type or "rag",
                "sections": [],
                "count": 0,
            },
        )
        section = _safe_section_title(citation)
        if section != "—" and section not in entry["sections"]:
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

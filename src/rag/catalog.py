"""Lightweight, versioned resource metadata for Planner awareness."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from src.config import get_rag_settings

from .constants import DATABASE_FILENAME
from .models import ChildChunk, ResourceCatalogEntry, SourceDocument

CATALOG_VERSION = "2"
_STRUCTURED_SUFFIXES = {"csv", "xlsx", "xls"}
_TOPIC_SEPARATOR = re.compile(r"\s*(?:>|＞|/|／|\||｜)\s*")
_PARAMETER_ALIASES = {
    "反应温度": ("反应温度", "聚合温度"),
    "反应压力": ("反应压力", "聚合压力"),
    "停留时间": ("停留时间",),
    "搅拌速率": ("搅拌速率", "搅拌速度", "搅拌转速"),
    "循环气组成": ("循环气组成",),
    "催化剂注入量": ("催化剂注入量", "催化剂用量"),
    "催化剂": ("催化剂",),
    "共聚单体配比": ("共聚单体配比", "共聚单体用量"),
    "共聚单体": ("共聚单体",),
    "氢气": ("氢气", "氢调"),
    "原料纯度": ("原料纯度", "原料杂质"),
}
_METRIC_ALIASES = {
    "熔融指数": ("熔融指数", "熔体流动速率", "MFR", "MI"),
    "密度": ("密度",),
    "分子量分布": ("分子量分布",),
    "分子量": ("分子量",),
    "灰分": ("灰分",),
    "鱼眼": ("鱼眼",),
    "凝胶": ("凝胶",),
    "粒径分布": ("粒径分布",),
}
_CAUSAL_MARKERS = ("影响", "导致", "使得", "引起", "决定", "调节", "作用于")
_CONTROL_MARKERS = ("控制范围", "操作范围", "上限", "下限", "设定值", "报警值", "控制限")
_NUMERIC_RANGE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:-|–|—|~|～|至|到)\s*\d+(?:\.\d+)?\s*"
    r"(?:℃|°C|MPa|kPa|Pa|%|rpm|min|h|小时|分钟)?",
    re.IGNORECASE,
)


def build_catalog_entry(
    document: SourceDocument,
    chunks: list[ChildChunk] | tuple[ChildChunk, ...] = (),
) -> ResourceCatalogEntry:
    """Build compact deterministic metadata without invoking an LLM or retrieval."""

    extension = str(document.metadata.get("extension") or "").strip().lower()
    file_name = Path(document.source).name or f"{document.title}.{extension}"
    topics = _document_topics(document)
    structured = extension in _STRUCTURED_SUFFIXES
    if structured:
        content_type = "structured_data"
    elif extension in {"pdf", "doc", "docx"}:
        content_type = "technical_document"
    else:
        content_type = "text_document"

    supports = ["rag", "citation", "qualitative_analysis"]
    if structured:
        supports.append("structured_data")
    if extension == "csv":
        supports.extend(("statistical_analysis", "chart"))
    capabilities, coverage_evidence = _analyze_capabilities(document, chunks)

    return ResourceCatalogEntry(
        resource_id=document.doc_id,
        version_id=document.version_id,
        file_name=file_name,
        file_type=extension or document.doc_type or "unknown",
        sha256=document.content_hash,
        indexed=True,
        summary=_document_summary(document, topics),
        topics=topics,
        content_type=content_type,
        has_structured_data=structured,
        supports=tuple(supports),
        capabilities=capabilities,
        coverage_evidence=coverage_evidence,
        catalog_version=CATALOG_VERSION,
    )


def load_active_catalog(storage_root: Path | None = None) -> list[dict[str, Any]]:
    """Read active catalog rows directly from SQLite without opening RAG backends."""

    root = storage_root or get_rag_settings().storage_root
    database_path = Path(root) / DATABASE_FILENAME
    if not database_path.is_file():
        return []
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(resource_catalog)")
        }
        capabilities_sql = (
            "c.capabilities_json" if "capabilities_json" in columns else "'{}'"
        )
        rows = connection.execute(
            f"""
            SELECT v.doc_id AS resource_id, c.version_id, c.file_name, c.file_type,
                   v.content_hash AS sha256, c.summary, c.topics_json,
                   c.content_type, c.has_structured_data, c.supports_json,
                   {capabilities_sql} AS capabilities_json,
                   c.catalog_version, c.created_at, c.updated_at, d.source
            FROM resource_catalog AS c
            JOIN document_versions AS v ON v.version_id = c.version_id
            JOIN documents AS d
              ON d.doc_id = v.doc_id AND d.active_version_id = v.version_id
            WHERE v.status = 'ready'
            ORDER BY c.file_name, v.doc_id
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).casefold():
            return []
        raise
    finally:
        if connection is not None:
            connection.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            topics = json.loads(row["topics_json"])
            supports = json.loads(row["supports_json"])
            capabilities = json.loads(row["capabilities_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            not isinstance(topics, list)
            or not isinstance(supports, list)
            or not isinstance(capabilities, dict)
        ):
            continue
        result.append(
            {
                "resource_id": row["resource_id"],
                "file_id": row["resource_id"],
                "version_id": row["version_id"],
                "name": row["file_name"],
                "file_name": row["file_name"],
                "type": row["file_type"],
                "file_type": row["file_type"],
                "path": row["source"],
                "sha256": row["sha256"],
                "indexed": True,
                "summary": row["summary"],
                "topics": [str(topic) for topic in topics],
                "content_type": row["content_type"],
                "has_structured_data": bool(row["has_structured_data"]),
                "supports": [str(capability) for capability in supports],
                "capabilities": capabilities,
                "catalog_version": row["catalog_version"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "catalog_source": "knowledge_base",
            }
        )
    return result


def _contains_alias(text: str, alias: str) -> bool:
    if alias.isascii() and alias.isalnum():
        return re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE) is not None
    return alias.casefold() in text.casefold()


def _detected_terms(text: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    return [
        canonical
        for canonical, values in aliases.items()
        if any(_contains_alias(text, alias) for alias in values)
    ]


def _analyze_capabilities(
    document: SourceDocument,
    chunks: list[ChildChunk] | tuple[ChildChunk, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    document_text = "\n".join(block.text for block in document.blocks)
    parameter_mentions = _detected_terms(document_text, _PARAMETER_ALIASES)
    metric_mentions = _detected_terms(document_text, _METRIC_ALIASES)
    analyzable = [chunk for chunk in chunks if str(chunk.content or "").strip()]
    statuses = {
        "causal_evidence": "unknown",
        "numeric_ranges": "unknown",
        "control_limits": "unknown",
    }
    provenance: dict[str, list[dict[str, Any]]] = {
        "causal_evidence": [],
        "numeric_ranges": [],
        "control_limits": [],
    }
    if analyzable:
        statuses = {key: "not_detected" for key in statuses}
        for chunk in analyzable:
            text = str(chunk.content)
            chunk_parameters = _detected_terms(text, _PARAMETER_ALIASES)
            chunk_metrics = _detected_terms(text, _METRIC_ALIASES)
            detected: list[str] = []
            if (
                chunk_parameters
                and chunk_metrics
                and any(marker in text for marker in _CAUSAL_MARKERS)
            ):
                detected.append("causal_evidence")
            if chunk_parameters and _NUMERIC_RANGE_RE.search(text):
                detected.append("numeric_ranges")
            if chunk_parameters and any(marker in text for marker in _CONTROL_MARKERS):
                detected.append("control_limits")
            for capability in detected:
                statuses[capability] = "detected"
                provenance[capability].append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "section_path": str(chunk.metadata.get("section_path") or ""),
                    }
                )
    capabilities: dict[str, Any] = {
        "parameter_mentions": parameter_mentions,
        "metric_mentions": metric_mentions,
        **statuses,
    }
    return capabilities, provenance


def _document_topics(document: SourceDocument) -> tuple[str, ...]:
    topics: list[str] = []
    for candidate in (document.title, *[block.section_path for block in document.blocks]):
        for value in _TOPIC_SEPARATOR.split(str(candidate or "")):
            normalized = value.strip(" \t\r\n-—_:：")
            if normalized and normalized not in topics:
                topics.append(normalized)
            if len(topics) >= 12:
                return tuple(topics)
    return tuple(topics)


def _document_summary(
    document: SourceDocument,
    topics: tuple[str, ...],
) -> str:
    selected = [topic for topic in topics if topic != document.title][:6]
    if selected:
        summary = f"{document.title}，主要涵盖{'、'.join(selected)}。"
    else:
        excerpt = next(
            (" ".join(block.text.split()) for block in document.blocks if block.text.strip()),
            "",
        )
        if excerpt:
            summary = f"{document.title}：{excerpt[:180].rstrip('。')}。"
        else:
            summary = f"{document.title}，已建立知识库索引。"
    if len(summary) <= 240:
        return summary
    return summary[:239].rstrip("、，。：;； ") + "。"

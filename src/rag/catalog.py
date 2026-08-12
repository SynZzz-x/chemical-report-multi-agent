"""Lightweight, versioned resource metadata for Planner awareness."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from src.config import get_rag_settings

from .constants import DATABASE_FILENAME
from .models import ResourceCatalogEntry, SourceDocument

CATALOG_VERSION = "1"
_STRUCTURED_SUFFIXES = {"csv", "xlsx", "xls"}
_TOPIC_SEPARATOR = re.compile(r"\s*(?:>|＞|/|／|\||｜)\s*")


def build_catalog_entry(document: SourceDocument) -> ResourceCatalogEntry:
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
        rows = connection.execute(
            """
            SELECT v.doc_id AS resource_id, c.version_id, c.file_name, c.file_type,
                   v.content_hash AS sha256, c.summary, c.topics_json,
                   c.content_type, c.has_structured_data, c.supports_json,
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
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(topics, list) or not isinstance(supports, list):
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
                "catalog_version": row["catalog_version"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "catalog_source": "knowledge_base",
            }
        )
    return result


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

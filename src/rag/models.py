"""Immutable records shared by RAG loading, storage, and retrieval layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONMetadata: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class StructuralBlock:
    text: str
    block_type: str
    section_path: str
    page_start: int | None
    page_end: int | None
    clause_no: str | None


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    content_hash: str
    version_id: str
    title: str
    doc_type: str
    source: str
    blocks: tuple[StructuralBlock, ...]
    metadata: JSONMetadata


@dataclass(frozen=True)
class ParentChunk:
    section_id: str
    parent_id: str
    version_id: str
    content: str
    metadata: JSONMetadata


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_id: str
    version_id: str
    content: str
    embedding_text: str
    ordinal: int
    metadata: JSONMetadata


@dataclass(frozen=True)
class ResourceCatalogEntry:
    resource_id: str
    version_id: str
    file_name: str
    file_type: str
    sha256: str
    indexed: bool
    summary: str
    topics: tuple[str, ...]
    content_type: str
    has_structured_data: bool
    supports: tuple[str, ...]
    capabilities: JSONMetadata
    coverage_evidence: JSONMetadata
    catalog_version: str


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    rank: int
    score: float
    backend: str

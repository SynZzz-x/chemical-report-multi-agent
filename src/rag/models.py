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
    version_id: str
    title: str
    doc_type: str
    source: str
    blocks: tuple[StructuralBlock, ...]
    metadata: JSONMetadata


@dataclass(frozen=True)
class ParentChunk:
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
class RankedHit:
    chunk_id: str
    rank: int
    score: float
    backend: str

"""Lazy public interfaces for the hybrid chemical RAG pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ChemicalTokenizer": (".tokenizer", "ChemicalTokenizer"),
    "ChemicalRAGService": (".service", "ChemicalRAGService"),
    "ChildChunk": (".models", "ChildChunk"),
    "ParentChunk": (".models", "ParentChunk"),
    "RankedHit": (".models", "RankedHit"),
    "ResourceCatalogEntry": (".models", "ResourceCatalogEntry"),
    "HybridRetriever": (".hybrid_retriever", "HybridRetriever"),
    "SourceDocument": (".models", "SourceDocument"),
    "StructuralBlock": (".models", "StructuralBlock"),
    "TEIEmbeddings": (".embeddings", "TEIEmbeddings"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

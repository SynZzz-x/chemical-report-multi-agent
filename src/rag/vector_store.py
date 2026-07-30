"""Direct persistent Chroma operations for hybrid RAG child vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config import RAGSettings, get_rag_settings

from .bm25_store import DISTANCE_METRIC, INDEX_FINGERPRINT_KEYS, index_fingerprint
from .models import ChildChunk, RankedHit

COLLECTION_NAME = "chemical_documents_v3_qwen3_1024_cosine"
_COLLECTION_CONFIGURATION = {"hnsw": {"space": DISTANCE_METRIC}}


class VectorStore:
    """Small direct wrapper over Chroma's persistent collection API."""

    def __init__(
        self,
        storage_root: Path,
        settings: RAGSettings | None = None,
        *,
        fingerprint: Mapping[str, str] | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is required for dense RAG retrieval. Install the project's "
                "chromadb dependency before opening the vector store."
            ) from exc

        self._settings = settings or get_rag_settings()
        self._fingerprint = dict(fingerprint or index_fingerprint(self._settings))
        if set(self._fingerprint) != INDEX_FINGERPRINT_KEYS:
            raise ValueError("RAG index fingerprint must include every compatibility key.")
        chroma_path = storage_root / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=self._fingerprint,
            configuration=_COLLECTION_CONFIGURATION,
            embedding_function=None,
        )
        self._validate_collection()

    @classmethod
    def open(
        cls,
        storage_root: Path,
        settings: RAGSettings | None = None,
        *,
        fingerprint: Mapping[str, str] | None = None,
    ) -> "VectorStore":
        """Construct a persistent Chroma vector store."""

        return cls(storage_root, settings, fingerprint=fingerprint)

    def upsert(
        self,
        chunks: Sequence[ChildChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store explicit child IDs, precomputed vectors, documents, and filters."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one precomputed embedding.")
        if not chunks:
            return
        vectors = [list(vector) for vector in embeddings]
        expected_dimension = self._settings.embedding_dimension
        if any(len(vector) != expected_dimension for vector in vectors):
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension} values per vector."
            )
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.embedding_text for chunk in chunks],
            metadatas=[_compact_metadata(chunk) for chunk in chunks],
            embeddings=vectors,
        )

    def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
        doc_type_filter: str | None = None,
    ) -> list[RankedHit]:
        """Query a precomputed vector and convert cosine distance to similarity."""

        if limit <= 0:
            return []
        vector = list(query_embedding)
        if len(vector) != self._settings.embedding_dimension:
            raise ValueError(
                "Query embedding dimension mismatch: expected "
                f"{self._settings.embedding_dimension}, got {len(vector)}."
            )
        where = {"doc_type": doc_type_filter} if doc_type_filter is not None else None
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=where,
            include=["distances"],
        )
        identifiers = result.get("ids", [[]])
        distances = result.get("distances", [[]])
        first_ids = identifiers[0] if identifiers else []
        first_distances = distances[0] if distances else []
        return [
            RankedHit(
                chunk_id=chunk_id,
                rank=rank,
                score=max(0.0, 1.0 - float(distance)),
                backend="dense",
            )
            for rank, (chunk_id, distance) in enumerate(
                zip(first_ids, first_distances), start=1
            )
        ]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete explicit vector IDs after a manifest version is retired."""

        if chunk_ids:
            self._collection.delete(ids=list(chunk_ids))

    def count(self) -> int:
        """Return the number of stored vectors."""

        return int(self._collection.count())

    def _validate_collection(self) -> None:
        metadata = dict(self._collection.metadata or {})
        if metadata != self._fingerprint:
            raise RuntimeError(
                "Chroma collection fingerprint mismatch. Rebuild the SQLite and "
                "Chroma indexes instead of reusing incompatible vectors."
            )
        configuration = _configuration_mapping(
            getattr(self._collection, "configuration", None)
        )
        hnsw = _configuration_mapping(configuration.get("hnsw"))
        space = _configuration_value(hnsw.get("space"))
        if space != DISTANCE_METRIC:
            raise RuntimeError(
                "Chroma collection is not configured for cosine distance. Rebuild "
                "the collection with the required cosine HNSW configuration."
            )


def _compact_metadata(chunk: ChildChunk) -> dict[str, str | int | float | bool]:
    """Select Chroma-safe filter metadata rather than copying the full JSON blob."""

    metadata = chunk.metadata
    result: dict[str, str | int | float | bool] = {
        "doc_id": str(metadata["doc_id"]),
        "doc_type": str(metadata["doc_type"]),
        "version_id": chunk.version_id,
        "parent_id": chunk.parent_id,
        "ordinal": chunk.ordinal,
    }
    for key in ("section_id", "section_path", "clause_no"):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


def _configuration_mapping(value: Any) -> Mapping[str, Any]:
    """Read Chroma configuration from dict, Pydantic, or attribute-backed forms."""

    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            dumped = method()
            if isinstance(dumped, Mapping):
                return dumped
    hnsw = getattr(value, "hnsw", None)
    if hnsw is not None:
        return {"hnsw": hnsw}
    space = getattr(value, "space", None)
    if space is not None:
        return {"space": space}
    return {}


def _configuration_value(value: Any) -> str | None:
    """Resolve a configuration enum or primitive to its lowercase text value."""

    resolved = getattr(value, "value", value)
    return resolved.lower() if isinstance(resolved, str) else None

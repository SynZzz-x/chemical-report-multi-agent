"""Service façade for durable hybrid-RAG ingestion and retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from src.config import RAGSettings, get_rag_settings

from .bm25_store import BM25Store
from .chunking import ChemicalDocumentLoader, StructureAwareChunker
from .embeddings import TEIEmbeddings
from .hybrid_retriever import HybridRetriever, _sanitize_exception
from .tokenizer import ChemicalTokenizer
from .vector_store import VectorStore


EMBEDDING_BATCH_SIZE = 64


class ChemicalRAGService:
    """Coordinate staged ingestion and expose the retrieval-only result contract."""

    def __init__(
        self,
        settings: RAGSettings | None = None,
        *,
        embeddings: TEIEmbeddings | None = None,
        tokenizer: ChemicalTokenizer | None = None,
        chunker: StructureAwareChunker | None = None,
        bm25_store: BM25Store | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._settings = settings or get_rag_settings()
        self._embeddings = embeddings or TEIEmbeddings(self._settings)
        self._tokenizer = tokenizer or ChemicalTokenizer(self._embeddings)
        self._chunker = chunker or StructureAwareChunker(self._tokenizer, self._settings)
        self._startup_warnings: list[str] = []
        self._bm25_store = bm25_store
        self._vector_store = vector_store
        if self._bm25_store is None:
            try:
                self._bm25_store = BM25Store.open(
                    self._settings.storage_root, self._tokenizer, self._settings
                )
            except Exception as exc:
                self._startup_warnings.append(f"bm25: {_sanitize_exception(exc)}")
        if self._vector_store is None:
            try:
                self._vector_store = VectorStore.open(self._settings.storage_root, self._settings)
            except Exception as exc:
                self._startup_warnings.append(f"dense: {_sanitize_exception(exc)}")
        self._cleanup_incomplete_versions()
        self._retriever = HybridRetriever(
            self._bm25_store,
            self._vector_store,
            self._embeddings,
            self._tokenizer,
            self._settings,
            startup_warnings=self._startup_warnings,
        )

    def close(self) -> None:
        """Close SQLite resources owned by this service."""

        if self._bm25_store is not None:
            self._bm25_store.close()

    def ingest(self, file_paths: Iterable[str]) -> dict[str, Any]:
        """Ingest each file independently, never activating incomplete versions."""

        if self._bm25_store is None or self._vector_store is None:
            return {
                "success": False,
                "loaded_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "warnings": list(self._startup_warnings),
                "files": [],
                "error": "Both SQLite and Chroma stores are required for ingestion.",
            }

        results: list[dict[str, Any]] = []
        total_chunks = 0
        for path in file_paths:
            outcome = self._ingest_one(path)
            results.append(outcome)
            if outcome["status"] == "loaded":
                total_chunks += int(outcome["chunks"])
        loaded = sum(item["status"] == "loaded" for item in results)
        skipped = sum(item["status"] == "skipped" for item in results)
        failed = sum(item["status"] == "failed" for item in results)
        return {
            "success": failed == 0,
            "loaded_files": loaded,
            "skipped_files": skipped,
            "failed_files": failed,
            "total_chunks": total_chunks,
            "warnings": list(self._startup_warnings),
            "files": results,
        }

    def query(
        self,
        question: str,
        top_k: int | None = None,
        doc_type_filter: str | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Retrieve evidence without generating an answer."""

        return self._retriever.retrieve(
            question, top_k, doc_type_filter, similarity_threshold
        )

    def stats(self) -> dict[str, Any]:
        """Return compact storage counts and currently unavailable backends."""

        result: dict[str, Any] = {"warnings": list(self._startup_warnings)}
        if self._bm25_store is not None:
            try:
                result.update(self._bm25_store.stats())
            except Exception as exc:
                result["warnings"].append(f"bm25: {_sanitize_exception(exc)}")
        else:
            result.update({"documents": 0, "ready_versions": 0, "chunks": 0, "fts_rows": 0})
        if self._vector_store is not None:
            try:
                result["vectors"] = self._vector_store.count()
            except Exception as exc:
                result["vectors"] = 0
                result["warnings"].append(f"dense: {_sanitize_exception(exc)}")
        else:
            result["vectors"] = 0
        result["success"] = not result["warnings"]
        return result

    def _ingest_one(self, path: str) -> dict[str, Any]:
        assert self._bm25_store is not None
        assert self._vector_store is not None
        document = None
        staged = False
        vector_ids: list[str] = []
        try:
            document = ChemicalDocumentLoader.load(path)
            parents, chunks = self._chunker.chunk(document)
            try:
                self._bm25_store.begin_version(document, parents, chunks)
            except ValueError as exc:
                if "already ready" in str(exc):
                    return {
                        "path": path,
                        "status": "skipped",
                        "chunks": len(chunks),
                        "version_id": document.version_id,
                    }
                raise
            staged = True
            vector_ids = [chunk.chunk_id for chunk in chunks]
            self._embed_and_upsert(chunks)
            _, prior_vector_ids = self._bm25_store.activate_version(
                document.doc_id, document.version_id
            )
            warnings: list[str] = []
            try:
                self._vector_store.delete(sorted(prior_vector_ids))
            except Exception as exc:
                warnings.append(f"dense cleanup: {_sanitize_exception(exc)}")
            return {
                "path": path,
                "status": "loaded",
                "chunks": len(chunks),
                "version_id": document.version_id,
                "warnings": warnings,
            }
        except Exception as exc:
            cleanup_warnings: list[str] = []
            if staged and document is not None:
                try:
                    vector_ids = list(self._bm25_store.mark_failed(document.version_id))
                except Exception as cleanup_exc:
                    cleanup_warnings.append(
                        f"manifest cleanup: {_sanitize_exception(cleanup_exc)}"
                    )
                try:
                    self._vector_store.delete(vector_ids)
                except Exception as cleanup_exc:
                    cleanup_warnings.append(
                        f"dense cleanup: {_sanitize_exception(cleanup_exc)}"
                    )
            return {
                "path": path,
                "status": "failed",
                "chunks": 0,
                "error": _sanitize_exception(exc),
                "warnings": cleanup_warnings,
            }

    def _embed_and_upsert(self, chunks: Sequence[Any]) -> None:
        assert self._vector_store is not None
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
            embeddings = self._embeddings.embed_documents(
                [chunk.embedding_text for chunk in batch]
            )
            self._vector_store.upsert(batch, embeddings)

    def _cleanup_incomplete_versions(self) -> None:
        """Remove vectors first, then their building/failed SQLite manifest rows."""

        if self._bm25_store is None:
            return
        if self._vector_store is None:
            self._startup_warnings.append(
                "startup cleanup: dense store unavailable; incomplete manifests retained."
            )
            return
        try:
            incomplete = self._bm25_store.incomplete_versions()
        except Exception as exc:
            self._startup_warnings.append(f"startup cleanup: {_sanitize_exception(exc)}")
            return
        for version_id, chunk_ids in incomplete.items():
            try:
                self._vector_store.delete(sorted(chunk_ids))
                self._bm25_store.cleanup_incomplete(version_id)
            except Exception as exc:
                self._startup_warnings.append(
                    f"startup cleanup: {_sanitize_exception(exc)}"
                )

"""Service façade for durable hybrid-RAG ingestion and retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import RAGSettings, get_rag_settings

from .bm25_store import BM25Store, index_fingerprint
from .catalog import build_catalog_entry
from .chunking import ChemicalDocumentLoader, StructureAwareChunker
from .embeddings import TEIEmbeddings
from .hybrid_retriever import HybridRetriever, _sanitize_exception
from .tokenizer import ChemicalTokenizer
from .vector_store import VectorStore, fingerprint_generation


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
                if not self._bm25_store.lexical_available:
                    self._startup_warnings.append(
                        "bm25: "
                        + (
                            self._bm25_store.lexical_warning
                            or "SQLite FTS5 lexical search is unavailable."
                        )
                    )
            except Exception as exc:
                self._startup_warnings.append(f"bm25: {_sanitize_exception(exc)}")
        if self._vector_store is None and self._bm25_store is not None:
            try:
                self._vector_store = VectorStore.open(self._settings.storage_root, self._settings)
            except Exception as exc:
                self._startup_warnings.append(f"dense: {_sanitize_exception(exc)}")
        elif self._vector_store is None:
            self._startup_warnings.append(
                "dense: canonical manifest is unavailable; vector retrieval was not opened."
            )
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

    @classmethod
    def rebuild(
        cls,
        file_paths: Iterable[str],
        settings: RAGSettings | None = None,
    ) -> dict[str, Any]:
        """Build beside the active index, then atomically archive and replace it."""

        sources = list(dict.fromkeys(file_paths))
        if not sources:
            return {
                "success": False,
                "activated": False,
                "error": "Rebuild requires at least one source document.",
            }
        base_settings = settings or get_rag_settings()
        storage_root = base_settings.storage_root
        generation = fingerprint_generation(index_fingerprint(base_settings))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging_root = storage_root.parent / (
            f".{storage_root.name}.rebuild-{generation}-{uuid4().hex[:8]}"
        )
        staged_settings = replace(base_settings, storage_root=staging_root)
        staged_service: ChemicalRAGService | None = None
        try:
            staged_service = cls(staged_settings)
            ingestion = staged_service.ingest(sources)
        except Exception as exc:
            return {
                "success": False,
                "activated": False,
                "generation": generation,
                "staging_path": str(staging_root),
                "error": _sanitize_exception(exc),
            }
        finally:
            if staged_service is not None:
                try:
                    staged_service.close()
                except Exception:
                    pass

        completed_files = int(ingestion.get("activated_files", 0)) + int(
            ingestion.get("skipped_files", 0)
        )
        rebuild_complete = (
            int(ingestion.get("failed_files", 0)) == 0
            and completed_files == len(sources)
            and int(ingestion.get("cleanup_pending_versions", 0)) == 0
            and int(ingestion.get("incomplete_versions", 0)) == 0
        )
        if not rebuild_complete:
            return {
                "success": False,
                "activated": False,
                "generation": generation,
                "staging_path": str(staging_root),
                "error": (
                    "Rebuild staging did not produce a complete searchable index; "
                    "the active index was not changed."
                ),
                "ingestion": ingestion,
            }

        archive_path: Path | None = None
        storage_root.parent.mkdir(parents=True, exist_ok=True)
        if storage_root.exists():
            archive_root = storage_root.parent / f"{storage_root.name}-archive"
            archive_root.mkdir(parents=True, exist_ok=True)
            archive_path = archive_root / (
                f"{timestamp}-{generation}-{uuid4().hex[:8]}"
            )
            storage_root.rename(archive_path)
        try:
            staging_root.rename(storage_root)
        except Exception as exc:
            if archive_path is not None and not storage_root.exists():
                archive_path.rename(storage_root)
            return {
                "success": False,
                "activated": False,
                "generation": generation,
                "staging_path": str(staging_root),
                "archive_path": str(archive_path) if archive_path else None,
                "error": f"Index activation failed: {_sanitize_exception(exc)}",
                "ingestion": ingestion,
            }
        return {
            "success": True,
            "activated": True,
            "generation": generation,
            "active_path": str(storage_root),
            "archive_path": str(archive_path) if archive_path else None,
            "ingestion": ingestion,
        }

    def ingest(self, file_paths: Iterable[str]) -> dict[str, Any]:
        """Ingest each file independently, never activating incomplete versions."""

        if self._bm25_store is None or self._vector_store is None:
            manifest = (
                self._manifest_diagnostics()
                if self._bm25_store is not None
                else {
                    "cleanup_pending_versions": 0,
                    "incomplete_versions": 0,
                    "warnings": [],
                }
            )
            return {
                "success": False,
                "loaded_files": 0,
                "loaded_with_warnings_files": 0,
                "activated_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "cleanup_pending_files": 0,
                "cleanup_tracking_failed_files": 0,
                "cleanup_pending_versions": manifest["cleanup_pending_versions"],
                "incomplete_versions": manifest["incomplete_versions"],
                "total_chunks": 0,
                "warnings": [*self._startup_warnings, *manifest["warnings"]],
                "files": [],
                "error": "Both SQLite and Chroma stores are required for ingestion.",
            }

        results: list[dict[str, Any]] = []
        total_chunks = 0
        activated_statuses = {
            "loaded",
            "loaded_with_warnings",
            "activated_with_untracked_cleanup",
        }
        for path in file_paths:
            outcome = self._ingest_one(path)
            results.append(outcome)
            if outcome["status"] in activated_statuses:
                total_chunks += int(outcome["chunks"])
        loaded = sum(item["status"] == "loaded" for item in results)
        loaded_with_warnings = sum(
            item["status"] == "loaded_with_warnings" for item in results
        )
        skipped = sum(item["status"] == "skipped" for item in results)
        failed = sum(item["status"] == "failed" for item in results)
        cleanup_pending = sum(bool(item.get("cleanup_pending")) for item in results)
        cleanup_tracking_failed = sum(
            bool(item.get("cleanup_tracking_failed")) for item in results
        )
        manifest = self._manifest_diagnostics()
        warnings = list(self._startup_warnings)
        for outcome in results:
            path = outcome["path"]
            warnings.extend(
                f"{path}: {warning}" for warning in outcome.get("warnings", [])
            )
        warnings.extend(manifest["warnings"])
        return {
            "success": (
                failed == 0
                and cleanup_pending == 0
                and cleanup_tracking_failed == 0
                and manifest["cleanup_pending_versions"] == 0
                and not self._startup_warnings
                and not manifest["warnings"]
            ),
            "loaded_files": loaded,
            "loaded_with_warnings_files": loaded_with_warnings,
            "activated_files": sum(
                item["status"] in activated_statuses for item in results
            ),
            "skipped_files": skipped,
            "failed_files": failed,
            "cleanup_pending_files": cleanup_pending,
            "cleanup_tracking_failed_files": cleanup_tracking_failed,
            "cleanup_pending_versions": manifest["cleanup_pending_versions"],
            "incomplete_versions": manifest["incomplete_versions"],
            "total_chunks": total_chunks,
            "warnings": warnings,
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
        result["success"] = (
            not result["warnings"]
            and result.get("cleanup_pending_versions", 0) == 0
        )
        return result

    def _ingest_one(self, path: str) -> dict[str, Any]:
        assert self._bm25_store is not None
        assert self._vector_store is not None
        document = None
        staged = False
        vector_ids: list[str] = []
        try:
            document = ChemicalDocumentLoader.load(path)
            if not document.blocks:
                raise ValueError(
                    "Document extraction produced no structural blocks; "
                    "the current active version was preserved."
                )
            if self._bm25_store.is_ready_version(document.version_id):
                if not self._bm25_store.has_catalog(document.version_id):
                    self._bm25_store.add_catalog_to_ready_version(
                        build_catalog_entry(document)
                    )
                return {
                    "path": path,
                    "status": "skipped",
                    "chunks": 0,
                    "version_id": document.version_id,
                }
            parents, chunks = self._chunker.chunk(document)
            if not parents or not chunks:
                raise ValueError(
                    "Document chunking produced no searchable parents or children; "
                    "the current active version was preserved."
                )
            try:
                self._bm25_store.begin_version(
                    document,
                    parents,
                    chunks,
                    build_catalog_entry(document),
                )
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
            prior_version_id, prior_vector_ids = self._bm25_store.activate_version(
                document.doc_id, document.version_id
            )
            warnings: list[str] = []
            cleanup_pending = False
            cleanup_tracking_failed = False
            if prior_version_id is not None:
                try:
                    self._vector_store.delete(sorted(prior_vector_ids))
                    self._bm25_store.cleanup_incomplete(prior_version_id)
                except Exception as exc:
                    cleanup_pending = True
                    warnings.append(
                        "retired-version cleanup: "
                        f"{_sanitize_exception(exc)}"
                    )
            return {
                "path": path,
                "status": (
                    "activated_with_untracked_cleanup"
                    if cleanup_tracking_failed
                    else "loaded_with_warnings" if warnings else "loaded"
                ),
                "chunks": len(chunks),
                "version_id": document.version_id,
                "activated": True,
                "cleanup_pending": cleanup_pending,
                "cleanup_tracking_failed": cleanup_tracking_failed,
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

    def _manifest_diagnostics(self) -> dict[str, Any]:
        """Read current recovery state after the independent ingestion attempts."""

        assert self._bm25_store is not None
        warnings: list[str] = []
        cleanup_pending_versions = 0
        incomplete_versions = 0
        try:
            cleanup_pending_versions = int(
                self._bm25_store.stats().get("cleanup_pending_versions", 0)
            )
            incomplete_versions = len(self._bm25_store.incomplete_versions())
        except Exception as exc:
            warnings.append(f"manifest diagnostics: {_sanitize_exception(exc)}")
        return {
            "cleanup_pending_versions": cleanup_pending_versions,
            "incomplete_versions": incomplete_versions,
            "warnings": warnings,
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
        """Idempotently retire every non-active incomplete or pending version."""

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

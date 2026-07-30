"""Worker-compatible façade for the durable hybrid chemical RAG service."""

from __future__ import annotations

from typing import Any

from src.config import RAGSettings
from src.rag.service import ChemicalRAGService


class ChemicalKnowledgeBase:
    """Preserve the Worker API while delegating retrieval to hybrid RAG."""

    def __init__(self, settings: RAGSettings | None = None) -> None:
        self._service = ChemicalRAGService(settings)

    def load_documents(self, file_paths: list[str]) -> dict[str, Any]:
        """Ingest source documents through the versioned hybrid-RAG pipeline."""

        return self._service.ingest(file_paths)

    def query(
        self,
        question: str,
        top_k: int = 5,
        doc_type_filter: str | None = None,
        similarity_threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Return retrieval-only evidence and its hybrid diagnostics."""

        return self._service.query(
            question,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            similarity_threshold=similarity_threshold,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return durable-store counts and backend availability diagnostics."""

        return self._service.stats()

"""HTTP adapter for a Text Embeddings Inference (TEI) service."""

from __future__ import annotations

from typing import Any

import requests

from src.config import RAGSettings, get_rag_settings

QUERY_EMBEDDING_INSTRUCTION = (
    "Instruct: 给定一个化工领域问题，检索能够回答该问题的相关技术资料、标准条款、"
    "工艺条件、设备信息或安全规范\n"
    "Query: "
)


class TEIEmbeddings:
    """Generate model-accurate token counts and vectors through TEI."""

    def __init__(
        self,
        settings: RAGSettings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._settings = settings or get_rag_settings()
        self._base_url = self._normalize_base_url(self._settings.embedding_base_url)
        self._session = session or requests.Session()
        if self._settings.embedding_api_key:
            self._session.headers.update(
                {"Authorization": f"Bearer {self._settings.embedding_api_key}"}
            )

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[:-3]
        return normalized.rstrip("/")

    def health(self) -> bool:
        """Raise for an unhealthy TEI response and return ``True`` otherwise."""

        response = self._session.get(
            f"{self._base_url}/health",
            timeout=self._settings.embedding_timeout_seconds,
        )
        response.raise_for_status()
        return True

    def count_tokens(self, text: str) -> int:
        """Return the TEI model's token count for ``text``."""

        response = self._session.post(
            f"{self._base_url}/tokenize",
            json={"inputs": text},
            timeout=self._settings.embedding_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens") if isinstance(payload, dict) else payload
        if not isinstance(tokens, list):
            raise ValueError("TEI /tokenize response must be a token list.")
        return len(tokens)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents without adding a retrieval instruction."""

        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the fixed Qwen chemical-retrieval instruction."""

        return self._embed([f"{QUERY_EMBEDDING_INSTRUCTION}{text}"])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._session.post(
            f"{self._base_url}/v1/embeddings",
            json={"model": self._settings.embedding_model, "input": texts},
            timeout=self._settings.embedding_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("TEI /v1/embeddings response must contain a data list.")

        try:
            indexed_rows = sorted(rows, key=lambda row: row["index"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Each TEI embedding row must contain an index.") from exc

        expected_indexes = list(range(len(texts)))
        actual_indexes = [row.get("index") for row in indexed_rows if isinstance(row, dict)]
        if len(indexed_rows) != len(texts) or actual_indexes != expected_indexes:
            raise ValueError("TEI must return exactly one indexed embedding per input.")

        return [self._validated_vector(row) for row in indexed_rows]

    def _validated_vector(self, row: dict[str, Any]) -> list[float]:
        vector = row.get("embedding")
        if not isinstance(vector, list):
            raise ValueError("Each TEI embedding row must contain an embedding list.")
        if len(vector) != self._settings.embedding_dimension:
            raise ValueError(
                "TEI embedding dimension mismatch: expected "
                f"{self._settings.embedding_dimension}, got {len(vector)}."
            )
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("TEI embedding values must be numeric.") from exc

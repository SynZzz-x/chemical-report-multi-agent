"""SQLite manifest and FTS5-backed lexical storage for hybrid RAG."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from src.config import RAGSettings, get_rag_settings

from .models import ChildChunk, ParentChunk, RankedHit, SourceDocument
from .tokenizer import CHEMICAL_TERMS, ChemicalTokenizer

DATABASE_FILENAME = "hybrid.sqlite"
INDEX_SCHEMA_VERSION = "3"
DISTANCE_METRIC = "cosine"
INDEX_FINGERPRINT_KEYS = frozenset(
    {
        "schema_version",
        "embedding_model",
        "embedding_model_revision",
        "embedding_dimension",
        "distance_metric",
        "tei_api_version",
        "document_embedding_text_format",
        "query_instruction_version",
        "chunking_config_hash",
        "normalization_version",
        "jieba_dictionary_version",
    }
)
_GB_STANDARD_RE = re.compile(
    r"\bGB(?:\s*[/_-]\s*T)?\s*[_/-]?\s*"
    r"(\d{3,}(?:\s*[_/-]\s*\d{2,4})?)\b",
    flags=re.IGNORECASE,
)
_ASTM_STANDARD_RE = re.compile(
    r"\bASTM\s*[_/-]?\s*([A-Z]{1,4})\s*[_/-]?\s*"
    r"(\d{2,}(?:\s*[_/-]\s*\d{2,4})?)\b",
    flags=re.IGNORECASE,
)
_QWEN_MODEL_RE = re.compile(
    r"\bQwen\s*[_/-]?\s*(\d+(?:\.\d+)*)"
    r"(?:\s*[_/-]?\s*(Embedding))?\b",
    flags=re.IGNORECASE,
)


def index_fingerprint(settings: RAGSettings | None = None) -> dict[str, str]:
    """Return every value that makes persisted RAG indexes incompatible."""

    settings = settings or get_rag_settings()
    chunking_values = {
        "child_target_tokens": settings.child_target_tokens,
        "child_max_tokens": settings.child_max_tokens,
        "child_overlap_tokens": settings.child_overlap_tokens,
        "parent_target_tokens": settings.parent_target_tokens,
        "parent_max_tokens": settings.parent_max_tokens,
    }
    chunking_hash = hashlib.sha256(
        json.dumps(chunking_values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    jieba_dictionary_version = hashlib.sha256(
        "\0".join(CHEMICAL_TERMS).encode()
    ).hexdigest()
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_dimension": str(settings.embedding_dimension),
        "distance_metric": DISTANCE_METRIC,
        "tei_api_version": "openai-compatible-v1",
        "document_embedding_text_format": "document-context-v1",
        "query_instruction_version": "chemical-retrieval-v1",
        "chunking_config_hash": chunking_hash,
        "normalization_version": "jieba-chemical-v1",
        "jieba_dictionary_version": jieba_dictionary_version,
    }


class BM25Store:
    """Canonical document-version manifest plus an FTS5 lexical index."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        tokenizer: ChemicalTokenizer,
        fingerprint: Mapping[str, str],
    ) -> None:
        self._connection = connection
        self._tokenizer = tokenizer
        self._fingerprint = dict(fingerprint)

    @classmethod
    def open(
        cls,
        storage_root: Path,
        tokenizer: ChemicalTokenizer,
        settings: RAGSettings | None = None,
        *,
        fingerprint: Mapping[str, str] | None = None,
    ) -> "BM25Store":
        """Open ``hybrid.sqlite`` after verifying its full index fingerprint."""

        storage_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(storage_root / DATABASE_FILENAME)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            cls._create_schema(connection)
            expected = dict(fingerprint or index_fingerprint(settings))
            if set(expected) != INDEX_FINGERPRINT_KEYS:
                raise ValueError("RAG index fingerprint must include every compatibility key.")
            cls._validate_fingerprint(connection, expected)
            return cls(connection, tokenizer, expected)
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        try:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        active_version_id TEXT,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS document_versions (
                        version_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                        content_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS parents (
                        parent_id TEXT PRIMARY KEY,
                        section_id TEXT NOT NULL,
                        version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        parent_id TEXT NOT NULL REFERENCES parents(parent_id),
                        version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        ordinal INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS rag_index_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        title, section_path, clause_number, bm25_text,
                        chunk_id UNINDEXED
                    )
                    """
                )
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower() or "virtual table" in str(exc).lower():
                raise RuntimeError(
                    "SQLite FTS5 is unavailable. Install or use a Python/SQLite build "
                    "compiled with ENABLE_FTS5 before opening the RAG index."
                ) from exc
            raise

    @staticmethod
    def _validate_fingerprint(
        connection: sqlite3.Connection, expected: Mapping[str, str]
    ) -> None:
        existing = dict(
            connection.execute("SELECT key, value FROM rag_index_meta").fetchall()
        )
        if not existing:
            has_data = connection.execute(
                "SELECT EXISTS(SELECT 1 FROM documents LIMIT 1)"
            ).fetchone()[0]
            if has_data:
                raise RuntimeError(
                    "Existing RAG data has no index fingerprint. Remove and rebuild "
                    "the incompatible index instead of reusing it."
                )
            with connection:
                connection.executemany(
                    "INSERT INTO rag_index_meta(key, value) VALUES (?, ?)",
                    sorted(expected.items()),
                )
            return
        if existing != dict(expected):
            missing = sorted(set(expected) - set(existing))
            unexpected = sorted(set(existing) - set(expected))
            changed = sorted(
                key
                for key in set(existing).intersection(expected)
                if existing[key] != expected[key]
            )
            differences = ", ".join(
                filter(
                    None,
                    (
                        f"missing={missing}" if missing else "",
                        f"unexpected={unexpected}" if unexpected else "",
                        f"changed={changed}" if changed else "",
                    ),
                )
            )
            raise RuntimeError(
                "RAG index fingerprint mismatch ("
                f"{differences}). Rebuild the SQLite and Chroma indexes."
            )

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def begin_version(
        self,
        document: SourceDocument,
        parents: Sequence[ParentChunk],
        chunks: Sequence[ChildChunk],
    ) -> None:
        """Stage one document version and all lexical rows without activating it."""

        self._validate_version_records(document, parents, chunks)
        now = _utc_now()
        with self._connection:
            existing = self._connection.execute(
                "SELECT doc_id, status FROM document_versions WHERE version_id = ?",
                (document.version_id,),
            ).fetchone()
            if existing is not None:
                if existing["doc_id"] != document.doc_id:
                    raise ValueError("version_id is already associated with another doc_id.")
                if existing["status"] == "ready":
                    raise ValueError("Cannot stage a version that is already ready.")
                self._delete_version_rows(document.version_id)
                self._connection.execute(
                    "DELETE FROM document_versions WHERE version_id = ?",
                    (document.version_id,),
                )
            self._connection.execute(
                """
                INSERT INTO documents(doc_id, source, active_version_id, updated_at)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(doc_id) DO UPDATE SET source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (document.doc_id, document.source, now),
            )
            self._connection.execute(
                """
                INSERT INTO document_versions(version_id, doc_id, content_hash, status, created_at)
                VALUES (?, ?, ?, 'building', ?)
                """,
                (document.version_id, document.doc_id, document.content_hash, now),
            )
            self._connection.executemany(
                """
                INSERT INTO parents(parent_id, section_id, version_id, content, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        parent.parent_id,
                        parent.section_id,
                        parent.version_id,
                        parent.content,
                        _metadata_json(parent.metadata),
                    )
                    for parent in parents
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO chunks(chunk_id, parent_id, version_id, content, metadata_json, ordinal)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.parent_id,
                        chunk.version_id,
                        chunk.content,
                        _metadata_json(chunk.metadata),
                        chunk.ordinal,
                    )
                    for chunk in chunks
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO chunks_fts(title, section_path, clause_number, bm25_text, chunk_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                [self._fts_row(document, chunk) for chunk in chunks],
            )

    def activate_version(self, doc_id: str, version_id: str) -> tuple[str | None, set[str]]:
        """Atomically activate a staged version and return superseded vector IDs."""

        with self._connection:
            version = self._connection.execute(
                "SELECT doc_id, status FROM document_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if version is None or version["doc_id"] != doc_id:
                raise ValueError("Cannot activate a version that does not belong to doc_id.")
            if version["status"] != "building":
                raise ValueError("Only a building version can be activated.")
            document = self._connection.execute(
                "SELECT active_version_id FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if document is None:
                raise ValueError("Cannot activate a version for an unknown document.")
            previous_version_id = document["active_version_id"]
            previous_chunk_ids: set[str] = set()
            if previous_version_id is not None:
                previous_chunk_ids = {
                    row["chunk_id"]
                    for row in self._connection.execute(
                        "SELECT chunk_id FROM chunks WHERE version_id = ?",
                        (previous_version_id,),
                    )
                }
                self._connection.execute(
                    "UPDATE document_versions SET status = 'inactive' WHERE version_id = ?",
                    (previous_version_id,),
                )
            self._connection.execute(
                "UPDATE document_versions SET status = 'ready' WHERE version_id = ?",
                (version_id,),
            )
            self._connection.execute(
                """
                UPDATE documents SET active_version_id = ?, updated_at = ? WHERE doc_id = ?
                """,
                (version_id, _utc_now(), doc_id),
            )
        return previous_version_id, previous_chunk_ids

    def mark_failed(self, version_id: str) -> set[str]:
        """Mark a non-active staged version failed and return its vector IDs."""

        with self._connection:
            row = self._connection.execute(
                """
                SELECT v.status, d.active_version_id
                FROM document_versions AS v
                JOIN documents AS d ON d.doc_id = v.doc_id
                WHERE v.version_id = ?
                """,
                (version_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Cannot mark an unknown document version failed.")
            if row["active_version_id"] == version_id:
                raise ValueError("Cannot mark an active document version failed.")
            if row["status"] not in {"building", "failed"}:
                raise ValueError("Only building document versions may be marked failed.")
            self._connection.execute(
                "UPDATE document_versions SET status = 'failed' WHERE version_id = ?",
                (version_id,),
            )
            return {
                item["chunk_id"]
                for item in self._connection.execute(
                    "SELECT chunk_id FROM chunks WHERE version_id = ?", (version_id,)
                )
            }

    def mark_cleanup_pending(self, version_id: str) -> set[str]:
        """Persist deferred vector retirement for a non-active version."""

        with self._connection:
            row = self._connection.execute(
                """
                SELECT v.status, d.active_version_id
                FROM document_versions AS v
                JOIN documents AS d ON d.doc_id = v.doc_id
                WHERE v.version_id = ?
                """,
                (version_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Cannot defer cleanup for an unknown document version.")
            if row["active_version_id"] == version_id:
                raise ValueError("Cannot defer cleanup for an active document version.")
            if row["status"] not in {"inactive", "cleanup_pending"}:
                raise ValueError("Only inactive document versions may defer cleanup.")
            self._connection.execute(
                "UPDATE document_versions SET status = 'cleanup_pending' WHERE version_id = ?",
                (version_id,),
            )
            return {
                item["chunk_id"]
                for item in self._connection.execute(
                    "SELECT chunk_id FROM chunks WHERE version_id = ?", (version_id,)
                )
            }

    def incomplete_versions(self) -> dict[str, set[str]]:
        """Return every recoverable manifest version with its vector IDs."""

        rows = self._connection.execute(
            """
            SELECT v.version_id, c.chunk_id
            FROM document_versions AS v
            LEFT JOIN chunks AS c ON c.version_id = v.version_id
            WHERE v.status IN ('building', 'failed', 'cleanup_pending')
            ORDER BY v.version_id, c.ordinal
            """
        )
        versions: dict[str, set[str]] = {}
        for row in rows:
            versions.setdefault(row["version_id"], set())
            if row["chunk_id"] is not None:
                versions[row["version_id"]].add(row["chunk_id"])
        return versions

    def cleanup_incomplete(self, version_id: str) -> set[str]:
        """Delete a recoverable manifest after its vectors are removed."""

        with self._connection:
            status = self._connection.execute(
                "SELECT status FROM document_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if status is None:
                return set()
            if status["status"] not in {"building", "failed", "cleanup_pending"}:
                raise ValueError("Only recoverable document versions may be cleaned up.")
            chunk_ids = {
                row["chunk_id"]
                for row in self._connection.execute(
                    "SELECT chunk_id FROM chunks WHERE version_id = ?", (version_id,)
                )
            }
            self._delete_version_rows(version_id)
            self._connection.execute(
                "DELETE FROM document_versions WHERE version_id = ?", (version_id,)
            )
            return chunk_ids

    def search(
        self,
        query: str,
        limit: int,
        doc_type_filter: str | None = None,
    ) -> list[RankedHit]:
        """Search only active chunks, applying document-type filtering before limit."""

        if limit <= 0:
            return []
        match_query = self._match_query(query)
        if not match_query:
            return []
        parameters: list[Any] = [match_query]
        doc_type_sql = ""
        if doc_type_filter is not None:
            doc_type_sql = "AND json_extract(c.metadata_json, '$.doc_type') = ?"
            parameters.append(doc_type_filter)
        parameters.append(limit)
        rows = self._connection.execute(
            f"""
            SELECT c.chunk_id,
                   bm25(chunks_fts, 5.0, 3.0, 4.0, 1.0, 0.0) AS bm25_score
            FROM chunks_fts
            JOIN chunks AS c ON c.chunk_id = chunks_fts.chunk_id
            JOIN document_versions AS v ON v.version_id = c.version_id
            JOIN documents AS d
                ON d.doc_id = v.doc_id AND d.active_version_id = v.version_id
            WHERE chunks_fts MATCH ?
              AND v.status = 'ready'
              {doc_type_sql}
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return [
            RankedHit(
                chunk_id=row["chunk_id"], rank=rank, score=float(row["bm25_score"]), backend="bm25"
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def active_chunk_ids(self, chunk_ids: Iterable[str] | None = None) -> set[str]:
        """Return active IDs, optionally restricted to supplied candidate IDs."""

        query = """
            SELECT c.chunk_id FROM chunks AS c
            JOIN document_versions AS v ON v.version_id = c.version_id
            JOIN documents AS d
                ON d.doc_id = v.doc_id AND d.active_version_id = v.version_id
            WHERE v.status = 'ready'
        """
        if chunk_ids is None:
            return {row["chunk_id"] for row in self._connection.execute(query)}
        candidates = list(dict.fromkeys(chunk_ids))
        if not candidates:
            return set()
        placeholders = ", ".join("?" for _ in candidates)
        return {
            row["chunk_id"]
            for row in self._connection.execute(
                f"{query} AND c.chunk_id IN ({placeholders})", candidates
            )
        }

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, ChildChunk]:
        """Return canonical chunks by ID, preserving only requested records."""

        identifiers = list(dict.fromkeys(chunk_ids))
        if not identifiers:
            return {}
        placeholders = ", ".join("?" for _ in identifiers)
        rows = self._connection.execute(
            f"""
            SELECT chunk_id, parent_id, version_id, content, metadata_json, ordinal
            FROM chunks WHERE chunk_id IN ({placeholders})
            """,
            identifiers,
        )
        return {
            row["chunk_id"]: _child_from_row(row)
            for row in rows
        }

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        """Return one canonical parent expansion unit."""

        row = self._connection.execute(
            """
            SELECT parent_id, section_id, version_id, content, metadata_json
            FROM parents WHERE parent_id = ?
            """,
            (parent_id,),
        ).fetchone()
        if row is None:
            return None
        return ParentChunk(
            parent_id=row["parent_id"],
            section_id=row["section_id"],
            version_id=row["version_id"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"]),
        )

    def get_siblings(self, chunk_id: str) -> list[ChildChunk]:
        """Return all chunks under the queried chunk's parent in ordinal order."""

        rows = self._connection.execute(
            """
            SELECT chunk_id, parent_id, version_id, content, metadata_json, ordinal
            FROM chunks
            WHERE parent_id = (SELECT parent_id FROM chunks WHERE chunk_id = ?)
            ORDER BY ordinal
            """,
            (chunk_id,),
        )
        return [_child_from_row(row) for row in rows]

    def stats(self) -> dict[str, int]:
        """Return compact manifest and FTS counts for the service façade."""

        row = self._connection.execute(
            """
            SELECT
                (SELECT count(*) FROM documents) AS documents,
                (SELECT count(*) FROM document_versions WHERE status = 'ready') AS ready_versions,
                (SELECT count(*) FROM document_versions
                 WHERE status = 'cleanup_pending') AS cleanup_pending_versions,
                (SELECT count(*) FROM chunks) AS chunks,
                (SELECT count(*) FROM chunks_fts) AS fts_rows
            """
        ).fetchone()
        return dict(row)

    def _fts_row(self, document: SourceDocument, chunk: ChildChunk) -> tuple[str, str, str, str, str]:
        metadata = chunk.metadata
        title = str(metadata.get("title", document.title))
        section_path = str(metadata.get("section_path", ""))
        clause_number = str(metadata.get("clause_no") or "")
        return (
            _indexed_text(self._tokenizer, title),
            _indexed_text(self._tokenizer, section_path),
            _indexed_text(self._tokenizer, clause_number),
            _indexed_text(self._tokenizer, chunk.content),
            chunk.chunk_id,
        )

    def _match_query(self, query: str) -> str:
        terms = [*self._tokenizer.bm25_terms(query), *canonical_identifier_aliases(query)]
        unique_terms = list(dict.fromkeys(terms))
        return " OR ".join(_fts_quote(term) for term in unique_terms if term)

    def _validate_version_records(
        self,
        document: SourceDocument,
        parents: Sequence[ParentChunk],
        chunks: Sequence[ChildChunk],
    ) -> None:
        parent_ids = {parent.parent_id for parent in parents}
        if len(parent_ids) != len(parents):
            raise ValueError("Parent IDs must be unique within a staged version.")
        if any(parent.version_id != document.version_id for parent in parents):
            raise ValueError("Every parent must belong to the source document version.")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within a staged version.")
        if any(chunk.version_id != document.version_id for chunk in chunks):
            raise ValueError("Every chunk must belong to the source document version.")
        if any(chunk.parent_id not in parent_ids for chunk in chunks):
            raise ValueError("Every chunk parent_id must refer to a staged parent.")

    def _delete_version_rows(self, version_id: str) -> None:
        chunk_ids = [
            row["chunk_id"]
            for row in self._connection.execute(
                "SELECT chunk_id FROM chunks WHERE version_id = ?", (version_id,)
            )
        ]
        if chunk_ids:
            placeholders = ", ".join("?" for _ in chunk_ids)
            self._connection.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids
            )
        self._connection.execute("DELETE FROM chunks WHERE version_id = ?", (version_id,))
        self._connection.execute("DELETE FROM parents WHERE version_id = ?", (version_id,))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_identifier_aliases(text: str) -> list[str]:
    """Return narrowly scoped aliases for supported standards and Qwen models."""

    aliases: list[str] = []
    for match in _GB_STANDARD_RE.finditer(text):
        aliases.append("GB" + _strip_identifier_separators(match.group(1)))
    for match in _ASTM_STANDARD_RE.finditer(text):
        aliases.append(
            "ASTM"
            + match.group(1).upper()
            + _strip_identifier_separators(match.group(2))
        )
    for match in _QWEN_MODEL_RE.finditer(text):
        model = "QWEN" + _strip_identifier_separators(match.group(1))
        aliases.append(model)
        if match.group(2) is not None:
            aliases.append(model + "EMBEDDING")
    return list(dict.fromkeys(aliases))


def _strip_identifier_separators(value: str) -> str:
    return re.sub(r"[\s_/-]+", "", value).upper()


def _indexed_text(tokenizer: ChemicalTokenizer, text: str) -> str:
    return " ".join(
        [*tokenizer.bm25_terms(text), *canonical_identifier_aliases(text)]
    )


def _fts_quote(term: str) -> str:
    return f'"{term.replace(chr(34), chr(34) * 2)}"'


def _child_from_row(row: sqlite3.Row) -> ChildChunk:
    return ChildChunk(
        chunk_id=row["chunk_id"],
        parent_id=row["parent_id"],
        version_id=row["version_id"],
        content=row["content"],
        embedding_text="",
        ordinal=int(row["ordinal"]),
        metadata=json.loads(row["metadata_json"]),
    )

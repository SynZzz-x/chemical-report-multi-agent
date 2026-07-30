"""Hybrid BM25/dense retrieval, RRF fusion, and evidence context expansion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
import re
from typing import Any

from src.config import RAGSettings, get_rag_settings

from .bm25_store import BM25Store
from .embeddings import TEIEmbeddings
from .models import ChildChunk, ParentChunk, RankedHit
from .tokenizer import ChemicalTokenizer
from .vector_store import VectorStore


_DENSE_FETCH_ATTEMPTS = 3


@dataclass
class _FusedHit:
    """One child together with the rank-only data used for RRF."""

    chunk_id: str
    score: float = 0.0
    bm25_rank: int | None = None
    dense_rank: int | None = None

    @property
    def best_rank(self) -> int:
        return min(rank for rank in (self.bm25_rank, self.dense_rank) if rank is not None)


@dataclass
class _QueryState:
    """Per-query warnings and token-count availability."""

    warnings: list[str] = field(default_factory=list)
    tei_degraded: bool = False

    def warn(self, backend: str, exc: BaseException) -> None:
        warning = f"{backend}: {_sanitize_exception(exc)}"
        if warning not in self.warnings:
            self.warnings.append(warning)


class HybridRetriever:
    """Fuse lexical and dense child rankings into bounded parent evidence."""

    def __init__(
        self,
        bm25_store: BM25Store | None,
        vector_store: VectorStore | None,
        embeddings: TEIEmbeddings,
        tokenizer: ChemicalTokenizer,
        settings: RAGSettings | None = None,
        *,
        startup_warnings: Iterable[str] = (),
    ) -> None:
        self._bm25_store = bm25_store
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._tokenizer = tokenizer
        self._settings = settings or get_rag_settings()
        self._startup_warnings = list(dict.fromkeys(startup_warnings))

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        doc_type_filter: str | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Return source evidence, never treating RRF as calibrated relevance."""

        if not isinstance(question, str) or not question.strip():
            return self._error_result(question, ["query: question must be a non-empty string."])
        if top_k is not None and top_k <= 0:
            return self._error_result(
                question, ["query: top_k must be positive."]
            )
        state = _QueryState(warnings=list(self._startup_warnings))
        if similarity_threshold is not None:
            state.warnings.append(
                "query: similarity_threshold is deprecated and ignored because "
                "rank-only RRF scores are not calibrated similarities."
            )
        bm25_hits, bm25_available = self._bm25_hits(question, doc_type_filter, state)
        dense_hits, dense_available = self._dense_hits(question, doc_type_filter, state)
        retrieval_mode = _retrieval_mode(bm25_available, dense_available)
        if retrieval_mode is None:
            return self._error_result(question, state.warnings)

        fused = self._fuse(bm25_hits, dense_hits)
        if self._bm25_store is None:
            state.warnings.append("manifest: canonical context store is unavailable.")
            return self._error_result(question, state.warnings, retrieval_mode)
        try:
            chunks = self._bm25_store.get_chunks(item.chunk_id for item in fused)
        except Exception as exc:  # Canonical context is required for either backend.
            state.warn("manifest", exc)
            return self._error_result(question, state.warnings, retrieval_mode)
        selected = self._select_children(fused, chunks)
        if not selected:
            return self._empty_result(question, retrieval_mode, state.warnings)

        groups = self._group_by_parent(selected, chunks, top_k)
        results = self._expand_groups(groups, state)
        return {
            "success": True,
            "question": question,
            "retrieval_mode": retrieval_mode,
            "total_results": len(results),
            "warnings": state.warnings,
            "retrieval_available": True,
            "has_evidence": bool(results),
            "evidence_assessment_required": bool(results),
            "evidence_instruction": (
                "Use only the returned source evidence. If it does not directly "
                "support the answer, state that the knowledge base is insufficient."
                if results
                else "No supporting evidence was retrieved. Do not answer from "
                "general knowledge or invent knowledge-base conclusions."
            ),
            "results": results,
        }

    def _bm25_hits(
        self, question: str, doc_type_filter: str | None, state: _QueryState
    ) -> tuple[list[RankedHit], bool]:
        if self._bm25_store is None:
            state.warnings.append("bm25: backend is unavailable.")
            return [], False
        if not self._bm25_store.lexical_available:
            warning = "bm25: " + (
                self._bm25_store.lexical_warning
                or "SQLite FTS5 lexical search is unavailable."
            )
            if warning not in state.warnings:
                state.warnings.append(warning)
            return [], False
        try:
            return (
                self._bm25_store.search(
                    question, self._settings.bm25_top_k, doc_type_filter
                ),
                True,
            )
        except Exception as exc:
            state.warn("bm25", exc)
            return [], False

    def _dense_hits(
        self,
        question: str,
        doc_type_filter: str | None,
        state: _QueryState,
    ) -> tuple[list[RankedHit], bool]:
        if self._vector_store is None:
            state.warnings.append("dense: backend is unavailable.")
            return [], False
        if self._bm25_store is None:
            state.warnings.append("dense: active manifest validation is unavailable.")
            return [], False
        try:
            query_embedding = self._embeddings.embed_query(question)
        except Exception as exc:
            state.tei_degraded = True
            state.warn("dense", exc)
            return [], False

        initial_limit = max(
            self._settings.dense_top_k
            * max(self._settings.dense_overfetch_factor, 3),
            120,
        )
        dense_hits: list[RankedHit] = []
        try:
            for attempt in range(_DENSE_FETCH_ATTEMPTS):
                limit = initial_limit * (attempt + 1)
                candidates = self._vector_store.search(
                    query_embedding, limit, doc_type_filter
                )
                active_ids = self._bm25_store.active_chunk_ids(
                    hit.chunk_id for hit in candidates
                )
                dense_hits = [
                    hit for hit in candidates if hit.chunk_id in active_ids
                ][: self._settings.dense_top_k]
                if len(dense_hits) >= self._settings.dense_top_k or len(candidates) < limit:
                    break
        except Exception as exc:
            state.warn("dense", exc)
            return [], False
        return dense_hits, True

    def _fuse(
        self, bm25_hits: Iterable[RankedHit], dense_hits: Iterable[RankedHit]
    ) -> list[_FusedHit]:
        fused: dict[str, _FusedHit] = {}
        for hit in bm25_hits:
            item = fused.setdefault(hit.chunk_id, _FusedHit(hit.chunk_id))
            item.score += 1.0 / (self._settings.rrf_k + hit.rank)
            item.bm25_rank = hit.rank
        for hit in dense_hits:
            item = fused.setdefault(hit.chunk_id, _FusedHit(hit.chunk_id))
            item.score += 1.0 / (self._settings.rrf_k + hit.rank)
            item.dense_rank = hit.rank
        return sorted(fused.values(), key=lambda item: (-item.score, item.best_rank, item.chunk_id))

    def _select_children(
        self, fused: Iterable[_FusedHit], chunks: dict[str, ChildChunk]
    ) -> list[_FusedHit]:
        selected: list[_FusedHit] = []
        per_parent: dict[str, int] = defaultdict(int)
        for item in fused:
            chunk = chunks.get(item.chunk_id)
            if chunk is None or per_parent[chunk.parent_id] >= self._settings.max_hits_per_parent:
                continue
            selected.append(item)
            per_parent[chunk.parent_id] += 1
            if len(selected) >= self._settings.rrf_child_top_k:
                break
        return selected

    def _group_by_parent(
        self,
        selected: Iterable[_FusedHit],
        chunks: dict[str, ChildChunk],
        top_k: int | None,
    ) -> list[tuple[ParentChunk, list[_FusedHit]]]:
        if self._bm25_store is None:
            return []
        grouped: dict[str, list[_FusedHit]] = {}
        for item in selected:
            chunk = chunks.get(item.chunk_id)
            if chunk is not None:
                grouped.setdefault(chunk.parent_id, []).append(item)
        group_limit = min(
            5,
            self._settings.final_top_k,
            self._settings.final_top_k if top_k is None else top_k,
        )
        groups: list[tuple[ParentChunk, list[_FusedHit]]] = []
        for parent_id, matches in grouped.items():
            parent = self._bm25_store.get_parent(parent_id)
            if parent is not None:
                groups.append((parent, matches))
        groups.sort(
            key=lambda group: (
                -max(item.score for item in group[1]),
                min(item.best_rank for item in group[1]),
                group[0].parent_id,
            )
        )
        return groups[:group_limit]

    def _expand_groups(
        self,
        groups: Iterable[tuple[ParentChunk, list[_FusedHit]]],
        state: _QueryState,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        rendered_contents: list[str] = []
        for parent, matches in groups:
            parent_candidate = _join_content([*rendered_contents, parent.content])
            if self._count_tokens(parent_candidate, state) <= self._settings.max_context_tokens:
                content = parent.content
                included_matches = matches
            else:
                content, included_matches = self._expanded_children(
                    parent, matches, rendered_contents, state
                )
                if not content:
                    continue
            results.append(self._evidence_group(parent, included_matches, content))
            rendered_contents.append(content)
        return results

    def _expanded_children(
        self,
        parent: ParentChunk,
        matches: list[_FusedHit],
        rendered_contents: list[str],
        state: _QueryState,
    ) -> tuple[str, list[_FusedHit]]:
        if self._bm25_store is None:
            return "", []
        selected_ids = {item.chunk_id for item in matches}
        try:
            siblings = self._bm25_store.get_siblings(matches[0].chunk_id)
        except Exception as exc:
            state.warn("manifest", exc)
            return "", []
        siblings = [chunk for chunk in siblings if chunk.parent_id == parent.parent_id]
        selected_indexes = {
            index
            for index, sibling in enumerate(siblings)
            if sibling.chunk_id in selected_ids
        }
        candidate_indexes = set(selected_indexes)
        for index in selected_indexes:
            if index:
                candidate_indexes.add(index - 1)
            if index + 1 < len(siblings):
                candidate_indexes.add(index + 1)
        ordered_candidates = [
            siblings[index] for index in sorted(candidate_indexes)
        ]

        content = _collapse_ordered_chunks(ordered_candidates)
        if (
            content
            and self._count_tokens(
                _join_content([*rendered_contents, content]), state
            )
            <= self._settings.max_context_tokens
        ):
            return content, matches

        ordered_matches = [
            sibling for sibling in siblings if sibling.chunk_id in selected_ids
        ]
        content = _collapse_ordered_chunks(ordered_matches)
        if (
            content
            and self._count_tokens(
                _join_content([*rendered_contents, content]), state
            )
            <= self._settings.max_context_tokens
        ):
            return content, matches

        included_ids: set[str] = set()
        included_chunks: list[ChildChunk] = []
        content = ""
        for chunk in ordered_matches:
            trial_chunks = [*included_chunks, chunk]
            trial_content = _collapse_ordered_chunks(trial_chunks)
            candidate = _join_content([*rendered_contents, trial_content])
            if self._count_tokens(candidate, state) > self._settings.max_context_tokens:
                continue
            included_chunks = trial_chunks
            included_ids.add(chunk.chunk_id)
            content = trial_content
        included_matches = [
            item for item in matches if item.chunk_id in included_ids
        ]
        return content, included_matches

    def _count_tokens(self, text: str, state: _QueryState) -> int:
        if state.tei_degraded:
            return _fallback_token_count(text)
        try:
            return self._tokenizer.model_tokens(text)
        except Exception as exc:
            state.tei_degraded = True
            state.warn("tei", exc)
            return _fallback_token_count(text)

    @staticmethod
    def _evidence_group(
        parent: ParentChunk, matches: list[_FusedHit], content: str
    ) -> dict[str, Any]:
        metadata = parent.metadata
        return {
            "content": content,
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
            "doc_type": metadata.get("doc_type", ""),
            "section_path": metadata.get("section_path", ""),
            "pages": {
                "start": metadata.get("page_start"),
                "end": metadata.get("page_end"),
            },
            "chunk_ids": [item.chunk_id for item in matches],
            "parent_id": parent.parent_id,
            "parent_score": max(item.score for item in matches),
            "matches": [
                {
                    "chunk_id": item.chunk_id,
                    "rrf_score": item.score,
                    "bm25_rank": item.bm25_rank,
                    "dense_rank": item.dense_rank,
                }
                for item in matches
            ],
        }

    @staticmethod
    def _empty_result(question: str, mode: str, warnings: list[str]) -> dict[str, Any]:
        return {
            "success": True,
            "question": question,
            "retrieval_mode": mode,
            "total_results": 0,
            "warnings": warnings,
            "retrieval_available": mode != "unavailable",
            "has_evidence": False,
            "evidence_assessment_required": False,
            "evidence_instruction": (
                "No supporting evidence was retrieved. Do not answer from general "
                "knowledge or invent knowledge-base conclusions."
            ),
            "results": [],
        }

    @staticmethod
    def _error_result(
        question: Any, warnings: list[str], mode: str = "unavailable"
    ) -> dict[str, Any]:
        return {
            "success": False,
            "question": question,
            "retrieval_mode": mode,
            "total_results": 0,
            "warnings": warnings,
            "retrieval_available": False,
            "has_evidence": False,
            "evidence_assessment_required": False,
            "evidence_instruction": (
                "Knowledge-base retrieval is unavailable. Do not answer from general "
                "knowledge or invent knowledge-base conclusions."
            ),
            "results": [],
            "error": "No retrieval backend is available.",
        }


def _retrieval_mode(bm25_available: bool, dense_available: bool) -> str | None:
    if bm25_available and dense_available:
        return "hybrid"
    if bm25_available:
        return "bm25_only"
    if dense_available:
        return "dense_only"
    return None


def _fallback_token_count(text: str) -> int:
    """Return a deliberately conservative estimate when TEI is unavailable."""

    return max(1, len(text.encode("utf-8")))


def _join_content(parts: Iterable[str]) -> str:
    """Join exactly as evidence is rendered so separator tokens are budgeted."""

    return "\n\n".join(part for part in parts if part)


def _collapse_ordered_chunks(chunks: Iterable[ChildChunk]) -> str:
    """Render canonical ordinal order while removing adjacent copied overlap."""

    parts: list[str] = []
    previous = ""
    for chunk in sorted(chunks, key=lambda item: (item.ordinal, item.chunk_id)):
        content = chunk.content.strip()
        if not content:
            continue
        piece = _strip_adjacent_overlap(previous, content)
        if piece:
            parts.append(piece)
        previous = content
    return _join_content(parts)


def _strip_adjacent_overlap(previous: str, current: str) -> str:
    if not previous:
        return current
    previous_normalized = _normalized_overlap_text(previous)
    current_normalized = _normalized_overlap_text(current)
    if not current_normalized:
        return ""
    if current_normalized in previous_normalized:
        return ""
    if _overlap_signature(current_normalized) == _overlap_signature(
        previous_normalized
    ):
        return ""

    previous_lines = [line.strip() for line in previous.splitlines() if line.strip()]
    current_lines = [line.strip() for line in current.splitlines() if line.strip()]
    max_lines = min(len(previous_lines), len(current_lines))
    for size in range(max_lines, 0, -1):
        if previous_lines[-size:] == current_lines[:size]:
            return "\n".join(current_lines[size:]).strip()

    max_overlap = min(len(previous), len(current))
    for size in range(max_overlap, 23, -1):
        if previous[-size:] == current[:size]:
            return current[size:].lstrip()
    return current


def _normalized_overlap_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _overlap_signature(text: str) -> str:
    """Ignore formatting only; preserve every chemical word, number, and unit."""

    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def _sanitize_exception(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    return text[:240]

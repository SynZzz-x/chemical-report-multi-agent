"""Hybrid BM25/dense retrieval, RRF fusion, and evidence context expansion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
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
            return self._empty_result(question, "unavailable", ["query: top_k must be positive."])
        # Kept in the public signature for worker compatibility. RRF is rank-only,
        # so a raw dense similarity cannot decide whether evidence is sufficient.
        _ = similarity_threshold

        state = _QueryState(warnings=list(self._startup_warnings))
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
            "evidence_assessment_required": bool(results),
            "results": results,
        }

    def _bm25_hits(
        self, question: str, doc_type_filter: str | None, state: _QueryState
    ) -> tuple[list[RankedHit], bool]:
        if self._bm25_store is None:
            state.warnings.append("bm25: backend is unavailable.")
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
        by_id = {chunk.chunk_id: chunk for chunk in siblings}
        selected_order = [item.chunk_id for item in matches if item.chunk_id in by_id]
        parts: list[str] = []
        included_matches: list[_FusedHit] = []
        matches_by_id = {item.chunk_id: item for item in matches}
        for chunk_id in selected_order:
            chunk = by_id[chunk_id]
            candidate = _join_content([*rendered_contents, *parts, chunk.content])
            if self._count_tokens(candidate, state) > self._settings.max_context_tokens:
                continue
            parts.append(chunk.content)
            included_matches.append(matches_by_id[chunk_id])

        if not included_matches:
            return "", []

        ordered_ids: list[str] = []
        for index, sibling in enumerate(siblings):
            if sibling.chunk_id not in {item.chunk_id for item in included_matches}:
                continue
            for neighbor in (siblings[index - 1] if index else None, siblings[index + 1] if index + 1 < len(siblings) else None):
                if neighbor is not None and neighbor.parent_id == parent.parent_id:
                    ordered_ids.append(neighbor.chunk_id)
        for chunk_id in dict.fromkeys(ordered_ids):
            if chunk_id in selected_ids:
                continue
            chunk = by_id[chunk_id]
            candidate = _join_content([*rendered_contents, *parts, chunk.content])
            if self._count_tokens(candidate, state) > self._settings.max_context_tokens:
                continue
            parts.append(chunk.content)
        return _join_content(parts), included_matches

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
            "evidence_assessment_required": False,
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
            "evidence_assessment_required": False,
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


def _sanitize_exception(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    return text[:240]

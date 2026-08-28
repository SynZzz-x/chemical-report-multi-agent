"""Deterministic invocation-local extraction of cited report claims."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .citations import extract_inline_evidence_ids
from .projection import collect_used_evidence_ids
from .text_projection import semantic_evidence_excerpt


_PARAGRAPH_BOUNDARY = re.compile(r"(?:\r?\n\s*){2,}")
_SENTENCE = re.compile(r".*?[。！？!?](?:\s*\[[^\]]+\])*|.+$", re.DOTALL)
_CLAIM_TEXT_LIMIT = 1000
_EVIDENCE_GAP_MARKERS = (
    "未检出",
    "未找到",
    "未发现",
    "缺乏",
    "没有直接证据",
    "无直接证据",
    "no direct evidence",
    "no evidence",
    "not found",
)
_RECOMMENDATION_MARKERS = (
    "工程上可考虑",
    "可考虑",
    "建议",
    "推荐",
    "应考虑",
)
_INFERENCE_MARKERS = (
    "据此可推测",
    "综合来看",
    "可推测",
    "推断",
    "可能表明",
    "由此可见",
)


def evidence_gap_wording(statement: str) -> bool:
    """Return whether a statement explicitly discloses an evidence gap."""

    text = statement.casefold()
    return any(marker in text for marker in _EVIDENCE_GAP_MARKERS)


def recommendation_wording(statement: str) -> bool:
    """Return whether a statement gives explicit engineering advice."""

    return any(marker in statement for marker in _RECOMMENDATION_MARKERS)


def inference_wording(statement: str) -> bool:
    """Return whether a statement expressly presents an inference."""

    text = statement.casefold()
    return any(marker in text for marker in _INFERENCE_MARKERS)


def _statements(content: str) -> list[str]:
    statements: list[str] = []
    for paragraph in _PARAGRAPH_BOUNDARY.split(str(content or "")):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        statements.extend(
            match.group(0).strip() for match in _SENTENCE.finditer(paragraph) if match.group(0).strip()
        )
    return statements


def _citation_index(citations: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for citation in citations:
        if not isinstance(citation, Mapping):
            continue
        evidence_id = str(citation.get("evidence_id") or "").strip().upper()
        if evidence_id:
            index.setdefault(evidence_id, citation)
    return index


def _claim_type(statement: str) -> str:
    if evidence_gap_wording(statement):
        return "evidence_gap"
    if recommendation_wording(statement):
        return "recommendation"
    if inference_wording(statement):
        return "inference"
    return "factual"


def derive_claims(content: str, citations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project cited claims and explicit evidence-gap disclosures for one call."""

    citation_index = _citation_index(citations)
    claims: list[dict[str, Any]] = []
    for statement in _statements(content):
        has_citation = bool(extract_inline_evidence_ids(statement))
        if not has_citation and not evidence_gap_wording(statement):
            continue

        evidence_ids = collect_used_evidence_ids(statement, citations)
        evidence = [
            {
                "evidence_id": evidence_id,
                "title": str(citation_index[evidence_id].get("title") or ""),
                "locator": str(citation_index[evidence_id].get("locator") or ""),
                "semantic_evidence_excerpt": semantic_evidence_excerpt(citation_index[evidence_id]),
            }
            for evidence_id in evidence_ids
        ]
        claims.append(
            {
                "claim_id": f"C{len(claims) + 1}",
                "text": semantic_evidence_excerpt(statement, limit=_CLAIM_TEXT_LIMIT),
                "claim_type": _claim_type(statement),
                "evidence_ids": evidence_ids,
                "evidence": evidence,
            }
        )
    return claims

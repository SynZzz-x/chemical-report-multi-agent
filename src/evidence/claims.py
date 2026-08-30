"""Deterministic invocation-local extraction of cited report claims."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .citations import extract_inline_evidence_ids
from .projection import collect_used_evidence_ids
from .text_projection import semantic_evidence_excerpt


_PARAGRAPH_BOUNDARY = re.compile(r"(?:\r?\n\s*){2,}")
_ASCII_SENTENCE_END = r"\.(?=\s*(?:\[[^\]\r\n]+\]|[\u3400-\u9fff]|$))"
_SENTENCE = re.compile(
    rf".*?(?:[。！？!?]|{_ASCII_SENTENCE_END})(?:\s*\[[^\]]+\])*|.+$",
    re.DOTALL,
)
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
_FENCED_CODE_BLOCK = re.compile(
    r"(?ms)^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=fence)[^\n]*(?:\n|$)"
)
_INLINE_CODE = re.compile(
    r"(?s)(?P<delimiter>`{2,})(?:(?!(?P=delimiter)).)*(?P=delimiter)|`[^`\r\n]*`"
)
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_TABLE_LINE = re.compile(r"^\s*\|")
_LABEL_ONLY = re.compile(r"^\s*(?:[-*+]\s*)?[^。！？!?\n:：]{1,40}[：:]\s*$")
_MATERIAL_DOMAIN = re.compile(
    r"(?:氢气|乙烯|共聚单体|反应压力|反应温度|催化剂|链转移|"
    r"MFR|MWD|Mw/Mn|熔融指数|熔指|密度|分子量|聚合)",
    re.IGNORECASE,
)
_NOMINAL_MATERIAL_CONSTRUCTION = re.compile(
    r"(?:最直接(?:、最灵敏)?的调节手段|最灵敏的调节手段|核心控制变量|关键决定因素)"
)
_NOMINAL_ASSERTIVE_FORM = re.compile(r"(?:是|为|属于|构成).{0,60}")
_CAUSAL_MATERIAL_CONSTRUCTION = re.compile(
    r"(?:主要由.{1,40}决定|直接决定|导致|使.{0,30}(?:升高|降低|提高|下降|扩大|缩小))"
)
_INFERENCE_MATERIAL_CONSTRUCTION = re.compile(
    r"(?:会.{0,12}(?:显著)?(?:提高|降低|升高|下降|增加|减少|扩大|缩小)|"
    r"显著(?:提高|降低|升高|下降|增加|减少|扩大|缩小))"
)
_ASSERTIVE_PREDICATE = re.compile(
    r"(?:是|为|决定|控制|导致|使|提高|降低|升高|下降|增加|减少|扩大|缩小|"
    r"会.{0,12}(?:显著)?(?:提高|降低|升高|下降|增加|减少|扩大|缩小)|"
    r"显著(?:提高|降低|升高|下降|增加|减少|扩大|缩小))"
)
_QUANTITATIVE_VALUE = re.compile(
    r"(?<![\d.])\d+(?:\.\d+)?\s*(?:%|％|ppm|ppb|℃|°C|K|Pa|kPa|MPa|bar|"
    r"mol/mol|kg/h|t/h|g/min|mg/L|g/L|h|小时|min|分钟)(?=$|[^\w])",
    re.IGNORECASE,
)
_QUANTITATIVE_ASSERTION = re.compile(
    r"(?:>=|<=|≥|≤|大于|小于|高于|低于|不超过|不少于|控制在|设定为|"
    r"保持在|达到|提高|降低|升高|下降|增加|减少|导致|使)"
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


def _mask_non_prose_content(content: str) -> str:
    """Remove code and Markdown-only lines before sentence splitting."""

    text = _INLINE_CODE.sub("", _FENCED_CODE_BLOCK.sub("", str(content or "")))
    lines = text.splitlines(keepends=True)
    prose_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            index + 1 < len(lines)
            and "|" in line
            and _TABLE_SEPARATOR.match(lines[index + 1].strip())
        ):
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                index += 1
            continue
        stripped = line.strip()
        if (
            _MARKDOWN_HEADING.match(stripped)
            or _TABLE_LINE.match(stripped)
            or _TABLE_SEPARATOR.match(stripped)
            or _LABEL_ONLY.match(stripped)
        ):
            index += 1
            continue
        prose_lines.append(line)
        index += 1
    return "".join(prose_lines)


def _non_prose_statement(statement: str) -> bool:
    """Return whether a statement is Markdown structure rather than report prose."""

    text = statement.strip()
    return bool(
        not text
        or "```" in text
        or "~~~" in text
        or _MARKDOWN_HEADING.match(text)
        or _TABLE_ROW.match(text)
        or _TABLE_SEPARATOR.match(text)
        or _LABEL_ONLY.match(text)
    )


def _requires_material_citation(statement: str) -> bool:
    """Recognize only high-confidence material assertions, never evidence support."""

    if not _MATERIAL_DOMAIN.search(statement):
        return False
    if _CAUSAL_MATERIAL_CONSTRUCTION.search(statement):
        return True
    if _NOMINAL_MATERIAL_CONSTRUCTION.search(
        statement
    ) and _NOMINAL_ASSERTIVE_FORM.search(statement):
        return True
    if inference_wording(statement) and _INFERENCE_MATERIAL_CONSTRUCTION.search(
        statement
    ):
        return True
    return bool(
        _QUANTITATIVE_VALUE.search(statement)
        and _QUANTITATIVE_ASSERTION.search(statement)
        and _ASSERTIVE_PREDICATE.search(statement)
    )


def find_uncited_material_claims(content: str) -> list[dict[str, str]]:
    """Find uncited high-confidence material assertions without judging support."""

    findings: list[dict[str, str]] = []
    prose_content = _mask_non_prose_content(content)
    for statement in _statements(prose_content):
        if extract_inline_evidence_ids(statement):
            continue
        if _non_prose_statement(statement) or evidence_gap_wording(statement):
            continue
        claim_type = "inference" if inference_wording(statement) else "factual"
        if claim_type == "factual" and recommendation_wording(statement):
            continue
        if _requires_material_citation(statement):
            findings.append(
                {
                    "text": semantic_evidence_excerpt(
                        statement, limit=_CLAIM_TEXT_LIMIT
                    ),
                    "claim_type": claim_type,
                }
            )
    return findings


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

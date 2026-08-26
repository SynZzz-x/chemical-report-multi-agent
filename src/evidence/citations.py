"""Deterministic parsing and normalization for inline evidence citations."""

from __future__ import annotations

import re
from collections.abc import Collection


_CITATION_GROUP = re.compile(
    r"\[\s*E\s*\d+(?:\s*[,，;；、]\s*E?\s*\d+)*\s*\]",
    re.IGNORECASE,
)
_EVIDENCE_NUMBER = re.compile(r"\d+")
_ADJACENT_CITATIONS = re.compile(r"(?:\s*\[E\d+\]){2,}", re.IGNORECASE)


def extract_inline_evidence_ids(content: str) -> set[str]:
    """Return every evidence ID from supported simple or compound citations."""

    ids: set[str] = set()
    for group in _CITATION_GROUP.finditer(str(content or "")):
        ids.update(
            f"E{value}" for value in _EVIDENCE_NUMBER.findall(group.group(0))
        )
    return ids


def normalize_inline_citations(
    content: str,
    known_ids: Collection[str],
) -> tuple[str, set[str], set[str]]:
    """Canonicalize citations and remove IDs outside the task evidence registry."""

    known = {str(value).strip().upper() for value in known_ids}
    found = extract_inline_evidence_ids(content)

    def normalize_group(match: re.Match[str]) -> str:
        group_ids = {
            f"E{value}"
            for value in _EVIDENCE_NUMBER.findall(match.group(0))
            if f"E{value}" in known
        }
        return "".join(
            f"[{value}]" for value in sorted(group_ids, key=lambda item: int(item[1:]))
        )

    normalized = _CITATION_GROUP.sub(normalize_group, str(content or ""))

    def normalize_run(match: re.Match[str]) -> str:
        group_ids = extract_inline_evidence_ids(match.group(0))
        return "".join(
            f"[{value}]" for value in sorted(group_ids, key=lambda item: int(item[1:]))
        )

    normalized = _ADJACENT_CITATIONS.sub(normalize_run, normalized)
    cited = found & known
    unknown = found - known
    return normalized, cited, unknown

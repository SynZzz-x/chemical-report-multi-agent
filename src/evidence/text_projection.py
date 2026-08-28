"""Pure, bounded text projections for evidence payloads and displays."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_WHITESPACE = re.compile(r"\s+")
_REPEATED_TABLE_SEPARATORS = re.compile(r"\|(?:\s*\|){2,}")
_SEMANTIC_ELLIPSIS = " … "


def _evidence_value(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    for field in ("supporting_text", "content", "text"):
        if value.get(field) is not None:
            return value[field]
    return ""


def normalize_evidence_text(value: Any) -> str:
    """Return a display-safe text copy without changing the supplied value."""

    text = str(_evidence_value(value) or "")
    text = _WHITESPACE.sub(" ", text).strip()
    return _REPEATED_TABLE_SEPARATORS.sub("|", text)


def semantic_evidence_excerpt(value: Any, limit: int = 4000) -> str:
    """Return a wide evidence excerpt that retains both ends when shortened."""

    text = normalize_evidence_text(value)
    limit = max(int(limit), 0)
    if len(text) <= limit:
        return text
    if limit <= len(_SEMANTIC_ELLIPSIS):
        return text[:limit]
    remaining = limit - len(_SEMANTIC_ELLIPSIS)
    head_length = remaining // 2
    return f"{text[:head_length]}{_SEMANTIC_ELLIPSIS}{text[-(remaining - head_length):]}"


def presentation_evidence_excerpt(value: Any, limit: int = 240) -> str:
    """Return a compact beginning-only excerpt for human-facing presentation."""

    return normalize_evidence_text(value)[: max(int(limit), 0)]

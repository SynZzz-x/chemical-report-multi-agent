"""Pure textual identity for deterministic RAG query deduplication."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def normalize_query_identity(query: Any) -> str:
    """Return a case-insensitive, whitespace-collapsed textual identity."""

    return re.sub(r"\s+", " ", str(query or "").strip().casefold())


def query_fingerprint(query: Any) -> str:
    """Hash only the normalized textual identity for compact observability."""

    identity = normalize_query_identity(query)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]

"""Deterministic projection from inline evidence markers to report sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ntpath
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_BRACKETED_EVIDENCE_GROUP = re.compile(r"\[([^\]]*\bE\d+[^\]]*)\]", re.IGNORECASE)
_EVIDENCE_MEMBER = re.compile(r"(?:\bE\s*)?(\d+)\b", re.IGNORECASE)


def collect_used_evidence_ids(
    content: str, citations: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return known inline IDs in stable first-appearance order."""

    known = {
        str(citation.get("evidence_id") or "").strip().upper()
        for citation in citations
        if isinstance(citation, Mapping) and citation.get("evidence_id")
    }
    used: list[str] = []
    seen: set[str] = set()
    for group in _BRACKETED_EVIDENCE_GROUP.finditer(str(content or "")):
        inherited_prefix = False
        for raw_member in re.split(r"[,，、;；]", group.group(1)):
            member = raw_member.strip()
            if re.search(r"\bE\s*\d+\b", member, re.IGNORECASE):
                inherited_prefix = True
            elif not inherited_prefix:
                continue
            match = _EVIDENCE_MEMBER.fullmatch(member)
            if match is None:
                continue
            evidence_id = f"E{int(match.group(1))}"
            if evidence_id in known and evidence_id not in seen:
                seen.add(evidence_id)
                used.append(evidence_id)
    return used


def project_used_citations(
    content: str, citations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Project registry records that are actually cited by the committed text."""

    by_id = {
        str(citation.get("evidence_id") or "").strip().upper(): dict(citation)
        for citation in citations
        if isinstance(citation, Mapping) and citation.get("evidence_id")
    }
    return [by_id[evidence_id] for evidence_id in collect_used_evidence_ids(content, citations)]


def canonical_source_identity(citation: Mapping[str, Any]) -> str:
    """Return the stable authority key for one evidence source."""

    source_type = str(citation.get("source_type") or "").strip().casefold()
    if source_type == "web":
        raw_url = str(citation.get("url") or "").strip()
        if raw_url:
            parts = urlsplit(raw_url)
            return urlunsplit(
                (parts.scheme.casefold(), parts.netloc.casefold(), parts.path, parts.query, "")
            )
    candidate = str(
        citation.get("file_path")
        or citation.get("file_name")
        or citation.get("title")
        or citation.get("url")
        or ""
    ).strip()
    return ntpath.basename(candidate.rstrip("/\\")) or candidate


def project_report_sources(
    content: str, citations: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return deduplicated source identities ordered by first inline citation."""

    projected: list[str] = []
    seen: set[str] = set()
    for citation in project_used_citations(content, citations):
        identity = canonical_source_identity(citation)
        key = identity.casefold()
        if identity and key not in seen:
            seen.add(key)
            projected.append(identity)
    return projected

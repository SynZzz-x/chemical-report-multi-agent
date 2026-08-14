"""Task-scoped evidence identity and deterministic report display IDs."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence


_EVIDENCE_MARKER = re.compile(r"\[(E\d+)\]", re.IGNORECASE)


def evidence_key(task_id: str, local_id: str) -> str:
    """Return the serializable internal identity for one task-local citation."""

    task = str(task_id or "").strip()
    evidence_id = str(local_id or "").strip().upper()
    if not task or not evidence_id:
        return ""
    return f"{task}:{evidence_id}"


def _citation_key(task_id: str, citation: Mapping[str, Any]) -> str:
    existing = str(citation.get("evidence_key") or "").strip()
    if existing:
        return existing
    local_id = str(
        citation.get("local_evidence_id") or citation.get("evidence_id") or ""
    ).strip()
    return evidence_key(task_id, local_id)


def build_display_evidence_map(
    sections: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Freeze global display IDs by section and citation first appearance."""

    display_map: dict[str, str] = {}
    for section in sections:
        task_id = str(section.get("task_id") or "").strip()
        for citation in section.get("citations") or []:
            if not isinstance(citation, Mapping):
                continue
            key = _citation_key(task_id, citation)
            if key and key not in display_map:
                display_map[key] = f"E{len(display_map) + 1}"
    return display_map


def _rewrite_markers(text: str, local_display: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        local_id = match.group(1).upper()
        return f"[{local_display.get(local_id, local_id)}]"

    return _EVIDENCE_MARKER.sub(replace, str(text or ""))


def _rewrite_nested(value: Any, local_display: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _rewrite_markers(value, local_display)
    if isinstance(value, list):
        return [_rewrite_nested(item, local_display) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_nested(item, local_display) for item in value)
    if isinstance(value, Mapping):
        rewritten: dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip().casefold()
            if normalized_key == "evidence_id" and isinstance(item, str):
                rewritten[key] = local_display.get(item.upper(), item.upper())
            elif normalized_key == "evidence_ids" and isinstance(item, (list, tuple)):
                rewritten[key] = [
                    local_display.get(str(evidence_id).upper(), str(evidence_id).upper())
                    for evidence_id in item
                ]
            else:
                rewritten[key] = _rewrite_nested(item, local_display)
        return rewritten
    return value


def normalize_sections_evidence(
    sections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return report-only copies with stable global evidence display IDs."""

    display_map = build_display_evidence_map(sections)
    normalized: list[dict[str, Any]] = []
    for original in sections:
        section = deepcopy(dict(original))
        task_id = str(section.get("task_id") or "").strip()
        citations: list[dict[str, Any]] = []
        local_display: dict[str, str] = {}
        for raw_citation in section.get("citations") or []:
            if not isinstance(raw_citation, Mapping):
                continue
            citation = dict(raw_citation)
            local_id = str(citation.get("evidence_id") or "").strip().upper()
            key = _citation_key(task_id, citation)
            display_id = display_map.get(key, local_id)
            if local_id and display_id:
                local_display[local_id] = display_id
            citation["local_evidence_id"] = str(
                citation.get("local_evidence_id") or local_id
            ).upper()
            citation["evidence_key"] = key
            citation["evidence_id"] = display_id
            citations.append(citation)
        section["citations"] = citations
        for field in ("text", "content", "text_output", "tables", "figures", "graph_spec"):
            if field in section:
                section[field] = _rewrite_nested(section[field], local_display)
        for figure in section.get("figures") or []:
            if not isinstance(figure, dict):
                continue
            figure["evidence_ids"] = [
                local_display.get(str(value).upper(), str(value).upper())
                for value in figure.get("evidence_ids") or []
            ]
        normalized.append(section)
    return normalized, display_map

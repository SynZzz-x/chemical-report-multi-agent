"""Task-scoped evidence identity and deterministic report display IDs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import posixpath
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .citations import extract_inline_evidence_ids
from .projection import collect_used_evidence_ids


_EVIDENCE_GROUP = re.compile(r"\[([^\]]*\bE\d+[^\]]*)\]", re.IGNORECASE)
_STABLE_SOURCE_ID_FIELDS = (
    "file_id",
    "resource_id",
    "source_id",
    "document_id",
    "doc_id",
)


def _canonical_url(value: Any) -> str:
    raw_url = str(value or "").strip()
    if not raw_url:
        return ""
    parts = urlsplit(raw_url)
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), parts.path, parts.query, "")
    )


def _normalized_source_path(value: Any) -> str:
    path = str(value or "").strip()
    if not path:
        return ""
    return posixpath.normpath(path.replace("\\", "/"))


def _is_presentation_basename(identity: str, full_path: str) -> bool:
    """Return whether an explicit identity is only the available path's label."""

    return bool(full_path) and identity.casefold() == posixpath.basename(
        full_path
    ).casefold()


def canonical_citation_identity(citation: Mapping[str, Any]) -> str:
    """Return the lossless correctness identity for one raw citation."""

    stable_id = next(
        (
            str(citation.get(field) or "").strip()
            for field in _STABLE_SOURCE_ID_FIELDS
            if str(citation.get(field) or "").strip()
        ),
        "",
    )
    canonical_url = _canonical_url(citation.get("url"))
    full_path = _normalized_source_path(citation.get("file_path"))
    explicit_canonical = str(
        citation.get("canonical_source_id") or citation.get("source_identity") or ""
    ).strip().casefold()
    title_fallback = str(citation.get("title") or "").strip().casefold()
    authority = (
        f"stable:{stable_id}"
        if stable_id
        else f"url:{canonical_url}"
        if canonical_url
        else f"canonical:{explicit_canonical}"
        if explicit_canonical
        and not _is_presentation_basename(explicit_canonical, full_path)
        else f"path:{full_path}"
        if full_path
        else f"fallback:{title_fallback}"
    )
    chunk_ids = tuple(str(value) for value in citation.get("chunk_ids") or ())
    evidence_identity = (
        {"chunk_ids": chunk_ids}
        if chunk_ids
        else {
            "supporting_text_sha256": hashlib.sha256(
                str(citation.get("supporting_text") or "").encode("utf-8")
            ).hexdigest()
        }
    )
    payload = {
        "source_type": str(citation.get("source_type") or "").casefold(),
        "authority": authority,
        "locator": str(citation.get("locator") or "").strip(),
        **evidence_identity,
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def evidence_key(task_id: str, local_id: str) -> str:
    """Return the serializable internal identity for one task-local citation."""

    task = str(task_id or "").strip()
    evidence_id = str(local_id or "").strip().upper()
    if not task or not evidence_id:
        return ""
    return f"{task}:{evidence_id}"


def citation_binding_key(task_id: str, citation: Mapping[str, Any]) -> str:
    """Preserve inherited task scope, otherwise bind the current local ID."""

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
        raw_citations = [
            citation
            for citation in section.get("citations") or []
            if isinstance(citation, Mapping)
        ]
        cited_order = collect_used_evidence_ids(
            str(
                section.get("text")
                or section.get("content")
                or section.get("text_output")
                or ""
            ),
            raw_citations,
        )
        for evidence_id in cited_order:
            display_by_identity: dict[str, str] = {}
            for citation in raw_citations:
                local_id = str(citation.get("evidence_id") or "").strip().upper()
                if local_id != evidence_id:
                    continue
                key = citation_binding_key(task_id, citation)
                if not key:
                    continue
                identity = canonical_citation_identity(citation)
                shared_display = display_by_identity.get(identity)
                if shared_display is not None:
                    previous_display = display_map.get(key)
                    if previous_display and previous_display != shared_display:
                        # A same-visible canonical duplicate can join two
                        # inherited keys that already appeared in prior tasks.
                        first, other = sorted(
                            (previous_display, shared_display),
                            key=lambda value: int(value[1:]),
                        )
                        display_map = {
                            bound_key: first if value == other else value
                            for bound_key, value in display_map.items()
                        }
                        shared_display = first
                    display_map[key] = shared_display
                elif key not in display_map:
                    display_map[key] = f"E{len(display_map) + 1}"
                display_by_identity[identity] = display_map[key]
    # Coalescing aliases must not leave gaps or allocate by alias-key count.
    compact_ids = {
        value: f"E{index}"
        for index, value in enumerate(dict.fromkeys(display_map.values()), start=1)
    }
    return {key: compact_ids[value] for key, value in display_map.items()}


def _rewrite_markers(text: str, local_display: Mapping[str, str]) -> str:
    def replace_group(match: re.Match[str]) -> str:
        rewritten: list[str] = []
        inherited_prefix = False
        for raw_member in re.split(r"([,，、;；])", match.group(1)):
            member = raw_member.strip()
            if not member or re.fullmatch(r"[,，、;；]", member):
                if member:
                    rewritten.append(member)
                continue
            explicit = re.fullmatch(r"E\s*(\d+)", member, re.IGNORECASE)
            inherited = re.fullmatch(r"(\d+)", member) if inherited_prefix else None
            identifier_match = explicit or inherited
            if identifier_match is None:
                rewritten.append(member)
                continue
            inherited_prefix = True
            local_id = f"E{int(identifier_match.group(1))}"
            rewritten.append(local_display.get(local_id, local_id))
        return "[" + " ".join(rewritten).replace(" ,", ",").replace(" ;", ";") + "]"

    return _EVIDENCE_GROUP.sub(replace_group, str(text or ""))


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
    # Keep every raw entry, but never let an unused local ID alias a used
    # display ID. This private extension does not change the public used map.
    all_display_ids = dict(display_map)
    reserved_ids = set(display_map.values())
    for section in sections:
        reserved_ids.update(
            extract_inline_evidence_ids(str(
                section.get("text") or section.get("content")
                or section.get("text_output") or ""
            ))
        )
    next_unused_id = len(set(display_map.values())) + 1
    for section in sections:
        task_id = str(section.get("task_id") or "").strip()
        for citation in section.get("citations") or []:
            if not isinstance(citation, Mapping):
                continue
            key = citation_binding_key(task_id, citation)
            if key and key not in all_display_ids:
                while f"E{next_unused_id}" in reserved_ids:
                    next_unused_id += 1
                all_display_ids[key] = f"E{next_unused_id}"
                reserved_ids.add(f"E{next_unused_id}")
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
            key = citation_binding_key(task_id, citation)
            display_id = all_display_ids.get(key, local_id)
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

"""Deterministic requirement identity, severity, and legacy reconstruction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .failure_semantics import RequirementRecord


_HARD_MARKERS = (
    "必须",
    "不得",
    "禁止",
    "不可",
    "不能",
    "务必",
    "缺口不可接受",
    "任何缺口都不可接受",
)
_CLAUSE_SPLIT = re.compile(r"[，。；;,.!?！？\n]+")


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _authoritative_clause(raw_request: str, requirement: str) -> str:
    requirement_key = _canonical(requirement)
    if not requirement_key:
        return ""
    for clause in _CLAUSE_SPLIT.split(str(raw_request or "")):
        if requirement_key in _canonical(clause):
            return clause.strip()
    return ""


def _is_explicit_hard(requirement: str, raw_request: str) -> bool:
    candidate = _text(requirement)
    clause = _authoritative_clause(raw_request, candidate)
    authority = clause or candidate
    return bool(clause) and any(marker in authority for marker in _HARD_MARKERS)


def _record(
    *,
    requirement_id: str,
    text: str,
    severity: str,
    kind: str,
    origin: str,
    source_message_id: str | None,
    source_field: str,
    source_index: int,
    derivation: str,
) -> RequirementRecord:
    return {
        "requirement_id": requirement_id,
        "text": text,
        "severity": severity,  # type: ignore[typeddict-item]
        "kind": kind,
        "status": "active",
        "contract_revision": 1,
        "provenance": {
            "origin": origin,  # type: ignore[typeddict-item]
            "source_message_id": source_message_id,
            "source_field": source_field,
            "source_index": source_index,
            "derivation": derivation,
        },
    }


def build_requirement_registry(
    parsed_request: Mapping[str, Any],
    raw_request: str,
    source_message_id: str | None,
) -> list[RequirementRecord]:
    """Build stable requirements without allowing Planner to invent authority."""

    candidates: list[tuple[str, str, int]] = []
    for field in ("constraints",):
        values = parsed_request.get(field)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            candidates.extend(
                (_text(value), field, index)
                for index, value in enumerate(values)
                if _text(value)
            )

    doc_length = _text(parsed_request.get("doc_length"))
    if doc_length and doc_length != "不限":
        candidates.append((doc_length, "doc_length", 0))

    sections = parsed_request.get("sections")
    if isinstance(sections, Sequence) and not isinstance(sections, (str, bytes)):
        candidates.extend(
            (_text(value), "sections", index)
            for index, value in enumerate(sections)
            if _text(value) and _authoritative_clause(raw_request, _text(value))
        )

    records: list[RequirementRecord] = []
    seen: set[str] = set()
    for candidate, source_field, source_index in candidates:
        key = _canonical(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        explicit = bool(_authoritative_clause(raw_request, candidate))
        hard = _is_explicit_hard(candidate, raw_request)
        records.append(
            _record(
                requirement_id=f"REQ-{len(records) + 1:03d}",
                text=candidate,
                severity="hard" if hard else "soft",
                kind=("length" if source_field == "doc_length" else "constraint"),
                origin="explicit_user" if explicit else "intake_inferred",
                source_message_id=source_message_id,
                source_field=source_field,
                source_index=source_index,
                derivation=(
                    "deterministic_explicit_hard_marker"
                    if hard
                    else "deterministic_soft_default"
                ),
            )
        )
    return records


def _message_payload(message: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(message, Mapping):
        content = message.get("content")
        message_id = message.get("id")
    else:
        content = getattr(message, "content", None)
        message_id = getattr(message, "id", None)
    try:
        payload = json.loads(str(content or ""))
    except (TypeError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict) or payload.get("type") != "INTAKE_SUMMARY":
        return None, None
    return payload, _text(message_id) or None


def reconstruct_legacy_requirements(
    state: Mapping[str, Any],
) -> list[RequirementRecord]:
    """Recover only authoritative legacy contract facts, never issue prose."""

    for message in reversed(list(state.get("messages") or [])):
        payload, message_id = _message_payload(message)
        if payload is None:
            continue
        raw_authority = "\n".join(
            _text(value) for value in payload.get("constraints") or [] if _text(value)
        )
        records = build_requirement_registry(payload, raw_authority, message_id)
        recovered: list[RequirementRecord] = []
        for index, record in enumerate(records, start=1):
            migrated = deepcopy(record)
            migrated["requirement_id"] = f"LEGACY-REQ-{index:03d}"
            migrated["provenance"]["origin"] = "legacy_reconstructed"
            recovered.append(migrated)
        return recovered

    recovered = []
    for task in state.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        task_id = _text(task.get("task_id")) or "unknown"
        for resource_index, resource in enumerate(task.get("use_resources") or []):
            resource_name = _text(resource)
            if not resource_name:
                continue
            recovered.append(
                _record(
                    requirement_id=f"LEGACY-TASK-{len(recovered) + 1:03d}",
                    text=f"任务 {task_id} 必须使用资源 {resource_name}",
                    severity="hard",
                    kind="resource",
                    origin="legacy_task_contract",
                    source_message_id=None,
                    source_field="use_resources",
                    source_index=resource_index,
                    derivation="deterministic_legacy_structural_field",
                )
            )
        if task.get("generate_figure") is True:
            recovered.append(
                _record(
                    requirement_id=f"LEGACY-TASK-{len(recovered) + 1:03d}",
                    text=f"任务 {task_id} 要求生成图形",
                    severity="hard",
                    kind="asset",
                    origin="legacy_task_contract",
                    source_message_id=None,
                    source_field="generate_figure",
                    source_index=0,
                    derivation="deterministic_legacy_structural_field",
                )
            )
    return recovered

"""Pure identity and upsert helpers for failure-semantics registries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

from .failure_semantics import DegradedIssueRecord, FailureClass


def _stable_strings(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            text
            for value in values
            if (text := str(value or "").strip())
        }
    )


def degradation_id(
    *,
    task_id: str,
    task_revision: int,
    subtype: str,
    affected_requirement_ids: Iterable[Any],
) -> str:
    identity = {
        "task_id": str(task_id),
        "task_revision": int(task_revision),
        "subtype": str(subtype).upper(),
        "affected_requirement_ids": _stable_strings(affected_requirement_ids),
    }
    canonical = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return "degraded-" + sha256(canonical.encode("utf-8")).hexdigest()


def build_degraded_issue(
    *,
    task_id: str,
    task_revision: int,
    subtype: str,
    affected_claims: Iterable[Any],
    affected_requirement_ids: Iterable[Any],
    attempted_repairs: list[dict[str, Any]],
    final_fallback: str,
    metadata: Mapping[str, Any] | None = None,
) -> DegradedIssueRecord:
    requirement_ids = _stable_strings(affected_requirement_ids)
    normalized_subtype = str(subtype).upper()
    return {
        "issue_id": degradation_id(
            task_id=task_id,
            task_revision=task_revision,
            subtype=normalized_subtype,
            affected_requirement_ids=requirement_ids,
        ),
        "task_id": str(task_id),
        "task_revision": int(task_revision),
        "failure_class": FailureClass.DEGRADABLE_QUALITY.value,
        "subtype": normalized_subtype,
        "reason": normalized_subtype,
        "affected_claims": _stable_strings(affected_claims),
        "affected_requirement_ids": requirement_ids,
        "attempted_repairs": deepcopy(attempted_repairs),
        "final_fallback": str(final_fallback),
        "status": "active",
        "metadata": dict(metadata or {}),
    }


def upsert_degraded_issue(
    registry: Iterable[Mapping[str, Any]],
    record: DegradedIssueRecord,
) -> list[DegradedIssueRecord]:
    """Upsert one record and supersede only an older revision of the same issue."""

    normalized: list[DegradedIssueRecord] = []
    replaced = False
    for raw in registry:
        existing = deepcopy(dict(raw))
        if str(existing.get("issue_id") or "") == record["issue_id"]:
            normalized.append(deepcopy(record))
            replaced = True
            continue
        if (
            str(existing.get("task_id") or "") == record["task_id"]
            and str(existing.get("subtype") or "") == record["subtype"]
            and str(existing.get("status") or "active") == "active"
        ):
            existing["status"] = "superseded"
        normalized.append(existing)  # type: ignore[arg-type]
    if not replaced:
        normalized.append(deepcopy(record))
    return normalized

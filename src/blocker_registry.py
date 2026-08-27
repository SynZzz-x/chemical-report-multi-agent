"""Pure blocker identity, dependency scheduling, and resolution transitions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from typing import Any

from .failure_semantics import (
    BlockerResolutionRecord,
    RequirementModification,
    UserBlockerRecord,
)


_CANONICAL_ACTIONS = {
    "UPLOAD_RESOURCES",
    "MODIFY_REQUIREMENT",
    "APPROVE_EXCEPTION",
    "CANCEL_JOB",
}
_TERMINAL_TASK_STATUSES = {"committed", "degraded"}


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted(
        {text for value in values if (text := str(value or "").strip())}
    )


def _digest(prefix: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return prefix + sha256(canonical.encode("utf-8")).hexdigest()


def affected_task_closure(
    tasks: Sequence[Mapping[str, Any]], origin_task_id: str
) -> list[str]:
    """Return the origin and explicit dependency descendants in plan order."""

    origin = str(origin_task_id)
    affected = {origin}
    changed = True
    while changed:
        changed = False
        for task in tasks:
            task_id = str(task.get("task_id") or "")
            dependencies = {
                str(dependency)
                for dependency in task.get("depends_on_task_ids") or []
            }
            if task_id and task_id not in affected and dependencies & affected:
                affected.add(task_id)
                changed = True
    return [
        str(task.get("task_id"))
        for task in tasks
        if str(task.get("task_id") or "") in affected
    ]


def runnable_task_ids(
    tasks: Sequence[Mapping[str, Any]], outcomes: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Select pending tasks whose explicit dependencies are terminal."""

    runnable: list[str] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        outcome = outcomes.get(task_id) or {}
        if str(outcome.get("status") or "pending") != "pending":
            continue
        dependencies = [
            str(dependency)
            for dependency in task.get("depends_on_task_ids") or []
        ]
        if all(
            str((outcomes.get(dependency) or {}).get("status") or "")
            in _TERMINAL_TASK_STATUSES
            for dependency in dependencies
        ):
            runnable.append(task_id)
    return runnable


def build_user_blocker(
    *,
    job_scope: str,
    task_id: str,
    subtype: str,
    requirement_ids: Iterable[Any],
    affected_task_ids: Iterable[Any],
    missing_resource_id: str | None,
    reason: str,
    attempted_repairs: list[dict[str, Any]],
    available_options: Iterable[Any],
    metadata: Mapping[str, Any] | None = None,
) -> UserBlockerRecord:
    normalized_requirements = _strings(requirement_ids)
    normalized_subtype = str(subtype).upper()
    blocker_id = _digest(
        "blocker-",
        {
            "job_scope": str(job_scope),
            "task_id": str(task_id),
            "subtype": normalized_subtype,
            "requirement_ids": normalized_requirements,
            "missing_resource_id": str(missing_resource_id or ""),
        },
    )
    return {
        "blocker_id": blocker_id,
        "status": "pending",
        "task_id": str(task_id),
        "requirement_ids": normalized_requirements,
        "affected_task_ids": [str(value) for value in affected_task_ids],
        "reason": str(reason),
        "required_user_action": normalized_subtype,
        "available_options": list(
            dict.fromkeys(
                str(option).strip()
                for option in available_options
                if str(option).strip()
            )
        ),
        "attempted_repairs": deepcopy(attempted_repairs),
        "metadata": dict(metadata or {}),
    }


def upsert_user_blocker(
    registry: Iterable[Mapping[str, Any]], record: UserBlockerRecord
) -> list[UserBlockerRecord]:
    normalized: list[UserBlockerRecord] = []
    replaced = False
    for raw in registry:
        if str(raw.get("blocker_id") or "") == record["blocker_id"]:
            normalized.append(deepcopy(record))
            replaced = True
        else:
            normalized.append(deepcopy(dict(raw)))  # type: ignore[arg-type]
    if not replaced:
        normalized.append(deepcopy(record))
    return normalized


def _resolution_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "results": deepcopy(state.get("results") or []),
        "task_revisions": deepcopy(state.get("task_revisions") or {}),
        "requirement_registry": deepcopy(state.get("requirement_registry") or []),
        "pending_user_blockers": deepcopy(state.get("pending_user_blockers") or []),
        "blocker_resolution_registry": deepcopy(
            state.get("blocker_resolution_registry") or []
        ),
        "resolved_user_blocker_ids": list(
            dict.fromkeys(state.get("resolved_user_blocker_ids") or [])
        ),
        "task_outcome_registry": deepcopy(state.get("task_outcome_registry") or {}),
        "resume_task_id": str(state.get("resume_task_id") or ""),
        "cancel_job": bool(state.get("cancel_job", False)),
    }


def _requirement_modification(
    registry: list[dict[str, Any]], update: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], RequirementModification, int]:
    requirement_id = str(update.get("requirement_id") or "").strip()
    index = next(
        (
            index
            for index, requirement in enumerate(registry)
            if str(requirement.get("requirement_id") or "") == requirement_id
        ),
        None,
    )
    if index is None:
        raise ValueError(f"unknown requirement: {requirement_id}")
    previous = registry[index]
    previous_revision = int(previous.get("contract_revision", 1) or 1)
    new_revision = previous_revision + 1
    withdraw = update.get("new_text") is None
    new_text = None if withdraw else str(update.get("new_text") or "").strip()
    if not withdraw and not new_text:
        raise ValueError("new requirement text must be non-empty")
    new_severity = (
        None
        if withdraw
        else str(update.get("new_severity") or previous.get("severity") or "soft")
    )
    if new_severity not in {None, "hard", "soft"}:
        raise ValueError("new requirement severity must be hard or soft")
    modified = deepcopy(previous)
    if not withdraw:
        modified["text"] = new_text
        modified["severity"] = new_severity
    modified["status"] = "withdrawn" if withdraw else "modified"
    modified["contract_revision"] = new_revision
    registry[index] = modified
    modification: RequirementModification = {
        "requirement_id": requirement_id,
        "operation": "withdraw" if withdraw else "update",
        "previous_text": str(previous.get("text") or ""),
        "new_text": new_text,
        "previous_severity": str(previous.get("severity") or "soft"),
        "new_severity": new_severity,
        "previous_contract_revision": previous_revision,
        "new_contract_revision": new_revision,
    }
    return registry, modification, new_revision


def apply_blocker_resolution(
    state: Mapping[str, Any],
    *,
    blocker_id: str,
    action: str,
    resource_ids: Iterable[Any] = (),
    requirement_update: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_action = str(action).upper()
    if normalized_action not in _CANONICAL_ACTIONS:
        raise ValueError(f"unsupported blocker resolution action: {normalized_action}")
    blockers = deepcopy(list(state.get("pending_user_blockers") or []))
    blocker_index = next(
        (
            index
            for index, blocker in enumerate(blockers)
            if str(blocker.get("blocker_id") or "") == str(blocker_id)
        ),
        None,
    )
    if blocker_index is None:
        raise ValueError(f"unknown blocker: {blocker_id}")
    blocker = blockers[blocker_index]
    available_options = {
        str(option).upper() for option in blocker.get("available_options") or []
    }
    if normalized_action not in available_options:
        raise ValueError("resolution action is not available for this blocker")
    normalized_resources = _strings(resource_ids)
    if normalized_action == "UPLOAD_RESOURCES" and not normalized_resources:
        raise ValueError("UPLOAD_RESOURCES requires at least one resource")
    canonical_update = dict(requirement_update or {})
    resolution_id = _digest(
        "resolution-",
        {
            "blocker_id": str(blocker_id),
            "action": normalized_action,
            "resource_ids": normalized_resources,
            "requirement_update": canonical_update,
        },
    )
    resolutions = deepcopy(list(state.get("blocker_resolution_registry") or []))
    if any(
        str(resolution.get("resolution_id") or "") == resolution_id
        for resolution in resolutions
    ):
        return _resolution_projection(state)
    if any(
        str(resolution.get("blocker_id") or "") == str(blocker_id)
        for resolution in resolutions
    ):
        raise ValueError("conflicting resolution already exists for blocker")

    requirements = deepcopy(list(state.get("requirement_registry") or []))
    modification = None
    contract_revision = max(
        [int(item.get("contract_revision", 1) or 1) for item in requirements]
        or [1]
    )
    if normalized_action == "MODIFY_REQUIREMENT":
        requested_requirement_id = str(
            canonical_update.get("requirement_id") or ""
        ).strip()
        if requested_requirement_id not in set(blocker.get("requirement_ids") or []):
            raise ValueError("requirement modification is outside blocker scope")
        requirements, modification, contract_revision = _requirement_modification(
            requirements, canonical_update
        )
    if normalized_action == "APPROVE_EXCEPTION":
        requirement_ids = set(blocker.get("requirement_ids") or [])
        approval_allowed = any(
            str(requirement.get("requirement_id") or "") in requirement_ids
            and str(requirement.get("kind") or "") in {"human_approval", "waiver"}
            for requirement in requirements
        )
        if not approval_allowed:
            raise ValueError("requirement does not allow approval or waiver")

    affected = list(blocker.get("affected_task_ids") or [blocker.get("task_id")])
    outcomes = deepcopy(dict(state.get("task_outcome_registry") or {}))
    original_results = [deepcopy(result) for result in state.get("results") or []]
    results = original_results
    resolved_ids = list(dict.fromkeys(state.get("resolved_user_blocker_ids") or []))
    cancel_job = False
    if normalized_action in {"UPLOAD_RESOURCES", "MODIFY_REQUIREMENT"}:
        results = [
            result
            for result in original_results
            if str(result.get("task_id") or "") not in set(affected)
        ]
        blocker["status"] = "retry_pending"
        for task_id in affected:
            if task_id in outcomes:
                outcomes[task_id]["status"] = "pending"
                outcomes[task_id]["blocker_ids"] = []
        status = "retry_pending"
    elif normalized_action == "APPROVE_EXCEPTION":
        blocker["status"] = "resolved"
        if str(blocker_id) not in resolved_ids:
            resolved_ids.append(str(blocker_id))
        origin_task_id = str(blocker.get("task_id") or "")
        for task_id in affected:
            if task_id not in outcomes:
                continue
            outcomes[task_id]["status"] = (
                "degraded" if task_id == origin_task_id else "pending"
            )
            outcomes[task_id]["blocker_ids"] = []
        status = "applied"
    else:
        blocker["status"] = "cancelled"
        if str(blocker_id) not in resolved_ids:
            resolved_ids.append(str(blocker_id))
        cancel_job = True
        status = "applied"
    blockers[blocker_index] = blocker

    runnable = runnable_task_ids(state.get("tasks") or [], outcomes)
    resume_task_id = next(
        (task_id for task_id in affected if task_id in runnable),
        "",
    )
    resolution: BlockerResolutionRecord = {
        "resolution_id": resolution_id,
        "blocker_id": str(blocker_id),
        "action": normalized_action,  # type: ignore[typeddict-item]
        "status": status,  # type: ignore[typeddict-item]
        "resource_ids": normalized_resources,
        "requirement_modification": modification,
        "affected_task_ids": affected,
        "contract_revision": contract_revision,
        "metadata": dict(metadata or {}),
    }
    resolutions.append(resolution)
    return {
        "results": results,
        "task_revisions": deepcopy(state.get("task_revisions") or {}),
        "requirement_registry": requirements,
        "pending_user_blockers": blockers,
        "blocker_resolution_registry": resolutions,
        "resolved_user_blocker_ids": resolved_ids,
        "task_outcome_registry": outcomes,
        "resume_task_id": resume_task_id,
        "cancel_job": cancel_job,
    }

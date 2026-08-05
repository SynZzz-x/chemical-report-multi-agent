"""Validation and atomic application for bounded local plan patches."""

from __future__ import annotations

from copy import deepcopy
from os.path import basename
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .policy import MAX_JOB_PATCHES, MAX_TASK_PATCHES


class PatchValidationError(ValueError):
    """Raised when a local plan patch is unsafe to apply."""


_UPDATE_FIELDS = {
    "task_name",
    "task_description",
    "query",
    "use_rag",
    "use_web",
    "generate_table",
    "generate_figure",
    "use_resources",
    "tool_requirements",
    "visualization",
}
_OPERATION_NAMES = {"update_task", "move_before", "insert_before"}


def _task_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PatchValidationError(f"{field} must be a non-empty task ID")
    return value.strip()


def _as_resource_names(values: Iterable[Any]) -> set[str]:
    names: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            names.update({text, basename(text)})
    return names


def _known_resources(state: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for resource in state.get("docs") or []:
        if not isinstance(resource, Mapping):
            continue
        identifiers.update(
            _as_resource_names(
                resource.get(key)
                for key in ("name", "path", "file_id", "resource_id")
            )
        )
    return identifiers


def _validate_resources(resources: Any, known_resources: set[str], field: str) -> None:
    if not isinstance(resources, list):
        raise PatchValidationError(f"{field} must be a list")
    unknown = _as_resource_names(resources) - known_resources
    if unknown:
        raise PatchValidationError(f"unknown resource: {sorted(unknown)[0]}")


def _normalise_counter(counter: Any, tasks: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for key, value in (counter or {}).items():
        task_id = None
        if isinstance(key, int) and 0 <= key < len(tasks):
            candidate = tasks[key].get("task_id")
            if candidate is not None:
                task_id = str(candidate)
        normalized[task_id or str(key)] = int(value or 0)
    return normalized


def _state_tasks(state: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    tasks = state.get("tasks") or []
    if not isinstance(tasks, list):
        raise PatchValidationError("state tasks must be a list")

    copied_tasks: List[Dict[str, Any]] = []
    identifiers: List[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise PatchValidationError("state task must be an object")
        task_copy = deepcopy(dict(task))
        identifier = _task_id(task_copy.get("task_id"), "state task_id")
        copied_tasks.append(task_copy)
        identifiers.append(identifier)

    if len(set(identifiers)) != len(identifiers):
        raise PatchValidationError("duplicate task_id in state")
    return copied_tasks, identifiers


def _require_nonempty_string(patch: Mapping[str, Any], field: str) -> str:
    value = patch.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PatchValidationError(f"{field} must be non-empty")
    return value.strip()


def _affected_task_ids(patch: Mapping[str, Any], known_ids: set[str]) -> List[str]:
    values = patch.get("affected_task_ids")
    if not isinstance(values, list) or not values:
        raise PatchValidationError("affected_task_ids must be a non-empty list")

    task_ids = [_task_id(value, "affected_task_ids") for value in values]
    if len(set(task_ids)) != len(task_ids):
        raise PatchValidationError("duplicate task_id in affected_task_ids")
    unknown = set(task_ids) - known_ids
    if unknown:
        raise PatchValidationError(f"unknown task in affected_task_ids: {sorted(unknown)[0]}")
    return task_ids


def _validate_update(
    operation: Mapping[str, Any], known_ids: set[str], known_resources: set[str]
) -> List[str]:
    task_id = _task_id(operation.get("task_id"), "update_task.task_id")
    if task_id not in known_ids:
        raise PatchValidationError(f"unknown task: {task_id}")
    changes = operation.get("changes")
    if not isinstance(changes, Mapping) or not changes:
        raise PatchValidationError("update_task.changes must be a non-empty object")
    disallowed = set(changes) - _UPDATE_FIELDS
    if disallowed:
        raise PatchValidationError(
            f"update_task change not allowed: {sorted(disallowed)[0]}"
        )
    if "use_resources" in changes:
        _validate_resources(changes["use_resources"], known_resources, "use_resources")
    return [task_id]


def _validate_move(operation: Mapping[str, Any], known_ids: set[str]) -> List[str]:
    task_id = _task_id(operation.get("task_id"), "move_before.task_id")
    before_task_id = _task_id(
        operation.get("before_task_id"), "move_before.before_task_id"
    )
    if task_id not in known_ids or before_task_id not in known_ids:
        raise PatchValidationError("move_before references an unknown task")
    if task_id == before_task_id:
        raise PatchValidationError("move_before task_id must differ from before_task_id")
    return [task_id, before_task_id]


def _validate_insert(
    operation: Mapping[str, Any], known_ids: set[str], known_resources: set[str]
) -> List[str]:
    before_task_id = _task_id(
        operation.get("before_task_id"), "insert_before.before_task_id"
    )
    if before_task_id not in known_ids:
        raise PatchValidationError(f"unknown task: {before_task_id}")
    task = operation.get("task")
    if not isinstance(task, Mapping):
        raise PatchValidationError("insert_before.task must be an object")
    task_id = _task_id(task.get("task_id"), "insert_before.task.task_id")
    if task_id in known_ids:
        raise PatchValidationError(f"duplicate task_id: {task_id}")
    if "use_resources" in task:
        _validate_resources(task["use_resources"], known_resources, "use_resources")
    known_ids.add(task_id)
    return [task_id, before_task_id]


def _apply_operations(tasks: List[Dict[str, Any]], operations: Sequence[Mapping[str, Any]]) -> None:
    for operation in operations:
        operation_name = operation["op"]
        if operation_name == "update_task":
            task_id = operation["task_id"].strip()
            target = next(task for task in tasks if task["task_id"] == task_id)
            target.update(deepcopy(dict(operation["changes"])))
        elif operation_name == "move_before":
            task_id = operation["task_id"].strip()
            before_task_id = operation["before_task_id"].strip()
            target_index = next(
                index for index, task in enumerate(tasks) if task["task_id"] == task_id
            )
            task = tasks.pop(target_index)
            before_index = next(
                index
                for index, candidate in enumerate(tasks)
                if candidate["task_id"] == before_task_id
            )
            tasks.insert(before_index, task)
        else:
            before_task_id = operation["before_task_id"].strip()
            before_index = next(
                index
                for index, candidate in enumerate(tasks)
                if candidate["task_id"] == before_task_id
            )
            tasks.insert(before_index, deepcopy(dict(operation["task"])))


def _validated_patch(state: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(patch, Mapping):
        raise PatchValidationError("patch must be an object")
    tasks, task_ids = _state_tasks(state)
    known_ids = set(task_ids)

    base_plan_revision = patch.get("base_plan_revision")
    if not isinstance(base_plan_revision, int) or isinstance(base_plan_revision, bool):
        raise PatchValidationError("base_plan_revision must be an integer")
    plan_revision = int(state.get("plan_revision", 1) or 1)
    if base_plan_revision != plan_revision:
        raise PatchValidationError("base_plan_revision does not match the current plan")

    reason_code = _require_nonempty_string(patch, "reason_code")
    reason = _require_nonempty_string(patch, "reason")
    expected_resolution = _require_nonempty_string(patch, "expected_resolution")
    known_resources = _known_resources(state)
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        raise PatchValidationError("operations must be a non-empty list")

    operation_task_ids: List[str] = []
    accepted_ids = {
        str(result.get("task_id"))
        for result in state.get("results") or []
        if isinstance(result, Mapping) and result.get("task_id") is not None
    }
    moving_task_ids: set[str] = set()
    operation_copies: List[Dict[str, Any]] = []
    for raw_operation in operations:
        if not isinstance(raw_operation, Mapping):
            raise PatchValidationError("operation must be an object")
        operation = deepcopy(dict(raw_operation))
        operation_name = operation.get("op")
        if operation_name not in _OPERATION_NAMES:
            raise PatchValidationError(f"unsupported operation: {operation_name}")
        if operation_name == "update_task":
            operation_task_ids.extend(
                _validate_update(operation, known_ids, known_resources)
            )
        elif operation_name == "move_before":
            affected = _validate_move(operation, known_ids)
            moving_task_ids.add(affected[0])
            operation_task_ids.extend(affected)
        else:
            operation_task_ids.extend(
                _validate_insert(operation, known_ids, known_resources)
            )
        operation_copies.append(operation)

    affected_task_ids = _affected_task_ids(patch, known_ids)
    affected_set = set(affected_task_ids)
    operation_set = set(operation_task_ids)
    extra_affected = affected_set - operation_set
    invalidated_accepted = extra_affected & accepted_ids
    if invalidated_accepted:
        raise PatchValidationError("accepted result invalidation was not declared by an operation")
    if operation_set != affected_set:
        raise PatchValidationError("operations must match affected_task_ids")
    if moving_task_ids & accepted_ids:
        raise PatchValidationError("cannot move an accepted task")

    simulated_tasks = deepcopy(tasks)
    _apply_operations(simulated_tasks, operation_copies)
    final_task_ids = [str(task["task_id"]) for task in simulated_tasks]
    resume_task_id = _task_id(patch.get("resume_task_id"), "resume_task_id")
    if resume_task_id not in final_task_ids or resume_task_id not in affected_set:
        raise PatchValidationError("resume_task_id must name an affected task")
    earliest_affected = next(task_id for task_id in final_task_ids if task_id in affected_set)
    if resume_task_id != earliest_affected:
        raise PatchValidationError("resume_task_id must be the earliest affected task")

    task_patch_count = _normalise_counter(state.get("task_patch_count"), tasks)
    if int(state.get("job_patch_count", 0) or 0) >= MAX_JOB_PATCHES:
        raise PatchValidationError("job patch limit reached")
    for task_id in affected_task_ids:
        if task_patch_count.get(task_id, 0) >= MAX_TASK_PATCHES:
            raise PatchValidationError(f"task patch limit reached: {task_id}")

    return {
        "base_plan_revision": base_plan_revision,
        "reason_code": reason_code,
        "reason": reason,
        "affected_task_ids": affected_task_ids,
        "operations": operation_copies,
        "resume_task_id": resume_task_id,
        "expected_resolution": expected_resolution,
    }


def validate_plan_patch(state: Mapping[str, Any], patch: Mapping[str, Any]) -> None:
    """Raise :class:`PatchValidationError` unless ``patch`` is safe to apply."""
    _validated_patch(state, patch)


def apply_plan_patch(state: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    """Return atomic top-level state updates for a fully validated local patch."""
    validated = _validated_patch(state, patch)
    original_tasks, original_task_ids = _state_tasks(state)
    tasks = deepcopy(original_tasks)
    _apply_operations(tasks, validated["operations"])

    affected_task_ids = validated["affected_task_ids"]
    affected_set = set(affected_task_ids)
    task_revisions = _normalise_counter(state.get("task_revisions"), original_tasks)
    for task_id in original_task_ids:
        task_revisions.setdefault(task_id, 1)
    for task_id in affected_task_ids:
        task_revisions[task_id] = task_revisions.get(task_id, 0) + 1

    task_patch_count = _normalise_counter(state.get("task_patch_count"), original_tasks)
    for task_id in affected_task_ids:
        task_patch_count[task_id] = task_patch_count.get(task_id, 0) + 1

    next_plan_revision = int(state.get("plan_revision", 1) or 1) + 1
    history = [deepcopy(item) for item in state.get("plan_patch_history") or []]
    history.append(
        {
            "base_plan_revision": validated["base_plan_revision"],
            "plan_revision": next_plan_revision,
            "reason_code": validated["reason_code"],
            "reason": validated["reason"],
            "affected_task_ids": list(affected_task_ids),
            "operations": deepcopy(validated["operations"]),
            "resume_task_id": validated["resume_task_id"],
            "expected_resolution": validated["expected_resolution"],
        }
    )
    results = [
        deepcopy(result)
        for result in state.get("results") or []
        if not (
            isinstance(result, Mapping)
            and str(result.get("task_id")) in affected_set
        )
    ]
    cursor = next(
        index
        for index, task in enumerate(tasks)
        if str(task["task_id"]) == validated["resume_task_id"]
    )

    return {
        "tasks": tasks,
        "results": results,
        "cursor": cursor,
        "plan_revision": next_plan_revision,
        "task_revisions": task_revisions,
        "task_patch_count": task_patch_count,
        "job_patch_count": int(state.get("job_patch_count", 0) or 0) + 1,
        "plan_patch_history": history,
        "pending_user_action": {},
    }

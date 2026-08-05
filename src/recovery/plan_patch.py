"""Validation and atomic application for bounded local plan patches."""

from __future__ import annotations

from copy import deepcopy
from os.path import basename
from typing import Any, Dict, List, Mapping, Sequence

from src.limits import MAX_PLAN_TASKS
from src.tool_names import canonical_tool_name

from .policy import MAX_JOB_PATCHES, MAX_TASK_PATCHES


class PatchValidationError(ValueError):
    """Raised when a local plan patch is unsafe to apply."""


_UPDATE_FIELDS = {
    "task_name",
    "task_description",
    "query",
    "use_rag",
    "use_web",
    "allow_web_fallback",
    "generate_table",
    "generate_figure",
    "use_resources",
    "tool_requirements",
    "visualization",
}
_OPERATION_NAMES = {"update_task", "move_before", "insert_before"}
_TASK_TYPES = {"analysis", "summary", "inference"}
_BOOLEAN_TASK_FIELDS = {
    "use_rag",
    "use_web",
    "generate_table",
    "generate_figure",
}
_REQUIRED_INSERTED_TASK_FIELDS = {
    "task_name",
    "task_description",
    "task_type",
    "use_rag",
    "use_web",
    "generate_table",
    "generate_figure",
    "query",
    "use_resources",
}
_ALLOWED_INSERTED_TASK_FIELDS = _REQUIRED_INSERTED_TASK_FIELDS | {
    "task_id",
    "allow_web_fallback",
    "tool_requirements",
    "visualization",
}
_COMPLETED_TASK_STATUSES = {"ACCEPTED", "COMPLETED", "DONE", "PASS", "PASSED"}


def _task_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PatchValidationError(f"{field} must be a non-empty task ID")
    return value.strip()


def _resource_aliases(state: Mapping[str, Any]) -> Dict[str, set[str]]:
    """Map every known resource alias to its canonical job-local identity."""
    aliases: Dict[str, set[str]] = {}
    for resource in state.get("docs") or []:
        if not isinstance(resource, Mapping):
            continue
        values = {
            key: value.strip()
            for key, value in resource.items()
            if key in {"name", "path", "file_id", "resource_id"}
            and isinstance(value, str)
            and value.strip()
        }
        canonical = next(
            (values[key] for key in ("path", "file_id", "resource_id", "name") if key in values),
            None,
        )
        if canonical is None:
            continue
        for value in values.values():
            for alias in {value, basename(value)}:
                aliases.setdefault(alias, set()).add(canonical)
    return aliases


def _normalise_resources(
    resources: Any, resource_aliases: Mapping[str, set[str]], field: str
) -> List[str]:
    if not isinstance(resources, list):
        raise PatchValidationError(f"{field} must be a list")
    normalized: List[str] = []
    for resource in resources:
        if not isinstance(resource, str):
            raise PatchValidationError(f"{field} must be a list of strings")
        alias = resource.strip()
        matches = resource_aliases.get(alias, set())
        if not matches:
            raise PatchValidationError(f"unknown resource: {alias}")
        if len(matches) != 1:
            raise PatchValidationError(f"ambiguous resource: {alias}")
        normalized.append(next(iter(matches)))
    return normalized


def _normalise_tool_requirements(requirements: Any) -> List[str]:
    if not isinstance(requirements, list):
        raise PatchValidationError("tool_requirements must be a list of strings")
    normalized: List[str] = []
    for requirement in requirements:
        canonical_name = canonical_tool_name(requirement)
        if canonical_name is None:
            raise PatchValidationError(
                "tool_requirements contains an invalid tool requirement"
            )
        normalized.append(canonical_name)
    return normalized


def _normalise_counter(counter: Any, tasks: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    if not isinstance(counter, Mapping):
        return normalized
    for key, value in counter.items():
        task_id = None
        cursor_key = None
        if isinstance(key, int) and not isinstance(key, bool):
            cursor_key = key
        elif isinstance(key, str) and key.isdecimal():
            cursor_key = int(key)
        if cursor_key is not None and 0 <= cursor_key < len(tasks):
            candidate = tasks[cursor_key].get("task_id")
            if candidate is not None:
                task_id = str(candidate)
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        identifier = task_id or str(key)
        normalized[identifier] = max(normalized.get(identifier, 0), count)
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
        task_copy["task_id"] = identifier
        copied_tasks.append(task_copy)
        identifiers.append(identifier)

    if len(set(identifiers)) != len(identifiers):
        raise PatchValidationError("duplicate task_id in state")
    return copied_tasks, identifiers


def _completed_task_ids(
    state: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> set[str]:
    """Return tasks that a local patch must never mutate or reorder."""
    completed = {
        str(result.get("task_id"))
        for result in state.get("results") or []
        if isinstance(result, Mapping) and result.get("task_id") is not None
    }
    try:
        cursor = max(0, int(state.get("cursor", 0) or 0))
    except (TypeError, ValueError):
        cursor = 0
    for task in tasks[:cursor]:
        if task.get("task_id") is not None:
            completed.add(str(task["task_id"]))
    for task in tasks:
        if (
            str(task.get("status") or "").strip().upper()
            in _COMPLETED_TASK_STATUSES
            and task.get("task_id") is not None
        ):
            completed.add(str(task["task_id"]))
    return completed


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


def _validate_task_fields(
    task: Dict[str, Any],
    resource_aliases: Mapping[str, set[str]],
    *,
    required: bool,
) -> None:
    if required:
        unknown = set(task) - _ALLOWED_INSERTED_TASK_FIELDS
        if unknown:
            raise PatchValidationError(
                f"insert_before.task unknown field: {sorted(unknown)[0]}"
            )
        missing = _REQUIRED_INSERTED_TASK_FIELDS - set(task)
        if missing:
            raise PatchValidationError(f"insert_before.task missing {sorted(missing)[0]}")

    for field in ("task_name", "task_description"):
        if field in task:
            if not isinstance(task[field], str) or not task[field].strip():
                raise PatchValidationError(f"{field} must be a non-empty string")
        elif required:
            raise PatchValidationError(f"insert_before.task missing {field}")

    if "task_type" in task:
        if task["task_type"] not in _TASK_TYPES:
            raise PatchValidationError("task_type must be analysis, summary, or inference")
    elif required:
        raise PatchValidationError("insert_before.task missing task_type")

    for field in _BOOLEAN_TASK_FIELDS:
        if field in task:
            if not isinstance(task[field], bool):
                raise PatchValidationError(f"{field} must be a boolean")
        elif required:
            raise PatchValidationError(f"insert_before.task missing {field}")
    if "allow_web_fallback" in task and not isinstance(
        task["allow_web_fallback"], bool
    ):
        raise PatchValidationError("allow_web_fallback must be a boolean")

    if "query" in task:
        if not isinstance(task["query"], str):
            raise PatchValidationError("query must be a string")
    elif required:
        raise PatchValidationError("insert_before.task missing query")

    if "use_resources" in task:
        task["use_resources"] = _normalise_resources(
            task["use_resources"], resource_aliases, "use_resources"
        )
    elif required:
        raise PatchValidationError("insert_before.task missing use_resources")

    if "tool_requirements" in task:
        task["tool_requirements"] = _normalise_tool_requirements(
            task["tool_requirements"]
        )
    if "visualization" in task:
        visualization = task["visualization"]
        if not isinstance(visualization, Mapping):
            raise PatchValidationError("visualization must be a mapping")
        if "allow_web_fallback" in visualization and not isinstance(
            visualization["allow_web_fallback"], bool
        ):
            raise PatchValidationError(
                "visualization.allow_web_fallback must be a boolean"
            )


def _validate_task_consistency(task: Mapping[str, Any]) -> None:
    raw_requirements = task.get("tool_requirements")
    if raw_requirements is None:
        return
    tool_requirements = _normalise_tool_requirements(raw_requirements)
    spider_requirement = next(
        (
            requirement
            for requirement in tool_requirements
            if canonical_tool_name(requirement) == "spider_tool"
        ),
        None,
    )
    if spider_requirement is None:
        return

    visualization = task.get("visualization")
    visualization_allows_web = isinstance(visualization, Mapping) and (
        visualization.get("allow_web_fallback") is True
    )
    if not (
        task.get("use_web") is True
        or task.get("allow_web_fallback") is True
        or visualization_allows_web
    ):
        raise PatchValidationError(
            f"{spider_requirement} requires explicit web permission"
        )


def _validate_update(
    operation: Dict[str, Any],
    known_ids: set[str],
    resource_aliases: Mapping[str, set[str]],
    completed_ids: set[str],
) -> List[str]:
    task_id = _task_id(operation.get("task_id"), "update_task.task_id")
    operation["task_id"] = task_id
    if task_id not in known_ids:
        raise PatchValidationError(f"unknown task: {task_id}")
    if task_id in completed_ids:
        raise PatchValidationError("cannot update a completed or accepted task")
    changes = operation.get("changes")
    if not isinstance(changes, Mapping) or not changes:
        raise PatchValidationError("update_task.changes must be a non-empty object")
    changes = deepcopy(dict(changes))
    operation["changes"] = changes
    disallowed = set(changes) - _UPDATE_FIELDS
    if disallowed:
        raise PatchValidationError(
            f"update_task change not allowed: {sorted(disallowed)[0]}"
        )
    _validate_task_fields(changes, resource_aliases, required=False)
    return [task_id]


def _move_before(order: List[str], task_id: str, before_task_id: str) -> None:
    task = order.pop(order.index(task_id))
    order.insert(order.index(before_task_id), task)


def _validate_move(
    operation: Dict[str, Any],
    known_ids: set[str],
    current_order: List[str],
    completed_ids: set[str],
) -> List[str]:
    task_id = _task_id(operation.get("task_id"), "move_before.task_id")
    before_task_id = _task_id(
        operation.get("before_task_id"), "move_before.before_task_id"
    )
    operation["task_id"] = task_id
    operation["before_task_id"] = before_task_id
    if task_id not in known_ids or before_task_id not in known_ids:
        raise PatchValidationError("move_before references an unknown task")
    if task_id == before_task_id:
        raise PatchValidationError("move_before task_id must differ from before_task_id")
    task_index = current_order.index(task_id)
    anchor_index = current_order.index(before_task_id)
    crossed = (
        current_order[anchor_index:task_index]
        if task_index > anchor_index
        else current_order[task_index + 1 : anchor_index]
    )
    if task_id in completed_ids:
        raise PatchValidationError("cannot move a completed or accepted task")
    if before_task_id in completed_ids:
        raise PatchValidationError("cannot move before a completed or accepted anchor")
    completed_crossed = set(crossed) & completed_ids
    if completed_crossed:
        raise PatchValidationError(
            "cannot move across a completed or accepted crossed task"
        )
    _move_before(current_order, task_id, before_task_id)
    return list(dict.fromkeys([task_id, before_task_id, *crossed]))


def _validate_insert(
    operation: Dict[str, Any],
    known_ids: set[str],
    resource_aliases: Mapping[str, set[str]],
    current_order: List[str],
    completed_ids: set[str],
) -> List[str]:
    before_task_id = _task_id(
        operation.get("before_task_id"), "insert_before.before_task_id"
    )
    operation["before_task_id"] = before_task_id
    if before_task_id not in known_ids:
        raise PatchValidationError(f"unknown task: {before_task_id}")
    if before_task_id in completed_ids:
        raise PatchValidationError(
            "cannot insert before a completed or accepted anchor"
        )
    task = operation.get("task")
    if not isinstance(task, Mapping):
        raise PatchValidationError("insert_before.task must be an object")
    task = deepcopy(dict(task))
    operation["task"] = task
    task_id = _task_id(task.get("task_id"), "insert_before.task.task_id")
    task["task_id"] = task_id
    if task_id in known_ids:
        raise PatchValidationError(f"duplicate task_id: {task_id}")
    _validate_task_fields(task, resource_aliases, required=True)
    known_ids.add(task_id)
    current_order.insert(current_order.index(before_task_id), task_id)
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
    resource_aliases = _resource_aliases(state)
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        raise PatchValidationError("operations must be a non-empty list")
    if len(operations) > MAX_PLAN_TASKS:
        raise PatchValidationError("plan patch operation limit exceeded")
    insert_count = sum(
        1
        for operation in operations
        if isinstance(operation, Mapping) and operation.get("op") == "insert_before"
    )
    if len(tasks) + insert_count > MAX_PLAN_TASKS:
        raise PatchValidationError("plan task maximum exceeded")

    operation_task_ids: List[str] = []
    completed_ids = _completed_task_ids(state, tasks)
    current_order = list(task_ids)
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
                _validate_update(
                    operation, known_ids, resource_aliases, completed_ids
                )
            )
        elif operation_name == "move_before":
            operation_task_ids.extend(
                _validate_move(operation, known_ids, current_order, completed_ids)
            )
        else:
            operation_task_ids.extend(
                _validate_insert(
                    operation,
                    known_ids,
                    resource_aliases,
                    current_order,
                    completed_ids,
                )
            )
        operation_copies.append(operation)

    simulated_tasks = deepcopy(tasks)
    _apply_operations(simulated_tasks, operation_copies)
    for task in simulated_tasks:
        _validate_task_consistency(task)

    affected_task_ids = _affected_task_ids(patch, known_ids)
    affected_set = set(affected_task_ids)
    operation_set = set(operation_task_ids)
    extra_affected = affected_set - operation_set
    invalidated_accepted = extra_affected & completed_ids
    if invalidated_accepted:
        raise PatchValidationError("accepted result invalidation was not declared by an operation")
    if operation_set != affected_set:
        raise PatchValidationError("operations must match affected_task_ids")

    final_task_ids = current_order
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

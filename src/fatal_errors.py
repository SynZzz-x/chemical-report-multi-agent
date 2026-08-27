"""Safe fatal diagnostics shared by graph and runner ownership layers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, Literal

from .failure_semantics import FatalSystemError, FailureClass


def build_fatal_system_error(
    error: BaseException,
    *,
    origin: Literal["graph", "runner"],
    component: str,
    operation: str,
    task_id: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> FatalSystemError:
    """Build a bounded record without serializing exception text or traceback."""

    subtype = type(error).__name__
    safe_metadata = {
        str(key): value
        for key, value in (metadata or {}).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    identity = json.dumps(
        {
            "origin": origin,
            "component": str(component),
            "operation": str(operation),
            "task_id": task_id,
            "subtype": subtype,
            "metadata": safe_metadata,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "failure_id": "fatal-" + sha256(identity.encode("utf-8")).hexdigest(),
        "failure_class": FailureClass.FATAL_SYSTEM.value,
        "subtype": subtype,
        "origin": origin,
        "component": str(component),
        "operation": str(operation),
        "task_id": task_id,
        "diagnostic_code": f"{str(component).upper()}_{str(operation).upper()}_FAILED",
        "retryable": False,
        "metadata": safe_metadata,
    }

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from ..config import get_cache_root


def _safe_component(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    return cleaned[:160] or fallback


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _scope_from_messages(messages: Iterable[Any]) -> Dict[str, str]:
    """从最近消息的 additional_kwargs 中兜底恢复作用域。

    Worker 子图目前没有显式声明 user_id/conversation_id/job_id，但顶层
    HumanMessage 已携带这些字段。此兜底可保证子图生成的图表、报告仍落在
    正确的 Job 目录中。后续重构 WorkerState 时可删除此兼容逻辑。
    """
    for message in reversed(list(messages or [])):
        if isinstance(message, Mapping):
            kwargs = _mapping(message.get("additional_kwargs"))
        else:
            kwargs = _mapping(getattr(message, "additional_kwargs", {}))

        if not kwargs:
            continue

        result = {
            "user_id": str(kwargs.get("user_id") or "").strip(),
            "conversation_id": str(kwargs.get("conversation_id") or "").strip(),
            "job_id": str(kwargs.get("job_id") or "").strip(),
        }
        if any(result.values()):
            return result

    return {}


def _resolve_scope(state: Dict[str, Any], config: Any = None) -> Dict[str, str]:
    state = state or {}
    metadata = _mapping(state.get("metadata"))
    config_map = _mapping(config)
    config_metadata = _mapping(config_map.get("metadata"))
    configurable = _mapping(config_map.get("configurable"))
    message_scope = _scope_from_messages(state.get("messages") or [])

    user_id = (
        state.get("user_id")
        or metadata.get("user_id")
        or config_metadata.get("user_id")
        or message_scope.get("user_id")
        or "legacy_user"
    )
    conversation_id = (
        state.get("conversation_id")
        or metadata.get("conversation_id")
        or config_metadata.get("conversation_id")
        or message_scope.get("conversation_id")
        or "legacy_conversation"
    )
    job_id = (
        state.get("job_id")
        or metadata.get("job_id")
        or config_metadata.get("job_id")
        or message_scope.get("job_id")
        or configurable.get("thread_id")
        or "default_job"
    )

    return {
        "user_id": _safe_component(user_id, "legacy_user"),
        "conversation_id": _safe_component(
            conversation_id,
            "legacy_conversation",
        ),
        "job_id": _safe_component(job_id, "default_job"),
    }


def get_session_cache_dir(
    state: Dict[str, Any],
    config: Any = None,
    base_cache_name: str = "cache",
) -> str:
    """返回当前 User/Conversation/Job 的统一工作目录。

    新目录结构：
        cache/users/{user_id}/conversations/{conversation_id}/jobs/{job_id}/

    ``base_cache_name`` 仅为兼容旧调用保留。默认值 ``cache`` 使用
    ``AGENT_CACHE_ROOT``（若设置）或项目根目录下的 cache。
    """
    scope = _resolve_scope(state, config)

    if base_cache_name == "cache":
        cache_root = get_cache_root()
    else:
        project_root = Path(__file__).resolve().parents[2]
        cache_root = project_root / _safe_component(base_cache_name, "cache")
        cache_root.mkdir(parents=True, exist_ok=True)

    job_dir = (
        cache_root
        / "users"
        / scope["user_id"]
        / "conversations"
        / scope["conversation_id"]
        / "jobs"
        / scope["job_id"]
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    return str(job_dir.resolve())


def get_job_subdir(
    state: Dict[str, Any],
    name: str,
    config: Any = None,
) -> str:
    """创建并返回 Job 下的受控子目录。"""
    safe_name = _safe_component(name, "artifacts")
    path = Path(get_session_cache_dir(state, config)) / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def manage_session_files(
    resources: List[Dict[str, Any]],
    session_cache_dir: str,
) -> List[Dict[str, Any]]:
    """将外部资源复制到当前 Job 的 uploads 目录。

    Streamlit 已直接将上传文件写入该目录，因此大多数情况下这里只做校验并
    原样返回。对于 CLI 或其他外部入口传入的文件，使用随机物理文件名复制，
    避免同名覆盖。
    """
    upload_dir = Path(session_cache_dir).resolve() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    managed_resources: List[Dict[str, Any]] = []

    for resource in resources or []:
        if not isinstance(resource, dict):
            continue

        normalized = dict(resource)
        old_path_value = normalized.get("path") or normalized.get("file_path")
        if not old_path_value:
            managed_resources.append(normalized)
            continue

        old_path = Path(str(old_path_value)).expanduser()
        if not old_path.is_file():
            managed_resources.append(normalized)
            continue

        file_id = str(
            normalized.get("file_id")
            or normalized.get("resource_id")
            or f"file_{uuid.uuid4().hex}"
        )
        normalized["file_id"] = file_id
        normalized["resource_id"] = file_id
        normalized.setdefault("original_name", normalized.get("name") or old_path.name)
        normalized.setdefault("name", normalized.get("original_name") or old_path.name)

        # 已经位于当前 Job 的 uploads 目录，不再复制。
        if _is_within(old_path, upload_dir):
            normalized["path"] = str(old_path.resolve())
            normalized.setdefault("stored_name", old_path.name)
            managed_resources.append(normalized)
            continue

        suffix = old_path.suffix.lower()
        destination = upload_dir / f"{_safe_component(file_id, 'file')}{suffix}"

        # 极少数情况下相同 file_id 指向不同内容，生成新 ID 避免覆盖。
        if destination.exists():
            try:
                same_file = (
                    destination.stat().st_size == old_path.stat().st_size
                    and os.path.samefile(destination, old_path)
                )
            except (OSError, FileNotFoundError):
                same_file = False

            if not same_file:
                file_id = f"file_{uuid.uuid4().hex}"
                normalized["file_id"] = file_id
                normalized["resource_id"] = file_id
                destination = upload_dir / f"{file_id}{suffix}"

        try:
            if not destination.exists():
                shutil.copy2(old_path, destination)
            normalized["path"] = str(destination.resolve())
            normalized["stored_name"] = destination.name
        except OSError as exc:
            normalized["copy_error"] = str(exc)
            normalized["path"] = str(old_path.resolve())

        managed_resources.append(normalized)

    return managed_resources

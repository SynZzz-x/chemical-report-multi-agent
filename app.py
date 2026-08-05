"""Streamlit entry for the chemical-report LangGraph application.

Phase-1 goals implemented here:
1. Separate user / conversation / job scopes.
2. Use the LangGraph job_id as thread_id.
3. Send only the current user message to LangGraph; never resend the whole UI history.
4. Keep Streamlit ui_messages for rendering only.
5. Store uploads under user/conversation/job isolated directories with generated file IDs.

Production note:
- The default local user_id is only suitable for a single-user deployment.
  Replace it with the authenticated user ID before multi-user deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.config import get_cache_root, get_local_user_id, missing_key_message
from src.control_messages import blocker_guidance, is_displayable_assistant_message
from src.graph import WorkFlow, WorkFlowAuto
from src.job_store import JobStore, interrupt_from_snapshot
from src.persistence import SQLitePersistence
from src.utils.path_manager import get_session_cache_dir


# -----------------------------------------------------------------------------
# Basic configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="化工行业多 Agent 报告系统",
    page_icon="🧪",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = get_cache_root()


@st.cache_resource
def _open_persistence() -> SQLitePersistence:
    return SQLitePersistence.open()


try:
    PERSISTENCE = _open_persistence()
    JOBS = JobStore(PERSISTENCE.store)
except Exception as exc:
    st.error(f"无法初始化 LangGraph SQLite 持久化：{exc}")
    st.stop()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_session_scope() -> None:
    """Initialize front-end scope identifiers.

    A production deployment must replace the local default with an
    authenticated, stable user identifier.
    """

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = get_local_user_id()

    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = _new_id("conv")

    if "active_job_id" not in st.session_state:
        st.session_state["active_job_id"] = _new_id("job")

    if "active_job_created_at" not in st.session_state:
        st.session_state["active_job_created_at"] = _utc_now_iso()

    if "ui_messages" not in st.session_state:
        st.session_state["ui_messages"] = []

    if "pending_interrupt" not in st.session_state:
        st.session_state["pending_interrupt"] = None

    if "compiled_mode" not in st.session_state:
        st.session_state["compiled_mode"] = None

    if "job_record_created" not in st.session_state:
        st.session_state["job_record_created"] = False

    if "last_run_failed" not in st.session_state:
        st.session_state["last_run_failed"] = False


_ensure_session_scope()


# -----------------------------------------------------------------------------
# Scope and LangGraph helpers
# -----------------------------------------------------------------------------


def _scope() -> dict[str, str]:
    return {
        "user_id": st.session_state["user_id"],
        "conversation_id": st.session_state["conversation_id"],
        "job_id": st.session_state["active_job_id"],
    }


def _graph_config() -> dict[str, Any]:
    scope = _scope()
    # A LangGraph thread represents one report-generation job, not one user and
    # not an entire conversation.
    return {
        "configurable": {"thread_id": scope["job_id"]},
        "tags": ["ChemicalReportAgent", "Streamlit", "LangGraph"],
        "metadata": {
            "app": "ChemicalReportAgent",
            **scope,
        },
        "recursion_limit": 100,
    }


def _job_metadata() -> dict[str, Any]:
    return {
        **_scope(),
        "timestamp": st.session_state["active_job_created_at"],
    }


def _compile_workflow(mode: str) -> None:
    record = _current_job()
    if record is not None and record.get("verifier_mode") != mode:
        raise ValueError("已有任务不能切换审核模式，请先新建报告任务。")

    workflow = WorkFlowAuto() if mode == "auto" else WorkFlow()
    st.session_state["app"] = workflow.compile(
        checkpointer=PERSISTENCE.checkpointer,
        store=PERSISTENCE.store,
    )
    st.session_state["compiled_mode"] = mode


def _current_job() -> dict[str, Any] | None:
    scope = _scope()
    return JOBS.get_job(scope["user_id"], scope["job_id"])


def _ensure_job_record(title: str, verifier_mode: str) -> dict[str, Any]:
    record = _current_job()
    if record is not None:
        if record.get("verifier_mode") != verifier_mode:
            raise ValueError("当前任务的审核模式与已编译工作流不一致。")
        st.session_state["job_record_created"] = True
        return record

    scope = _scope()
    record = JOBS.create_job(
        **scope,
        title=title,
        verifier_mode=verifier_mode,
        ui_messages=st.session_state["ui_messages"],
    )
    st.session_state["job_record_created"] = True
    return record


def _update_job(**changes: Any) -> None:
    if not st.session_state.get("job_record_created"):
        return
    scope = _scope()
    try:
        JOBS.update_job(scope["user_id"], scope["job_id"], **changes)
    except Exception as exc:
        st.warning(f"任务恢复信息保存失败：{exc}")


def _snapshot_values() -> dict[str, Any]:
    app = st.session_state.get("app")
    if app is None:
        return {}
    try:
        snapshot = app.get_state(_graph_config())
        return dict(snapshot.values or {})
    except Exception:
        return {}


def _start_new_job() -> None:
    st.session_state["active_job_id"] = _new_id("job")
    st.session_state["active_job_created_at"] = _utc_now_iso()
    st.session_state["pending_interrupt"] = None
    st.session_state["ui_messages"] = []
    st.session_state["job_record_created"] = False
    st.session_state["last_run_failed"] = False


def _start_new_conversation() -> None:
    st.session_state["conversation_id"] = _new_id("conv")
    _start_new_job()


def _restore_job(job_id: str) -> None:
    user_id = st.session_state["user_id"]
    record = JOBS.get_job(user_id, job_id)
    if record is None:
        raise ValueError("任务不存在或不属于当前用户。")

    mode = record.get("verifier_mode") or "manual"
    missing = object()
    restored_keys = (
        "conversation_id",
        "active_job_id",
        "active_job_created_at",
        "ui_messages",
        "job_record_created",
        "pending_interrupt",
        "compiled_mode",
        "last_run_failed",
        "verifier_mode",
        "app",
    )
    previous_state = {
        key: st.session_state.get(key, missing)
        for key in restored_keys
    }

    try:
        st.session_state["conversation_id"] = record["conversation_id"]
        st.session_state["active_job_id"] = record["job_id"]
        st.session_state["active_job_created_at"] = record["created_at"]
        st.session_state["ui_messages"] = list(record.get("ui_messages") or [])
        st.session_state["job_record_created"] = True
        st.session_state["last_run_failed"] = False
        _compile_workflow(mode)

        if PERSISTENCE.checkpointer.get_tuple(_graph_config()) is None:
            raise ValueError("任务索引存在，但对应 checkpoint 缺失。")

        snapshot = st.session_state["app"].get_state(_graph_config())
        pending = interrupt_from_snapshot(snapshot)
        st.session_state["pending_interrupt"] = pending
        st.session_state["verifier_mode"] = mode

        changes: dict[str, Any] = {"pending_interrupt": pending}
        if pending is not None:
            changes["status"] = "waiting"
        elif record.get("status") in {"running", "waiting"}:
            changes["status"] = (
                "completed"
                if not (getattr(snapshot, "next", ()) or ())
                else "failed"
            )
        JOBS.update_job(user_id, job_id, **changes)
    except Exception:
        for key, value in previous_state.items():
            if value is missing:
                st.session_state.pop(key, None)
            else:
                st.session_state[key] = value
        raise


def _restore_job_from_sidebar(job_id: str) -> None:
    try:
        _restore_job(job_id)
    except Exception as exc:
        st.session_state["restore_error"] = str(exc)
        st.session_state.pop("restore_success", None)
    else:
        st.session_state["restore_success"] = "任务已恢复。"
        st.session_state.pop("restore_error", None)


# -----------------------------------------------------------------------------
# UI-message helpers: rendering only, never used as LangGraph input
# -----------------------------------------------------------------------------


def _append_ui_message(
    role: str,
    content: str = "",
    *,
    kind: str = "text",
    payload: dict[str, Any] | None = None,
    message_id: str | None = None,
) -> None:
    message_id = message_id or _new_id("ui")

    # Protect against accidental duplicate rendering when an event is replayed.
    if any(item.get("id") == message_id for item in st.session_state["ui_messages"]):
        return

    st.session_state["ui_messages"].append(
        {
            "id": message_id,
            "role": role,
            "kind": kind,
            "content": content,
            "payload": payload or {},
            **_scope(),
        }
    )
    _update_job(ui_messages=list(st.session_state["ui_messages"]))


def _render_plan(payload: dict[str, Any]) -> None:
    guidance = payload.get("guidance_text")
    if guidance:
        st.write(guidance)

    structured = payload.get("structured_msg") or {}
    tasks = structured.get("tasks") or []
    mapping = structured.get("resource_mapping") or {}

    with st.expander("任务规划与资源清单", expanded=True):
        st.markdown("### 任务规划")
        if not tasks:
            st.info("当前没有可展示的任务。")
        for index, task in enumerate(tasks, start=1):
            task_name = task.get("task_name") or f"任务 {index}"
            st.markdown(f"**{index}. {task_name}**")
            description = task.get("task_description")
            if description:
                st.caption(description)

        st.markdown("### 所需资源")
        has_resource = False
        for name, resources in mapping.items():
            if not resources:
                continue
            has_resource = True
            st.markdown(f"**{name}**")
            for resource in resources:
                st.markdown(f"- {resource}")
        if not has_resource:
            st.info("当前规划未检测到明确的外部资源需求。")


def _render_verify(payload: dict[str, Any]) -> None:
    guidance = payload.get("guidance_text") or "请审核当前任务结果。"
    st.write(guidance)

    task_name = payload.get("task_name") or "当前任务"
    summary = payload.get("content_summary") or "暂无内容"
    with st.expander(f"任务执行结果：{task_name}", expanded=True):
        st.markdown(summary)
        if len(summary) >= 2000:
            st.caption("内容较长，当前仅展示摘要。")


def _render_ui_message(item: dict[str, Any]) -> None:
    role = item.get("role", "assistant")
    kind = item.get("kind", "text")
    content = item.get("content", "")
    payload = item.get("payload") or {}

    with st.chat_message(role):
        if kind == "plan":
            _render_plan(payload)
        elif kind == "verify":
            _render_verify(payload)
        else:
            st.write(content)


def _render_history() -> None:
    for item in st.session_state["ui_messages"]:
        _render_ui_message(item)


# -----------------------------------------------------------------------------
# Upload isolation
# -----------------------------------------------------------------------------


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return cleaned[:160] or "unknown"


def _job_upload_dir() -> Path:
    scope = _scope()
    directory = (
        CACHE_ROOT
        / "users"
        / _safe_component(scope["user_id"])
        / "conversations"
        / _safe_component(scope["conversation_id"])
        / "jobs"
        / _safe_component(scope["job_id"])
        / "uploads"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_uploaded_files(uploaded_files: Iterable[Any]) -> list[dict[str, Any]]:
    """Persist files with generated names and return backward-compatible docs."""

    documents: list[dict[str, Any]] = []
    upload_dir = _job_upload_dir()
    scope = _scope()

    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name
        suffix = Path(original_name).suffix.lower()
        file_id = _new_id("file")
        stored_name = f"{file_id}{suffix}"
        file_path = (upload_dir / stored_name).resolve()

        # The resolved path must remain inside the scoped upload directory.
        if upload_dir.resolve() not in file_path.parents:
            raise ValueError(f"非法文件路径：{original_name}")

        data = bytes(uploaded_file.getbuffer())
        file_path.write_bytes(data)

        documents.append(
            {
                # Existing Planner/Worker code expects name/path/type.
                "name": original_name,
                "path": str(file_path),
                "type": getattr(uploaded_file, "type", None) or "text/csv",
                # New scoped metadata.
                "file_id": file_id,
                # 兼容现有 Planner/Worker；后续统一只保留 file_id。
                "resource_id": file_id,
                "original_name": original_name,
                "stored_name": stored_name,
                "size_bytes": len(data),
                **scope,
            }
        )

    return documents



# -----------------------------------------------------------------------------
# Stream and message helpers
# -----------------------------------------------------------------------------


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    return str(
        getattr(message, "type", None)
        or getattr(message, "role", None)
        or ""
    ).lower()


def _message_id(message: Any, node: str, content: str) -> str:
    if isinstance(message, dict):
        explicit = message.get("id")
    else:
        explicit = getattr(message, "id", None)
    if explicit:
        return str(explicit)

    # A stable fallback is useful if Streamlit rerenders an already-seen update.
    raw = f"{_scope()['job_id']}|{node}|{content}".encode("utf-8")
    return f"graph_{hashlib.sha256(raw).hexdigest()[:24]}"


def _summarize_step(node: str, delta: dict[str, Any]) -> str:
    if node == "Intake":
        return "已解析当前用户输入"
    if node == "Planner":
        tasks = delta.get("tasks") or []
        cursor = int(delta.get("cursor") or 0)
        return f"任务数：{len(tasks)}，当前序号：{cursor + 1 if tasks else 0}"
    if node == "Worker":
        result = delta.get("current_result") or {}
        name = result.get("section_name") or result.get("task_id") or "当前任务"
        status = result.get("status") or "-"
        return f"{name}：{status}"
    if node == "Verifier":
        return f"审核决策：{delta.get('decision', '-')}"
    if node == "Summarizer":
        return "报告及评价已生成"
    if node == "Exit":
        return "流程结束"
    return ""


def _recover_stream_failure(
    app: Any,
    config: dict[str, Any],
    fallback_interrupt: Any | None,
) -> None:
    try:
        snapshot = app.get_state(config)
        recovered_interrupt = interrupt_from_snapshot(snapshot)
    except Exception:
        recovered_interrupt = fallback_interrupt

    st.session_state["last_run_failed"] = True
    st.session_state["pending_interrupt"] = recovered_interrupt
    if recovered_interrupt is not None:
        _update_job(
            status="waiting",
            pending_interrupt=recovered_interrupt,
        )
    else:
        _update_job(status="failed", pending_interrupt=None)


def _safe_stream_updates(
    app: Any,
    stream_input: Any,
    config: dict[str, Any],
    *,
    fallback_interrupt: Any | None = None,
):
    try:
        yield from app.stream(stream_input, config, stream_mode="updates")
    except RuntimeError as exc:
        _recover_stream_failure(app, config, fallback_interrupt)
        st.error(str(exc))
        st.info(missing_key_message("DEEPSEEK_API_KEY"))
    except Exception as exc:
        _recover_stream_failure(app, config, fallback_interrupt)
        st.exception(exc)


def _handle_interrupt(update: dict[str, Any]) -> bool:
    if "__interrupt__" not in update:
        return False

    interrupt_items = update.get("__interrupt__") or []
    payload = interrupt_items[0].value if interrupt_items else None
    st.session_state["pending_interrupt"] = payload
    _update_job(status="waiting", pending_interrupt=payload)

    if not isinstance(payload, dict):
        text = str(payload or "工作流已暂停，等待输入。")
        _append_ui_message("assistant", text)
        with st.chat_message("assistant"):
            st.write(text)
        return True

    payload_type = payload.get("type")
    if payload_type == "confirm_plan_and_resources" or "structured_msg" in payload:
        _append_ui_message(
            "assistant",
            kind="plan",
            payload=payload,
            message_id=_new_id("interrupt_plan"),
        )
        with st.chat_message("assistant"):
            _render_plan(payload)
        return True

    if payload_type == "verify_result":
        _append_ui_message(
            "assistant",
            kind="verify",
            payload=payload,
            message_id=_new_id("interrupt_verify"),
        )
        with st.chat_message("assistant"):
            _render_verify(payload)
        return True

    if payload_type == "needs_user_input":
        guidance = blocker_guidance(payload) or "需要你的输入后才能继续当前任务。"
        _append_ui_message("assistant", guidance)
        with st.chat_message("assistant"):
            st.write(guidance)
        return True

    guidance = payload.get("guidance_text") or json.dumps(payload, ensure_ascii=False)
    _append_ui_message("assistant", guidance)
    with st.chat_message("assistant"):
        st.write(guidance)
    return True


def _handle_node_delta(node: str, delta: dict[str, Any]) -> None:
    messages = delta.get("messages") or []
    for message in messages:
        # 用户消息已经由输入区渲染；节点恢复后返回的 HumanMessage 不应
        # 再以 assistant 身份展示。
        content = _message_content(message).strip()
        if not is_displayable_assistant_message(_message_role(message), content):
            continue

        msg_id = _message_id(message, node, content)
        before = len(st.session_state["ui_messages"])
        _append_ui_message("assistant", content, message_id=msg_id)
        after = len(st.session_state["ui_messages"])

        # Render only if it was newly appended.
        if after > before:
            with st.chat_message("assistant"):
                st.write(content)


# -----------------------------------------------------------------------------
# Report downloads
# -----------------------------------------------------------------------------


def _report_paths_from_state() -> list[Path]:
    values = _snapshot_values()
    final_result = values.get("final_result") or {}

    candidates: list[Path] = []
    for item in final_result.get("attachments") or []:
        if item:
            candidates.append(Path(str(item)))
    if final_result.get("path"):
        candidates.append(Path(str(final_result["path"])))

    record = _current_job()
    if not candidates and record:
        candidates.extend(Path(path) for path in record.get("report_paths") or [])

    # Compatibility fallback for the current Summarizer/path_manager.
    if not candidates and values:
        try:
            session_dir = Path(get_session_cache_dir(values, _graph_config()))
            report_dir = session_dir / "report"
            candidates.extend(
                [
                    report_dir / "report.docx",
                    report_dir / "report.pdf",
                    report_dir / "report_rewritten.md",
                    report_dir / "report.md",
                ]
            )
        except Exception:
            pass

    try:
        job_root = Path(
            get_session_cache_dir(values, _graph_config())
        ).resolve()
    except Exception:
        return []

    allowed_suffixes = {".docx", ".pdf", ".md"}
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        try:
            resolved.relative_to(job_root)
        except ValueError:
            continue
        if resolved.suffix.lower() not in allowed_suffixes:
            continue
        normalized = str(resolved)
        if normalized in seen:
            continue
        seen.add(normalized)
        if resolved.is_file():
            unique.append(resolved)
    return unique


def _render_report_downloads() -> None:
    paths = _report_paths_from_state()
    if not paths:
        return

    labels = {
        ".docx": "下载 Word",
        ".pdf": "下载 PDF",
        ".md": "下载 Markdown",
    }

    st.subheader("报告文件")
    columns = st.columns(min(3, len(paths)))
    for index, path in enumerate(paths):
        suffix = path.suffix.lower()
        label = labels.get(suffix, f"下载 {path.name}")
        data = path.read_bytes()
        with columns[index % len(columns)]:
            st.download_button(
                label=label,
                data=data,
                file_name=path.name,
                key=f"download_{_scope()['job_id']}_{index}_{path.name}",
            )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------


with st.sidebar:
    st.title("运行配置")

    scope = _scope()
    st.caption("当前作用域")
    st.code(
        "\n".join(
            [
                f"user_id={scope['user_id']}",
                f"conversation_id={scope['conversation_id']}",
                f"job_id={scope['job_id']}",
                f"thread_id={scope['job_id']}",
            ]
        ),
        language="text",
    )
    st.caption("本地默认 user_id=local-user；多用户部署时应由登录系统提供。")

    verifier_mode = st.radio(
        "审核模式",
        options=["manual", "auto"],
        key="verifier_mode",
        format_func=lambda value: "人工审核" if value == "manual" else "自动审核",
        horizontal=True,
    )

    col_compile, col_clear = st.columns(2)
    with col_compile:
        if st.button("编译工作流", use_container_width=True):
            try:
                _compile_workflow(verifier_mode)
            except Exception as exc:
                st.error(f"工作流编译失败：{exc}")
            else:
                st.success("工作流已编译")

    with col_clear:
        if st.button("清除工作流", use_container_width=True):
            st.session_state.pop("app", None)
            st.session_state["compiled_mode"] = None
            st.rerun()

    if st.session_state.get("compiled_mode"):
        current_mode = (
            "人工审核" if st.session_state["compiled_mode"] == "manual" else "自动审核"
        )
        st.info(f"当前已编译：{current_mode}")
        if st.session_state["compiled_mode"] != verifier_mode:
            st.warning("审核模式已更改，请重新编译工作流。")

    st.divider()

    if st.button("新建报告任务", use_container_width=True):
        _start_new_job()
        st.rerun()

    if st.button("新建对话", use_container_width=True):
        _start_new_conversation()
        st.rerun()

    if st.button("清空页面消息", use_container_width=True):
        st.session_state["ui_messages"] = []
        _update_job(ui_messages=[])
        st.rerun()

    if st.session_state.get("pending_interrupt"):
        st.warning("当前任务正在等待你的确认或审核意见。")

    st.divider()
    st.subheader("历史任务")
    job_records = JOBS.list_jobs(st.session_state["user_id"])

    if job_records:
        job_by_id = {record["job_id"]: record for record in job_records}
        selected_job_id = st.selectbox(
            "选择任务",
            options=list(job_by_id),
            format_func=lambda job_id: (
                f"{job_by_id[job_id].get('title', '未命名任务')} · "
                f"{job_by_id[job_id].get('status', '-')}"
            ),
        )
        st.button(
            "恢复选中任务",
            use_container_width=True,
            on_click=_restore_job_from_sidebar,
            args=(selected_job_id,),
        )
    else:
        st.caption("暂无可恢复任务。")

    if restore_error := st.session_state.pop("restore_error", None):
        st.error(f"恢复失败：{restore_error}")
    if restore_success := st.session_state.pop("restore_success", None):
        st.success(restore_success)

    with st.expander("环境配置", expanded=False):
        st.caption("复制 .env.example 为 .env，并在启动前导出环境变量。")
        st.code(
            "set -a\nsource .env\nset +a\nstreamlit run app.py",
            language="bash",
        )


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------


st.title("化工行业多 Agent 报告系统")
st.caption("Streamlit 只负责输入与渲染；LangGraph State 负责业务上下文。")

_render_history()
_render_report_downloads()

if "app" not in st.session_state:
    st.warning("请先在左侧编译工作流。")
    st.stop()

if st.session_state.get("compiled_mode") != verifier_mode:
    st.warning("审核模式与已编译工作流不一致，请重新编译或恢复原模式。")
    st.stop()


chat_value = st.chat_input(
    "请输入报告需求、确认意见或审核反馈……",
    accept_file="multiple",
    file_type=["csv"],
)

if chat_value:
    # Streamlit 1.50 returns a ChatInputValue; keep dict compatibility as well.
    if isinstance(chat_value, str):
        user_text = chat_value
        uploaded_files = []
    elif isinstance(chat_value, dict):
        user_text = str(chat_value.get("text") or "")
        uploaded_files = list(chat_value.get("files") or [])
    else:
        user_text = str(getattr(chat_value, "text", "") or "")
        uploaded_files = list(getattr(chat_value, "files", []) or [])

    new_docs = _save_uploaded_files(uploaded_files) if uploaded_files else []
    existing_values = _snapshot_values()

    graph_text = user_text.strip()
    if not graph_text and new_docs:
        graph_text = "我已上传附件，请结合附件继续处理当前任务。"

    if not graph_text:
        st.warning("请输入内容或上传 CSV 文件。")
        st.stop()

    display_content = graph_text
    if new_docs:
        names = ", ".join(doc["original_name"] for doc in new_docs)
        display_content = f"{display_content}\n\n已上传附件：{names}"

    _ensure_job_record(graph_text, st.session_state["compiled_mode"])
    _update_job(status="running", pending_interrupt=None)
    st.session_state["last_run_failed"] = False

    human_message_id = _new_id("msg")
    _append_ui_message(
        "user",
        display_content,
        message_id=human_message_id,
    )
    with st.chat_message("user"):
        st.write(display_content)

    human_message = HumanMessage(
        id=human_message_id,
        content=graph_text,
        additional_kwargs={
            **_scope(),
            "attachment_ids": [doc["file_id"] for doc in new_docs],
        },
    )

    base_update: dict[str, Any] = {
        **_scope(),
        "current_user_input": graph_text,
        "metadata": _job_metadata(),
    }

    pending_interrupt = st.session_state.get("pending_interrupt")

    if pending_interrupt is not None:
        # Interrupt 的用户反馈由被恢复的节点写入 messages。app.py 只负责
        # 传递本轮 resume 数据，避免同一条 HumanMessage 被写入两次。
        stream_input: Any = Command(
            resume={
                "text": graph_text,
                "message_id": human_message_id,
                "docs": new_docs,
            },
            update=base_update,
        )
    else:
        # 普通输入只提交本轮增量消息和本轮新增附件。历史内容已保存在
        # LangGraph checkpoint 中，docs 由 State.merge_docs reducer 合并。
        state_update: dict[str, Any] = {
            **base_update,
            "messages": [human_message],
            "docs": new_docs,
        }

        # 仅初始化全新的 Job；后续轮次不能重置 tasks/results。
        if not existing_values:
            state_update.update(
                {
                    "tasks": [],
                    "cursor": 0,
                    "current_result": {},
                    "results": [],
                    "decision": "NEXT",
                    "feedback": {},
                    "final_result": {},
                    "planner_action": "",
                    "guidance": {},
                    "worker_state": {},
                    "task_retry_count": {},
                    "replan_count": 0,
                    "workflow_action": "",
                    "plan_revision": 1,
                    "task_revisions": {},
                    "evidence_recovery_count": {},
                    "task_patch_count": {},
                    "job_patch_count": 0,
                    "pending_user_action": {},
                    "plan_patch_history": [],
                    "verification_warnings": [],
                }
            )

        stream_input = state_update

    st.session_state["pending_interrupt"] = None

    with st.status("工作流执行中……", expanded=True) as status:
        for update in _safe_stream_updates(
            st.session_state["app"],
            stream_input,
            _graph_config(),
            fallback_interrupt=pending_interrupt,
        ):
            if _handle_interrupt(update):
                status.update(label="工作流等待输入", state="complete", expanded=True)
                continue

            for node, delta in update.items():
                if not isinstance(delta, dict):
                    continue
                summary = _summarize_step(node, delta)
                st.write(f"**{node}**{f' · {summary}' if summary else ''}")
                _handle_node_delta(node, delta)

        if st.session_state.get("last_run_failed"):
            status.update(label="本轮执行失败", state="error", expanded=True)
        elif st.session_state.get("pending_interrupt") is None:
            status.update(label="本轮执行完成", state="complete", expanded=False)

    if (
        st.session_state.get("pending_interrupt") is None
        and not st.session_state.get("last_run_failed")
    ):
        report_paths = [str(path) for path in _report_paths_from_state()]
        _update_job(
            status="completed",
            pending_interrupt=None,
            report_paths=report_paths,
        )

    # Re-run so report download buttons and the complete UI history are rebuilt
    # from ui_messages and the latest graph state.
    st.rerun()

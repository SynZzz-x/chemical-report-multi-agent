import argparse
import hashlib
import json
import logging
import os
import uuid
from typing import Any

from langgraph.types import Command

from src.graph import WorkFlow, WorkFlowAuto
from src.fatal_errors import build_fatal_system_error
from src.config import (
    configure_langsmith_from_env,
    get_local_user_id,
    missing_key_message,
)
from src.control_messages import blocker_guidance, is_displayable_assistant_message
from src.job_store import JobStore, interrupt_from_snapshot
from src.job_outcome import project_job_outcome
from src.persistence import SQLitePersistence
from src.runtime_config import execution_config


TEST_QUERY = "撰写电力数据集ETTh2与ETTm1统计对比分析报告"

logger = logging.getLogger()
# 确保 logs 目录存在
if not os.path.exists("logs"):
    os.makedirs("logs")


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


def _project_snapshot(
    snapshot: Any,
    *,
    scope: dict[str, str],
    existing_messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    values = dict(getattr(snapshot, "values", {}) or {})
    messages = list(existing_messages)
    known_ids = {str(item.get("id")) for item in messages if item.get("id")}

    for index, message in enumerate(values.get("messages") or []):
        role = _message_role(message)
        content = _message_content(message).strip()
        if not is_displayable_assistant_message(role, content):
            continue

        explicit_id = (
            message.get("id")
            if isinstance(message, dict)
            else getattr(message, "id", None)
        )
        if explicit_id:
            message_id = str(explicit_id)
        else:
            digest = hashlib.sha256(
                f"{scope['job_id']}|{index}|{content}".encode("utf-8")
            ).hexdigest()[:24]
            message_id = f"cli_graph_{digest}"
        if message_id in known_ids:
            continue
        known_ids.add(message_id)
        messages.append(
            {
                "id": message_id,
                "role": "assistant",
                "kind": "text",
                "content": content,
                "payload": {},
                **scope,
            }
        )

    final_result = values.get("final_result") or {}
    report_paths = [
        str(path)
        for path in final_result.get("attachments") or []
        if path
    ]
    if final_result.get("path"):
        report_paths.append(str(final_result["path"]))
    report_paths = list(dict.fromkeys(report_paths))
    return messages, report_paths


def _interrupt_ui_message(
    value: Any,
    *,
    scope: dict[str, str],
    event_id: str | None = None,
) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = value
        content = str(
            blocker_guidance(value)
            or value.get("guidance_text")
            or value.get("content_summary")
            or json.dumps(value, ensure_ascii=False, default=str)
        )
        if value.get("type") == "verify_result":
            kind = "verify"
        elif (
            value.get("type") == "confirm_plan_and_resources"
            or "structured_msg" in value
        ):
            kind = "plan"
        else:
            kind = "text"
    else:
        payload = {}
        content = str(value or "工作流已暂停，等待输入。")
        kind = "text"

    identity = event_id or f"event_{uuid.uuid4().hex}"
    digest = hashlib.sha256(
        f"{scope['job_id']}|{identity}".encode("utf-8")
    ).hexdigest()[:24]
    return {
        "id": f"cli_interrupt_{digest}",
        "role": "assistant",
        "kind": kind,
        "content": content,
        "payload": payload,
        **scope,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent CLI Debugger")
    parser.add_argument("--auto-verify", action="store_true", help="Enable automatic verification")
    parser.add_argument("--thread-id", help="Resume an existing LangGraph thread")
    parser.add_argument("--user-id", help="Override AGENT_USER_ID for Store scope")
    args = parser.parse_args()
    configure_langsmith_from_env()

    # 配置
    resume_requested = args.thread_id is not None
    if resume_requested and not args.thread_id.strip():
        parser.error("--thread-id must not be empty")
    thread_id = args.thread_id if resume_requested else f"job_{uuid.uuid4().hex}"
    user_id = args.user_id or get_local_user_id()
    conversation_id = f"conv_cli_{thread_id}"
    safe_log_id = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in thread_id
    )[:160]
    log_file = f"logs/report_agent_{safe_log_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(), # 输出到控制台
            logging.FileHandler(log_file, encoding='utf-8') # 输出到文件
        ]
    )

    try:
        persistence = SQLitePersistence.open()
    except Exception as exc:
        fatal = build_fatal_system_error(
            exc,
            origin="runner",
            component="SQLitePersistence",
            operation="open",
            task_id=None,
            metadata={"job_id": thread_id},
        )
        logger.error(
            "RUNNER_FATAL failure_id=%s diagnostic_code=%s error_type=%s",
            fatal["failure_id"],
            fatal["diagnostic_code"],
            fatal["subtype"],
        )
        return
    try:
        jobs = JobStore(persistence.store)
        existing_job = jobs.get_job(user_id, thread_id)
        if resume_requested and existing_job is None:
            logger.error("无法恢复：任务不存在或不属于当前用户。")
            return

        if existing_job is not None:
            conversation_id = existing_job["conversation_id"]
            verifier_mode = existing_job.get("verifier_mode") or "manual"
        else:
            verifier_mode = "auto" if args.auto_verify else "manual"

        if verifier_mode == "auto":
            logger.info(">>> Using Automatic Verification Workflow")
            workflow = WorkFlowAuto()
        else:
            logger.info(">>> Using Manual Verification Workflow")
            workflow = WorkFlow()

        app = workflow.compile(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
        )
        config = execution_config({
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "job_id": thread_id,
            },
        })
        if (
            resume_requested
            and persistence.checkpointer.get_tuple(config) is None
        ):
            logger.error("无法恢复：任务记录存在，但对应 checkpoint 缺失。")
            return

        snapshot = app.get_state(config)
        config = execution_config(config, getattr(snapshot, "values", {}) or {})
        last_interrupt_value = interrupt_from_snapshot(snapshot)
        is_interrupted = last_interrupt_value is not None
        job_record_created = existing_job is not None
        scope = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "job_id": thread_id,
        }
        ui_messages = list(
            existing_job.get("ui_messages") or []
            if existing_job
            else []
        )
        report_paths = list(
            existing_job.get("report_paths") or []
            if existing_job
            else []
        )

        def persist_projection(current_snapshot: Any | None, **changes: Any) -> None:
            nonlocal ui_messages, report_paths
            if current_snapshot is not None:
                projected_messages, projected_paths = _project_snapshot(
                    current_snapshot,
                    scope=scope,
                    existing_messages=ui_messages,
                )
                ui_messages = projected_messages
                if projected_paths:
                    report_paths = projected_paths
            changes["ui_messages"] = list(ui_messages)
            changes["report_paths"] = list(report_paths)
            jobs.update_job(user_id, thread_id, **changes)

        if job_record_created:
            projection = project_job_outcome(
                dict(snapshot.values or {}),
                last_interrupt_value,
                graph_incomplete=bool(getattr(snapshot, "next", ()) or ()),
            )
            restore_changes: dict[str, Any] = {
                "pending_interrupt": last_interrupt_value,
            }
            if (
                existing_job.get("status") != "failed"
                or projection["status"] != "completed"
            ):
                restore_changes.update(projection)
            persist_projection(snapshot, **restore_changes)

        logger.info(f"=== Agent CLI Debugger (Thread: {thread_id}) ===")
        logger.info(f"User scope: {user_id}")
        logger.info(f"Log file: {log_file}")
        logger.info("输入消息开始对话，输入 'q' 退出。")
        logger.info("如果遇到中断，请输入内容以 resume。")
        if is_interrupted:
            logger.info(f"[System] Restored pending interrupt: {last_interrupt_value}")

        def recover_after_failure(error: BaseException):
            nonlocal is_interrupted, last_interrupt_value
            fatal = build_fatal_system_error(
                error,
                origin="runner",
                component="CLIRunner",
                operation="graph_stream",
                task_id=None,
                metadata={"job_id": thread_id},
            )
            is_interrupted = False
            last_interrupt_value = None
            if not job_record_created:
                logger.error(
                    "RUNNER_FATAL failure_id=%s diagnostic_code=%s error_type=%s",
                    fatal["failure_id"],
                    fatal["diagnostic_code"],
                    fatal["subtype"],
                )
                return

            try:
                persist_projection(
                    None,
                    status="failed",
                    pending_interrupt=None,
                    fatal_system_error=fatal,
                )
            except Exception as store_exc:
                logger.error(
                    "[Persistence Error] Could not save runner fatal state: error_type=%s",
                    type(store_exc).__name__,
                )

        while True:
            try:
                user_input = input(f"\nUser [{thread_id}]> ").strip()
            except EOFError:
                break

            logger.info(f"\nUser> {user_input}")
            if user_input.lower() == "q":
                break
            if not user_input:
                continue

            user_ui_message = {
                "id": f"cli_user_{uuid.uuid4().hex}",
                "role": "user",
                "kind": "text",
                "content": user_input,
                "payload": {},
                **scope,
            }
            candidate_ui_messages = [*ui_messages, user_ui_message]

            try:
                if not job_record_created:
                    jobs.create_job(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        job_id=thread_id,
                        title=user_input,
                        verifier_mode=verifier_mode,
                        ui_messages=list(ui_messages),
                    )
                    job_record_created = True

                jobs.update_job(
                    user_id,
                    thread_id,
                    status="running",
                    pending_interrupt=None,
                    ui_messages=candidate_ui_messages,
                    report_paths=list(report_paths),
                )
            except Exception as store_exc:
                logger.error(
                    f"[Persistence Error] Could not start job: {store_exc}"
                )
                continue

            ui_messages = candidate_ui_messages
            state_update = {
                **scope,
                "current_user_input": user_input,
                "metadata": dict(scope),
            }
            if is_interrupted:
                logger.info(f"[System] Resuming with input: {user_input}")
                inputs = Command(
                    resume=user_input,
                    update=state_update,
                )
                is_interrupted = False
                last_interrupt_value = None
            else:
                inputs = {
                    **state_update,
                    "messages": [{"role": "user", "content": user_input}],
                }

            logger.info("-" * 20 + " Stream Start " + "-" * 20)
            try:
                for chunk in app.stream(inputs, config=config):
                    logger.info(f"Chunk: {chunk}")
                    if "__interrupt__" in chunk:
                        is_interrupted = True
                        interrupt_item = chunk["__interrupt__"][0]
                        last_interrupt_value = interrupt_item.value
                        checkpoint_id = None
                        try:
                            interrupt_snapshot = app.get_state(config)
                            snapshot_config = (
                                getattr(interrupt_snapshot, "config", {}) or {}
                            )
                            checkpoint_id = (
                                snapshot_config.get("configurable", {}) or {}
                            ).get("checkpoint_id")
                        except Exception:
                            pass
                        interrupt_id = getattr(interrupt_item, "id", None)
                        event_id = (
                            f"{interrupt_id}|{checkpoint_id}"
                            if interrupt_id and checkpoint_id
                            else interrupt_id or checkpoint_id
                        )
                        interrupt_message = _interrupt_ui_message(
                            last_interrupt_value,
                            scope=scope,
                            event_id=event_id,
                        )
                        if not any(
                            item.get("id") == interrupt_message["id"]
                            for item in ui_messages
                        ):
                            ui_messages.append(interrupt_message)
                        logger.info(
                            f"\n[System] Interrupted! Value: {last_interrupt_value}"
                        )
            except RuntimeError as exc:
                recover_after_failure(exc)
                logger.error("[Config Error] error_type=%s", type(exc).__name__)
                logger.error(missing_key_message("DEEPSEEK_API_KEY"))
            except Exception as exc:
                recover_after_failure(exc)
                logger.error("[Runner Error] error_type=%s", type(exc).__name__)
            else:
                completed_snapshot = None
                try:
                    completed_snapshot = app.get_state(config)
                    last_interrupt_value = interrupt_from_snapshot(
                        completed_snapshot
                    )
                    is_interrupted = last_interrupt_value is not None
                except Exception as state_exc:
                    logger.error(
                        "[Recovery Error] Could not project final checkpoint "
                        f"state: {state_exc}"
                    )

                try:
                    outcome = project_job_outcome(
                        dict(getattr(completed_snapshot, "values", {}) or {}),
                        last_interrupt_value,
                        graph_incomplete=completed_snapshot is None,
                    )
                    persist_projection(completed_snapshot, **outcome)
                    if outcome["status"] == "failed":
                        diagnostic = outcome.get("fatal_system_error") or {}
                        logger.error(
                            "[System] Workflow failed (diagnostic_code=%s)",
                            diagnostic.get("diagnostic_code") or "WORKFLOW_FAILED",
                        )
                except Exception as store_exc:
                    logger.error(
                        "[Persistence Error] Graph execution completed, but "
                        f"job metadata could not be saved: {store_exc}"
                    )

            logger.info("-" * 20 + " Stream End " + "-" * 20)
    finally:
        persistence.close()

if __name__ == "__main__":
    main()

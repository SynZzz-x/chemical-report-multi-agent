import uuid
import os
import argparse
import logging
from langgraph.types import Command
from src.graph import WorkFlow, WorkFlowAuto
from src.config import (
    configure_langsmith_from_env,
    get_local_user_id,
    missing_key_message,
)
from src.job_store import JobStore, interrupt_from_snapshot
from src.persistence import SQLitePersistence


TEST_QUERY = "撰写电力数据集ETTh2与ETTm1统计对比分析报告"

logger = logging.getLogger()
# 确保 logs 目录存在
if not os.path.exists("logs"):
    os.makedirs("logs")

def main():
    parser = argparse.ArgumentParser(description="Agent CLI Debugger")
    parser.add_argument("--auto-verify", action="store_true", help="Enable automatic verification")
    parser.add_argument("--thread-id", help="Resume an existing LangGraph thread")
    parser.add_argument("--user-id", help="Override AGENT_USER_ID for Store scope")
    args = parser.parse_args()
    configure_langsmith_from_env()

    # 配置
    thread_id = args.thread_id or f"job_{uuid.uuid4().hex}"
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

    persistence = SQLitePersistence.open()
    try:
        jobs = JobStore(persistence.store)
        existing_job = jobs.get_job(user_id, thread_id)
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
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "job_id": thread_id,
            },
        }
        snapshot = app.get_state(config)
        last_interrupt_value = interrupt_from_snapshot(snapshot)
        is_interrupted = last_interrupt_value is not None
        job_record_created = existing_job is not None

        if is_interrupted and job_record_created:
            jobs.update_job(
                user_id,
                thread_id,
                status="waiting",
                pending_interrupt=last_interrupt_value,
            )

        logger.info(f"=== Agent CLI Debugger (Thread: {thread_id}) ===")
        logger.info(f"User scope: {user_id}")
        logger.info(f"Log file: {log_file}")
        logger.info("输入消息开始对话，输入 'q' 退出。")
        logger.info("如果遇到中断，请输入内容以 resume。")
        if is_interrupted:
            logger.info(f"[System] Restored pending interrupt: {last_interrupt_value}")

        def recover_after_failure(previous_interrupt):
            nonlocal is_interrupted, last_interrupt_value

            try:
                recovered = interrupt_from_snapshot(app.get_state(config))
            except Exception as state_exc:
                logger.error(
                    f"[Recovery Error] Could not reload checkpoint state: {state_exc}"
                )
                recovered = (
                    last_interrupt_value
                    if is_interrupted
                    else previous_interrupt
                )

            is_interrupted = recovered is not None
            last_interrupt_value = recovered
            if not job_record_created:
                return

            try:
                if is_interrupted:
                    jobs.update_job(
                        user_id,
                        thread_id,
                        status="waiting",
                        pending_interrupt=last_interrupt_value,
                    )
                else:
                    jobs.update_job(
                        user_id,
                        thread_id,
                        status="failed",
                        pending_interrupt=None,
                    )
            except Exception as store_exc:
                logger.error(
                    f"[Persistence Error] Could not save failure state: {store_exc}"
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

            previous_interrupt = (
                last_interrupt_value
                if is_interrupted
                else None
            )
            try:
                if not job_record_created:
                    jobs.create_job(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        job_id=thread_id,
                        title=user_input,
                        verifier_mode=verifier_mode,
                    )
                    job_record_created = True

                jobs.update_job(
                    user_id,
                    thread_id,
                    status="running",
                    pending_interrupt=None,
                )

                if is_interrupted:
                    logger.info(f"[System] Resuming with input: {user_input}")
                    inputs = Command(resume=user_input)
                    is_interrupted = False
                    last_interrupt_value = None
                else:
                    inputs = {"messages": [{"role": "user", "content": user_input}]}

                logger.info("-" * 20 + " Stream Start " + "-" * 20)
                for chunk in app.stream(inputs, config=config):
                    logger.info(f"Chunk: {chunk}")
                    if "__interrupt__" in chunk:
                        is_interrupted = True
                        last_interrupt_value = chunk["__interrupt__"][0].value
                        jobs.update_job(
                            user_id,
                            thread_id,
                            status="waiting",
                            pending_interrupt=last_interrupt_value,
                        )
                        logger.info(
                            f"\n[System] Interrupted! Value: {last_interrupt_value}"
                        )

                if not is_interrupted:
                    jobs.update_job(
                        user_id,
                        thread_id,
                        status="completed",
                        pending_interrupt=None,
                    )
            except RuntimeError as exc:
                recover_after_failure(previous_interrupt)
                logger.error(f"\n[Config Error] {exc}")
                logger.error(missing_key_message("OPENAI_API_KEY"))
            except Exception as exc:
                recover_after_failure(previous_interrupt)
                logger.error(f"\n[Error] {exc}")

            logger.info("-" * 20 + " Stream End " + "-" * 20)
    finally:
        persistence.close()

if __name__ == "__main__":
    main()

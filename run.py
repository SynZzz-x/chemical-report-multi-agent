import uuid
import os
import argparse
import logging
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from src.graph import WorkFlow, WorkFlowAuto
from src.config import configure_langsmith_from_env, missing_key_message


TEST_QUERY = "撰写电力数据集ETTh2与ETTm1统计对比分析报告"

logger = logging.getLogger()
# 确保 logs 目录存在
if not os.path.exists("logs"):
    os.makedirs("logs")

def main():
    parser = argparse.ArgumentParser(description="Agent CLI Debugger")
    parser.add_argument("--auto-verify", action="store_true", help="Enable automatic verification")
    args = parser.parse_args()
    configure_langsmith_from_env()
    
    # 配置
    thread_id = str(uuid.uuid4())
    log_file = f"logs/report_agent_{thread_id}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(), # 输出到控制台
            logging.FileHandler(log_file, encoding='utf-8') # 输出到文件
        ]
    )

    # 初始化 Graph
    if args.auto_verify:
        logger.info(">>> Using Automatic Verification Workflow")
        workflow = WorkFlowAuto()
    else:
        logger.info(">>> Using Manual Verification Workflow")
        workflow = WorkFlow()
        
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"=== Agent CLI Debugger (Thread: {thread_id}) ===")
    logger.info(f"Log file: {log_file}")
    logger.info("输入消息开始对话，输入 'q' 退出。")
    logger.info("如果遇到中断，请输入内容以 resume。")
    
    # 状态标记：当前是否处于中断挂起状态
    is_interrupted = False
    last_interrupt_value = None

    while True:
        # 1. 获取用户输入
        try:
            user_input = input(f"\nUser [{thread_id}]> ").strip()
        except EOFError:
            break
            
        logger.info(f"\nUser> {user_input}")
        
        if user_input.lower() == 'q':
            break
            
        # 2. 构造输入 payload
        if is_interrupted:
            # 如果处于中断状态，使用 Command(resume=...)
            logger.info(f"[System] Resuming with input: {user_input}")
            inputs = Command(resume=user_input)
            # 重置中断状态，等待新一轮流式输出判断
            is_interrupted = False
            last_interrupt_value = None
        else:
            # 正常对话输入
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            
        # 3. 执行流式调用
        logger.info("-" * 20 + " Stream Start " + "-" * 20)
        try:
            for chunk in app.stream(inputs, config=config):
                # 记录原始 chunk
                logger.info(f"Chunk: {chunk}")
                
                # 检查中断信号
                if "__interrupt__" in chunk:
                    is_interrupted = True
                    last_interrupt_value = chunk["__interrupt__"][0].value
                    logger.info(f"\n[System] Interrupted! Value: {last_interrupt_value}")
                    # 注意：在 stream 循环中遇到中断 chunk 后，通常循环会继续（如果还有其他并行分支）或结束
                    # 我们标记状态即可，等待下一次循环处理 resume
                    
        except RuntimeError as e:
            logger.error(f"\n[Config Error] {e}")
            logger.error(missing_key_message("OPENAI_API_KEY"))
        except Exception as e:
            logger.error(f"\n[Error] {e}")
            
        logger.info("-" * 20 + " Stream End " + "-" * 20)

if __name__ == "__main__":
    main()

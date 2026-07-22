from langchain_core.runnables import RunnableConfig

from ..state import State


def exiting(state: State, config: RunnableConfig, **kwargs):
    """占位主函数：Exit/Exiting 节点入口（验证用，与 planner 相同结构）"""
    return state
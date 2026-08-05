"""Test stubs for optional agent runtime dependencies.

The unit tests in this suite exercise pure helper logic. They should run in a
lightweight environment without importing real LangChain, LangGraph, or model
clients.
"""

from __future__ import annotations

import sys
import types


class _Message:
    def __init__(self, content: str = "", **kwargs):
        self.content = content
        self.additional_kwargs = kwargs


class _AIMessage(_Message):
    type = "ai"


class _HumanMessage(_Message):
    type = "human"


class _SystemMessage(_Message):
    type = "system"


class _ToolMessage(_Message):
    type = "tool"


class _ChatPromptTemplate:
    @classmethod
    def from_messages(cls, messages):
        return cls()

    def format_messages(self, **kwargs):
        return []

    def __or__(self, other):
        return other

    @classmethod
    def from_template(cls, template):
        return cls()


def _install_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


langchain_openai = _install_module("langchain_openai")
langchain_openai.ChatOpenAI = object

langchain_core = _install_module("langchain_core")
langchain_core.runnables = _install_module("langchain_core.runnables")
langchain_core.runnables.RunnableConfig = dict
langchain_core.messages = _install_module("langchain_core.messages")
langchain_core.messages.AIMessage = _AIMessage
langchain_core.messages.SystemMessage = _SystemMessage
langchain_core.messages.HumanMessage = _HumanMessage
langchain_core.messages.ToolMessage = _ToolMessage
langchain_core.messages.AnyMessage = _Message
langchain_core.messages.BaseMessage = _Message
langchain_core.prompts = _install_module("langchain_core.prompts")
langchain_core.prompts.ChatPromptTemplate = _ChatPromptTemplate
langchain_core.prompts.PromptTemplate = _ChatPromptTemplate
langchain_core.tools = _install_module("langchain_core.tools")
langchain_core.tools.BaseTool = object

langgraph = _install_module("langgraph")
langgraph.types = _install_module("langgraph.types")
langgraph.types.interrupt = lambda payload: None
langgraph.graph = _install_module("langgraph.graph")
langgraph.graph.StateGraph = object
langgraph.graph.END = "__end__"
langgraph.graph.message = _install_module("langgraph.graph.message")
langgraph.graph.message.BaseMessage = _Message
langgraph.graph.message.add_messages = lambda left, right: (left or []) + (right or [])

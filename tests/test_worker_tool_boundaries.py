from dataclasses import dataclass

import pytest

from src.nodes.worker.agent.graph import AutonomousToolNode, ChemicalKnowledgeBaseTool, ToolManager
from src.tool_names import canonical_tool_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ChemicalKnowledgeBaseTool", "chemical_knowledge_base_tool"),
        ("chemical_knowledge_base_tool", "chemical_knowledge_base_tool"),
        ("CSVTool", "csv_analysis_tool"),
        ("csv_analysis_tool", "csv_analysis_tool"),
        ("ChartTool", "chart_generator_tool"),
        ("chart_generator_tool", "chart_generator_tool"),
        ("SpiderTool", "spider_tool"),
        ("spider_tool", "spider_tool"),
        ("spider", "spider"),
        ("spidertool", "spidertool"),
        (" CustomExactTool ", "CustomExactTool"),
    ],
)
def test_canonical_tool_name_maps_only_supported_exact_aliases(name, expected):
    assert canonical_tool_name(name) == expected


@pytest.mark.parametrize("name", ["", "   ", None])
def test_canonical_tool_name_rejects_blank_or_non_string_names(name):
    assert canonical_tool_name(name) is None


def test_worker_knowledge_base_schema_is_query_only():
    schema = ChemicalKnowledgeBaseTool.get_args_schema(object())

    assert "operation" not in schema.model_fields
    assert set(schema.model_fields) == {"query", "top_k", "doc_type_filter"}


def test_tool_manager_initializes_only_tools_required_by_task():
    initialized = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def validate_task(self, task):
            return True

    def tool_class(name):
        class Tool:
            def __init__(self, config):
                initialized.append(name)
                self.name = name

            def validate_task(self, task):
                return True

        return Tool

    @dataclass
    class Config:
        ENABLED_TOOLS: tuple[str, ...] = (
            "ChemicalKnowledgeBaseTool",
            "CSVTool",
            "ChartTool",
            "SpiderTool",
        )

    manager = ToolManager.__new__(ToolManager)
    manager.config = Config()
    manager.tool_classes = {
        "ChemicalKnowledgeBaseTool": tool_class("chemical_knowledge_base_tool"),
        "CSVTool": tool_class("csv_analysis_tool"),
        "ChartTool": tool_class("chart_generator_tool"),
        "SpiderTool": tool_class("spider_tool"),
    }

    tools = manager.get_available_tools_for_task({"use_rag": True})

    assert [tool.name for tool in tools] == ["chemical_knowledge_base_tool"]
    assert initialized == ["chemical_knowledge_base_tool"]


def test_tool_manager_initializes_nothing_for_text_only_task():
    initialized = []

    class Tool:
        def __init__(self, config):
            initialized.append("unexpected")
            self.name = "unexpected"

        def validate_task(self, task):
            return True

    @dataclass
    class Config:
        ENABLED_TOOLS: tuple[str, ...] = ("SpiderTool",)

    manager = ToolManager.__new__(ToolManager)
    manager.config = Config()
    manager.tool_classes = {"SpiderTool": Tool}

    assert manager.get_available_tools_for_task({"task_type": "summary"}) == []
    assert initialized == []


def _spider_only_manager(initialized):
    class Spider:
        def __init__(self, config):
            initialized.append("spider_tool")
            self.name = "spider_tool"

        def validate_task(self, task):
            return True

    @dataclass
    class Config:
        ENABLED_TOOLS: tuple[str, ...] = ("SpiderTool",)

    manager = ToolManager.__new__(ToolManager)
    manager.config = Config()
    manager.tool_classes = {"SpiderTool": Spider}
    return manager


def test_tool_manager_never_substring_matches_spider_requirement():
    initialized = []
    manager = _spider_only_manager(initialized)

    tools = manager.get_available_tools_for_task(
        {"use_web": False, "tool_requirements": ["spider"]}
    )

    assert tools == []
    assert initialized == []


@pytest.mark.parametrize("requirement", ["SpiderTool", "spider_tool"])
def test_tool_manager_accepts_only_supported_exact_spider_aliases(requirement):
    initialized = []
    manager = _spider_only_manager(initialized)

    tools = manager.get_available_tools_for_task(
        {"use_web": True, "tool_requirements": [requirement]}
    )

    assert [tool.name for tool in tools] == ["spider_tool"]
    assert initialized == ["spider_tool"]


@pytest.mark.parametrize(
    "task",
    [
        {"use_web": False, "tool_requirements": ["SpiderTool"]},
        {"tool_requirements": ["spider_tool"]},
        {
            "use_web": False,
            "tool_requirements": ["spider_tool"],
            "visualization": {"allow_web_fallback": "true"},
        },
    ],
)
def test_tool_manager_rejects_legacy_spider_metadata_without_explicit_web_access(task):
    initialized = []
    manager = _spider_only_manager(initialized)

    tools = manager.get_available_tools_for_task(task)

    assert tools == []
    assert initialized == []


@pytest.mark.parametrize(
    "authorization",
    [
        {"use_web": True},
        {"allow_web_fallback": True},
        {"visualization": {"allow_web_fallback": True}},
    ],
)
def test_tool_manager_accepts_spider_only_with_explicit_web_access(authorization):
    initialized = []
    manager = _spider_only_manager(initialized)

    tools = manager.get_available_tools_for_task(
        {"tool_requirements": ["spider_tool"], **authorization}
    )

    assert [tool.name for tool in tools] == ["spider_tool"]
    assert initialized == ["spider_tool"]


def test_worker_web_feedback_preserves_custom_spidertool_identifier():
    execution_task, _, _ = AutonomousToolNode._prepare_execution_task(
        {
            "task_id": "T1",
            "use_web": True,
            "tool_requirements": ["spidertool", "SpiderTool", "spider_tool"],
        },
        {"execution_feedback": {"allow_web": False}},
    )

    assert execution_task["tool_requirements"] == ["spidertool"]


def test_rag_is_prefetched_before_autonomous_tool_selection():
    calls = []

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def invoke(self, parameters):
            calls.append(parameters)
            return {"success": True, "evidence": []}

    prefetched = AutonomousToolNode._prefetch_rag(
        {"use_rag": True, "query": "反应温度 熔融指数"},
        [KnowledgeTool()],
    )

    assert calls == [{"query": "反应温度 熔融指数", "top_k": 5}]
    assert prefetched[0]["prefetched"] is True
    assert prefetched[0]["tool"] == "chemical_knowledge_base_tool"

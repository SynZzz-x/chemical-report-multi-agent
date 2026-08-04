from dataclasses import dataclass

from src.nodes.worker.agent.graph import AutonomousToolNode, ChemicalKnowledgeBaseTool, ToolManager


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

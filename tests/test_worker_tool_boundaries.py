from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.evidence.models import EvidenceRecord
from src.nodes.worker.agent import graph as worker_graph_module
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


def _concept_graph_node(monkeypatch, provider_events):
    settings = SimpleNamespace(
        web_fallback=True,
        web_allowed_source_classes=("government",),
        web_max_queries=3,
    )
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(concept_graph_settings=settings),
    )

    class Provider:
        def __init__(self, *args, **kwargs):
            provider_events.append("constructed")

        def search(self, queries):
            provider_events.append(("searched", tuple(queries)))
            return (
                EvidenceRecord(
                    evidence_id="web",
                    source_type="web",
                    title="Public source",
                    supporting_text="alpha beta",
                    url="https://example.org/source",
                    accessed_at="2026-08-05T00:00:00Z",
                ),
            )

    monkeypatch.setattr(
        "src.evidence.web.LegacySpiderWebEvidenceProvider",
        Provider,
    )

    class GraphTool:
        def execute(self, task, evidence, output_dir):
            return {"success": True, "record_count": len(evidence.records)}

    monkeypatch.setattr(
        "src.nodes.worker.tools.concept_graph_tool.ConceptGraphTool",
        GraphTool,
    )
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(
        SPIDER_ENABLED=True,
        SPIDER_DIR="/tmp/spider",
        MAX_SPIDER_RESULTS=3,
        CHARTS_DIR="/tmp/charts",
    )
    return node


@pytest.mark.parametrize(
    "task",
    [
        {
            "task_name": "Causal graph",
            "use_web": False,
            "visualization": {
                "kind": "causal",
                "required_concepts": ["alpha"],
                "web_queries": ["alpha"],
            },
        },
        {
            "task_name": "Flow chart",
            "use_web": False,
            "allow_web_fallback": False,
            "visualization": {
                "kind": "flowchart",
                "required_concepts": ["alpha"],
                "web_queries": ["alpha"],
            },
        },
        {
            "task_name": "Legacy graph",
            "use_web": "false",
            "visualization": {
                "kind": "causal",
                "required_concepts": ["alpha"],
                "web_queries": ["alpha"],
                "allow_web_fallback": "false",
            },
        },
    ],
)
def test_concept_graph_never_constructs_web_provider_without_explicit_authorization(
    monkeypatch, task
):
    provider_events = []
    node = _concept_graph_node(monkeypatch, provider_events)

    request = node._concept_graph_request(task)
    _, coverage, graph_result = node._prepare_concept_graph(task, [])

    assert request["allow_web_fallback"] is False
    assert provider_events == []
    assert coverage.status == "unavailable"
    assert graph_result["success"] is False


@pytest.mark.parametrize(
    "authorization",
    [
        {"use_web": True},
        {"allow_web_fallback": True},
        {"visualization_allow_web": True},
    ],
)
def test_concept_graph_uses_web_provider_only_with_explicit_authorization(
    monkeypatch, authorization
):
    provider_events = []
    node = _concept_graph_node(monkeypatch, provider_events)
    visualization = {
        "kind": "causal",
        "required_concepts": ["alpha"],
        "web_queries": ["alpha"],
    }
    task = {"task_name": "Authorized graph", "visualization": visualization}
    if authorization.get("visualization_allow_web") is True:
        visualization["allow_web_fallback"] = True
    else:
        task.update(authorization)

    request = node._concept_graph_request(task)
    _, coverage, graph_result = node._prepare_concept_graph(task, [])

    assert request["allow_web_fallback"] is True
    assert provider_events[0] == "constructed"
    assert provider_events[1][0] == "searched"
    assert coverage.status == "sufficient"
    assert graph_result["success"] is True


def test_concept_graph_uses_sufficient_internal_rag_without_public_web(monkeypatch):
    provider_events = []
    node = _concept_graph_node(monkeypatch, provider_events)
    task = {
        "task_name": "RAG graph",
        "use_web": False,
        "visualization": {
            "kind": "causal",
            "required_concepts": ["alpha", "beta"],
        },
    }
    rag_call = {
        "tool": "chemical_knowledge_base_tool",
        "success": True,
        "parameters": {"query": "alpha beta"},
        "full_result": {
            "evidence": [
                {
                    "title": "Internal document",
                    "source": "/job/internal.pdf",
                    "content": "alpha influences beta",
                }
            ]
        },
    }

    evidence, coverage, graph_result = node._prepare_concept_graph(task, [rag_call])

    assert provider_events == []
    assert coverage.status == "sufficient"
    assert evidence.records[0].source_type == "rag"
    assert graph_result == {"success": True, "record_count": 1}


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

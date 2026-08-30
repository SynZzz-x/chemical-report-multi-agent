from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evidence.models import EvidenceBundle, EvidenceRecord
from src.evidence.normalizer import normalize_rag_tool_calls
from src.evidence.query_identity import normalize_query_identity, query_fingerprint
from src.nodes.worker.agent import graph as worker_graph_module
from src.nodes.worker.agent.graph import AutonomousToolNode, ChemicalKnowledgeBaseTool, ToolManager
from src.tool_names import canonical_tool_name


def test_query_identity_is_textual_only():
    assert normalize_query_identity("  聚乙烯   质量异常 ") == "聚乙烯 质量异常"
    assert normalize_query_identity("聚乙烯质量问题") != normalize_query_identity(
        "聚乙烯 质量异常"
    )
    assert normalize_query_identity(" PE  Quality ") == "pe quality"
    assert query_fingerprint(" PE  Quality ") == query_fingerprint("pe quality")


def test_worker_prefetch_and_tool_loop_share_query_identity_authority():
    import inspect

    prefetch_source = inspect.getsource(AutonomousToolNode._prefetch_rag)
    loop_source = inspect.getsource(AutonomousToolNode._execute_tool_loop)

    assert "normalize_query_identity(" in prefetch_source
    assert "normalize_query_identity(" in loop_source
    assert ".casefold()" not in prefetch_source
    assert ".casefold()" not in loop_source


def test_worker_evidence_prompts_require_adjacent_material_claim_citations():
    import inspect

    template = (
        Path(__file__).parents[1] / "src" / "prompts" / "worker_system_template.md"
    ).read_text(encoding="utf-8")
    context_source = inspect.getsource(AutonomousToolNode._evidence_context_for_generation)
    binding_source = inspect.getsource(AutonomousToolNode._bind_claims_to_evidence)

    required_phrase = "Every material quantitative, causal, priority/superlative, or strong operational assertion"
    assert required_phrase in template
    assert required_phrase in context_source
    assert required_phrase in binding_source


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


def test_chart_subtask_preserves_parent_llm_observability_scope():
    captured = {}

    class ChartGenerator:
        def process_planner_input(self, planner_input):
            captured.update(planner_input)
            return {"current_result": {"figures": [], "tables": []}}

    tool = worker_graph_module.ChartTool.__new__(worker_graph_module.ChartTool)
    tool.config = SimpleNamespace(CHART_CACHE_ENABLED=False)
    tool.chart_generator = ChartGenerator()

    tool._generate_charts_for_dataset(
        {
            "task_id": "T4",
            "_job_id": "job-4",
            "_plan_revision": 3,
            "_task_revision": 2,
        },
        "/tmp/nonexistent-observability.csv",
        ["line"],
        "趋势图",
        None,
        None,
        "blue",
        False,
        False,
        0,
        1,
    )

    chart_task = captured["tasks"][0]
    assert chart_task["task_id"].startswith("T4_")
    assert chart_task["_observability_task_id"] == "T4"
    assert chart_task["_job_id"] == "job-4"
    assert chart_task["_plan_revision"] == 3
    assert chart_task["_task_revision"] == 2


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


def test_tool_manager_excludes_tool_that_failed_runtime_initialization():
    class UnavailableSpider:
        def __init__(self, _config):
            self.name = "spider_tool"

        def is_available(self):
            return False

        def validate_task(self, _task):
            return True

    @dataclass
    class Config:
        ENABLED_TOOLS: tuple[str, ...] = ("SpiderTool",)

    manager = ToolManager.__new__(ToolManager)
    manager.config = Config()
    manager.tool_classes = {"SpiderTool": UnavailableSpider}

    assert manager.get_available_tools_for_task(
        {"use_web": True, "tool_requirements": ["spider_tool"]}
    ) == []


def test_length_rewrite_execution_disables_tools_and_carries_original_result():
    source_result = {
        "task_id": "T1",
        "text_output": "需要压缩的原正文。[E1]",
        "citations": [{"evidence_id": "E1"}],
        "tables": [{"title": "参数表"}],
    }

    execution_task, instructions, _ = AutonomousToolNode._prepare_execution_task(
        {
            "task_id": "T1",
            "use_rag": True,
            "use_web": True,
            "query": "原查询",
            "tool_requirements": ["chemical_knowledge_base_tool", "spider_tool"],
        },
        {
            "execution_feedback": {
                "mode": "length_rewrite",
                "instructions": "压缩到 500 字以内。",
                "source_result": source_result,
            }
        },
    )

    assert execution_task["use_rag"] is False
    assert execution_task["use_web"] is False
    assert execution_task["query"] == ""
    assert execution_task["tool_requirements"] == []
    assert execution_task["_length_rewrite_source_result"] == source_result
    assert "不得新增事实" in instructions


def test_length_rewrite_targets_safety_margin_below_hard_maximum():
    _, instructions, _ = AutonomousToolNode._prepare_execution_task(
        {
            "task_id": "T1",
            "task_description": "撰写工艺分析，字数：2000-2500字。",
        },
        {
            "execution_feedback": {
                "mode": "length_rewrite",
                "instructions": "压缩到要求范围内。",
                "source_result": {"task_id": "T1", "text_output": "原正文"},
            }
        },
    )

    assert "本次缓冲目标：不超过 2300 字" in instructions
    assert "当前正文确定性计数：3 字" in instructions
    assert "硬性范围：最低 2000 字，最高 2500 字" in instructions
    assert "硬上限 2500 字" in instructions


def test_length_rewrite_safety_margin_uses_shared_runtime_config(monkeypatch):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(length_rewrite_safety_ratio=0.9),
    )

    _, instructions, _ = AutonomousToolNode._prepare_execution_task(
        {
            "task_id": "T1",
            "task_description": "字数：2000-2500字。",
        },
        {
            "execution_feedback": {
                "mode": "length_rewrite",
                "source_result": {"task_id": "T1", "text_output": "原正文"},
            }
        },
    )

    assert "目标有效字数不超过 2250 字" in instructions


def test_short_length_rewrite_never_triggers_generic_detail_retry(monkeypatch):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(rag_max_queries=3)
        ),
    )

    class Model:
        model_name = "deepseek-chat"

        def __init__(self):
            self.calls = 0

        def invoke(self, _messages, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content="短篇改写结果。", tool_calls=[])

    model = Model()
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=3)
    node.llm_client = model

    content, _, _ = node._execute_tool_loop(
        model,
        [],
        [],
        {
            "task_id": "T1",
            "task_description": "不超过50字。",
            "_llm_purpose": "length_rewrite",
            "_length_rewrite_source_result": {"text_output": "原始正文。"},
        },
    )

    assert content == "短篇改写结果。"
    assert model.calls == 1


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


def test_worker_job_web_revocation_overrides_task_and_recovery_authorization():
    execution_task, _, _ = AutonomousToolNode._prepare_execution_task(
        {
            "task_id": "T1",
            "use_web": True,
            "allow_web_fallback": True,
            "tool_requirements": ["chemical_knowledge_base_tool", "spider_tool"],
            "visualization": {
                "allow_web_fallback": True,
                "web_queries": ["latest public source"],
            },
        },
        {"execution_feedback": {"allow_web": True}},
    )

    enforced = AutonomousToolNode._enforce_job_web_policy(
        execution_task,
        web_authorized=False,
    )

    assert enforced["use_web"] is False
    assert enforced["allow_web_fallback"] is False
    assert enforced["tool_requirements"] == ["chemical_knowledge_base_tool"]
    assert enforced["visualization"]["allow_web_fallback"] is False
    assert enforced["visualization"]["web_queries"] == []


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


def test_concept_graph_semantic_generation_is_bounded_per_task_revision(monkeypatch):
    provider_events = []
    node = _concept_graph_node(monkeypatch, provider_events)
    semantic_calls = []

    class GraphTool:
        def execute(self, task, evidence, output_dir):
            semantic_calls.append((task["task_id"], task["_task_revision"]))
            return {"success": False, "error": "semantic validation failed"}

    monkeypatch.setattr(
        "src.nodes.worker.tools.concept_graph_tool.ConceptGraphTool", GraphTool
    )
    task = {
        "task_id": "T3",
        "task_name": "RAG graph",
        "_task_revision": 2,
        "_concept_graph_attempts": {},
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

    first = node._prepare_concept_graph(task, [rag_call])[2]
    second = node._prepare_concept_graph(task, [rag_call])[2]

    assert first["success"] is False
    assert second["success"] is False
    assert second["semantic_attempt_skipped"] is True
    assert semantic_calls == [("T3", 2)]


def test_rag_is_prefetched_before_autonomous_tool_selection(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(
                rag_max_queries=3, rag_adaptive_reserve=1
            )
        ),
    )

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
    assert (
        'Worker RAG prefetch 1/3: source=planner_query query="反应温度 熔融指数"'
        in capsys.readouterr().out
    )


def test_recovery_prefetch_reuses_prior_evidence_and_runs_only_incremental_queries(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(
                rag_max_queries=3, rag_adaptive_reserve=1
            )
        ),
    )

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def invoke(self, parameters):
            calls.append(parameters)
            return {"success": True, "evidence": [{"content": parameters["query"]}]}

    inherited = {
        "tool": "chemical_knowledge_base_tool",
        "parameters": {"query": "反应温度 熔融指数", "top_k": 5},
        "full_result": {"success": True, "evidence": [{"content": "existing"}]},
        "success": True,
    }
    task = {
        "task_id": "T2",
        "use_rag": True,
        "query": "反应温度 熔融指数",
        "_plan_revision": 1,
        "_task_revision": 1,
        "_inherited_rag_calls": [inherited],
        "_recovery_queries": [
            "反应压力 熔融指数",
            "停留时间 分子量分布",
            "反应温度 熔融指数",
        ],
    }

    prefetched = AutonomousToolNode._prefetch_rag(task, [KnowledgeTool()])

    assert calls == [
        {"query": "反应压力 熔融指数", "top_k": 5},
        {"query": "停留时间 分子量分布", "top_k": 5},
    ]
    assert prefetched[0]["inherited"] is True
    assert prefetched[0]["budgeted_for_attempt"] is False
    assert [call["parameters"]["query"] for call in prefetched[1:]] == [
        "反应压力 熔融指数",
        "停留时间 分子量分布",
    ]
    output = capsys.readouterr().out
    assert 'source=recovery_gap query="反应压力 熔融指数"' in output
    assert 'source=recovery_gap query="停留时间 分子量分布"' in output
    assert 'source=planner_query query="反应温度 熔融指数"' not in output
    assert "inherited_rag_calls=1" in output
    assert "inherited_evidence_items=1" in output
    assert "recovery_prefetch_queries=2" in output
    assert "adaptive_budget=1" in output


def test_recovery_prefetch_reserves_adaptive_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(
                rag_max_queries=3, rag_adaptive_reserve=1
            )
        ),
    )

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def invoke(self, parameters):
            calls.append(parameters)
            return {"success": True, "evidence": []}

    prefetched = AutonomousToolNode._prefetch_rag(
        {
            "task_id": "T2",
            "use_rag": True,
            "query": "原始查询",
            "_recovery_queries": ["缺口一", "缺口二", "缺口三"],
            "_inherited_rag_calls": [
                {
                    "tool": "chemical_knowledge_base_tool",
                    "parameters": {"query": "原始查询"},
                    "full_result": {"success": True, "evidence": []},
                    "success": True,
                }
            ],
            "_plan_revision": 1,
            "_task_revision": 1,
        },
        [KnowledgeTool()],
    )

    assert [call["query"] for call in calls] == ["缺口一", "缺口二"]
    assert len(prefetched) == 3


def test_inherited_rag_calls_do_not_consume_current_attempt_adaptive_budget(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(rag_max_queries=3)
        ),
    )

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        {
                            "id": "rag-adaptive",
                            "name": "chemical_knowledge_base_tool",
                            "args": {"query": "催化剂注入量 灰分"},
                        }
                    ],
                )
            return SimpleNamespace(content="完成正文。" * 80, tool_calls=[])

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def invoke(self, parameters):
            return {"success": True, "evidence": []}

    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=3, MAX_CHARTS_PER_TASK=0)
    node.llm_client = Model()
    initial_calls = [
        {
            "tool": "chemical_knowledge_base_tool",
            "parameters": {"query": "旧查询"},
            "success": True,
            "inherited": True,
            "budgeted_for_attempt": False,
        },
        {
            "tool": "chemical_knowledge_base_tool",
            "parameters": {"query": "恢复查询"},
            "success": True,
            "prefetched": True,
            "budgeted_for_attempt": True,
        },
    ]

    node._execute_tool_loop(
        node.llm_client,
        [],
        [KnowledgeTool()],
        {"task_name": "工艺参数"},
        initial_tool_calls=initial_calls,
    )

    assert (
        'Worker RAG adaptive query 2/3: query="催化剂注入量 灰分"'
        in capsys.readouterr().out
    )


def test_task_result_materializes_markdown_tables_and_removes_mermaid_placeholders():
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    content = """工艺参数分析正文，以下内容用于验证正式表格资产生成。

| 参数 | 质量影响 | 证据 |
| --- | --- | --- |
| 温度 | 影响熔融指数 | [E1] |

```mermaid
graph TD
A --> B
```

正文继续说明该关系必须经过正式概念图生成器验证。"""
    evidence = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺文档",
                supporting_text="温度影响熔融指数。",
                file_path="/docs/process.pdf",
            ),
        )
    )

    result = node._create_task_result(
        {
            "task_id": "T3",
            "task_name": "工艺参数",
            "generate_table": True,
            "generate_figure": True,
            "use_resources": [],
            "_plan_revision": 2,
            "_task_revision": 3,
        },
        0,
        content,
        [],
        {},
        0.1,
        True,
        False,
        evidence_bundle=evidence,
    )

    assert result["tables"][0]["type"] == "markdown"
    assert result["tables"][0]["evidence_refs"] == ["E1"]
    assert "| 温度 |" in result["text_output"]
    assert "graph TD" not in result["text_output"]
    assert result["plan_revision"] == 2
    assert result["task_revision"] == 3


def test_adaptive_rag_query_logs_its_position_in_shared_budget(monkeypatch, capsys):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(rag_max_queries=3)
        ),
    )

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        {
                            "id": "rag-2",
                            "name": "chemical_knowledge_base_tool",
                            "args": {"query": "氢气 熔融指数"},
                        }
                    ],
                )
            return SimpleNamespace(content="完成正文。" * 80, tool_calls=[])

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def invoke(self, parameters):
            return {"success": True, "evidence": []}

    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=3, MAX_CHARTS_PER_TASK=0)
    node.llm_client = Model()
    initial_call = {
        "tool": "chemical_knowledge_base_tool",
        "parameters": {"query": "聚合温度"},
        "success": True,
        "prefetched": True,
    }

    node._execute_tool_loop(
        node.llm_client,
        [],
        [KnowledgeTool()],
        {"task_name": "工艺参数"},
        initial_tool_calls=[initial_call],
    )

    assert (
        'Worker RAG adaptive query 2/3: query="氢气 熔融指数"'
        in capsys.readouterr().out
    )


def test_worker_binds_rag_claims_to_known_evidence_ids(caplog):
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    caplog.set_level("INFO", logger=worker_graph_module.__name__)

    class Model:
        def invoke(self, messages):
            pytest.fail("valid inline citations must bypass the binding LLM")

    node.llm_client = Model()
    evidence = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="聚乙烯工艺说明",
                supporting_text="聚合温度会影响熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析温度对质量的影响"},
        "聚合温度会影响熔融指数。[ E1 ]",
        evidence,
    )

    assert revised == "聚合温度会影响熔融指数。[E1]"
    assert "citation_binding_mode=deterministic_repair" in caplog.text


def test_worker_rejects_unknown_evidence_ids_from_binding_pass(capsys):
    node = AutonomousToolNode.__new__(AutonomousToolNode)

    class Model:
        def invoke(self, messages):
            return SimpleNamespace(content="聚合温度会影响熔融指数。[E404]")

    node.llm_client = Model()
    evidence = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="聚乙烯工艺说明",
                supporting_text="聚合温度会影响熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析温度对质量的影响"},
        "原始正文。",
        evidence,
    )

    assert revised == "原始正文。"
    assert "E404" in capsys.readouterr().out


def test_worker_binding_pass_reviews_partially_cited_content(caplog):
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    caplog.set_level("INFO", logger=worker_graph_module.__name__)

    class Model:
        def __init__(self):
            self.called = False

        def invoke(self, messages):
            self.called = True
            return SimpleNamespace(content="不应调用")

    model = Model()
    node.llm_client = model
    evidence = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="温度证据",
                supporting_text="温度影响熔指。",
                file_path="/srv/docs/process.docx",
            ),
            EvidenceRecord(
                evidence_id="E2",
                source_type="rag",
                title="压力证据",
                supporting_text="压力影响密度。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析温度和压力"},
        "温度影响熔指。[E1]压力影响密度。",
        evidence,
    )

    assert model.called is False
    assert revised == "温度影响熔指。[E1]压力影响密度。"
    assert "citation_binding_mode=direct" in caplog.text


def test_worker_binding_deduplicates_and_orders_adjacent_known_ids_without_llm():
    node = AutonomousToolNode.__new__(AutonomousToolNode)

    class Model:
        def invoke(self, messages):
            pytest.fail("deterministic citation canonicalization must bypass LLM")

    node.llm_client = Model()
    evidence = EvidenceBundle(
        records=tuple(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_type="rag",
                title=evidence_id,
                supporting_text="证据。",
                file_path=f"/srv/docs/{evidence_id}.docx",
            )
            for evidence_id in ("E1", "E2")
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析工艺"},
        "结论。[E2] [E1][E2]",
        evidence,
    )

    assert revised == "结论。[E1][E2]"


def test_worker_binding_uses_llm_only_when_all_valid_inline_citations_are_missing(
    caplog,
):
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    caplog.set_level("WARNING", logger=worker_graph_module.__name__)

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            assert '"evidence_id": "E1"' in messages[0].content
            return SimpleNamespace(content="聚合温度会影响熔融指数。[E1]")

    model = Model()
    node.llm_client = model
    evidence = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="聚乙烯工艺说明",
                supporting_text="聚合温度会影响熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析温度对质量的影响"},
        "聚合温度会影响熔融指数。",
        evidence,
    )

    assert model.calls == 1
    assert revised == "聚合温度会影响熔融指数。[E1]"
    assert "citation_binding_mode=llm_fallback" in caplog.text


def test_worker_fallback_removes_unknown_ids_from_original_content():
    node = AutonomousToolNode.__new__(AutonomousToolNode)

    class Model:
        def invoke(self, messages):
            return SimpleNamespace(content="模型仍然返回伪引用。[E404]")

    node.llm_client = Model()
    evidence = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="温度证据",
                supporting_text="温度影响熔指。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析温度"},
        "原文包含伪引用。[E404]",
        evidence,
    )

    assert revised == "原文包含伪引用。"
    assert "E404" not in revised


def test_worker_removes_unknown_id_from_compound_citation_without_llm():
    node = AutonomousToolNode.__new__(AutonomousToolNode)

    class Model:
        def invoke(self, messages):
            pytest.fail("a compound citation with a known ID is locally repairable")

    node.llm_client = Model()
    evidence = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺证据",
                supporting_text="温度影响熔指。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析温度"},
        "温度影响熔指。[E1, E404]",
        evidence,
    )

    assert revised == "温度影响熔指。[E1]"


def test_worker_normalizes_compound_citation_with_inherited_e_prefix():
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.llm_client = SimpleNamespace(
        invoke=lambda *_args, **_kwargs: pytest.fail(
            "compound citation normalization must bypass LLM"
        )
    )
    evidence = EvidenceBundle(
        records=tuple(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_type="rag",
                title=evidence_id,
                supporting_text="证据。",
                file_path=f"/srv/docs/{evidence_id}.docx",
            )
            for evidence_id in ("E1", "E2")
        )
    )

    revised = node._bind_claims_to_evidence(
        {"task_description": "分析工艺"},
        "结论。[E1, 2]",
        evidence,
    )

    assert revised == "结论。[E1][E2]"


def test_worker_generation_evidence_context_assigns_stable_inline_ids():
    context = AutonomousToolNode._evidence_context_for_generation(
        [
            {
                "tool": "chemical_knowledge_base_tool",
                "parameters": {"query": "聚乙烯工艺"},
                "success": True,
                "full_result": {
                    "evidence": [
                        {
                            "content": "温度影响熔融指数。",
                            "source": "/srv/docs/process.docx",
                            "title": "工艺证据",
                        }
                    ]
                },
            }
        ]
    )

    assert '"evidence_id": "E1"' in context
    assert "正文中直接使用对应的 [E编号]" in context


def test_worker_generation_evidence_context_preserves_no_evidence_constraint():
    context = AutonomousToolNode._evidence_context_for_generation(
        [
            {
                "tool": "chemical_knowledge_base_tool",
                "parameters": {"query": "缺失主题"},
                "success": False,
                "full_result": {"error": "unavailable", "has_evidence": False},
            }
        ]
    )

    assert "未提供可验证的来源证据" in context
    assert "不得使用模型常识补答" in context


def test_prefetched_evidence_is_first_iteration_inventory_and_finishes_in_one_call(
    monkeypatch,
):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(
                rag_max_queries=3, rag_adaptive_reserve=1
            )
        ),
    )

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def __init__(self):
            self.calls = []

        def invoke(self, parameters):
            self.calls.append(parameters)
            return {
                "success": True,
                "evidence": [
                    {
                        "title": "聚乙烯生产工艺与质量控制概述",
                        "source": "/srv/private/process.docx",
                        "section_path": "§5 关键工艺参数及其影响",
                        "content": "参数趋势、实验室分析与批次追踪可用于异常诊断。" * 30,
                    }
                ],
            }

    class Model:
        def __init__(self):
            self.calls = []

        def invoke(self, messages, **_kwargs):
            self.calls.append(messages)
            return SimpleNamespace(content="基于已有证据完成报告。[E1]" + "正文" * 100, tool_calls=[])

    query = "聚乙烯 质量异常 排查建议"
    task = {"task_id": "T1", "use_rag": True, "query": query}
    tool = KnowledgeTool()
    prefetched = AutonomousToolNode._prefetch_rag(task, [tool])
    context = AutonomousToolNode._evidence_context_for_generation(prefetched)

    assert '"prefetched_queries": ["聚乙烯 质量异常 排查建议"]' in context
    assert f'"query_fingerprints": ["{query_fingerprint(query)}"]' in context
    assert '"prefetch_queries_used": 1' in context
    assert '"adaptive_queries_remaining": 2' in context
    assert '"supporting_text_excerpt"' in context
    assert '"supporting_text"' not in context
    assert "/srv/private/process.docx" not in context
    assert len(context) < 1400
    inventory = json.loads(context[context.index("{") :])
    assert len(inventory["evidence"][0]["supporting_text_excerpt"]) == 240
    assert "预检索已完成" in context
    assert "具体的新证据缺口" in context

    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=3, MAX_CHARTS_PER_TASK=0)
    model = Model()
    node.llm_client = model
    _content, calls, _stats = node._execute_tool_loop(
        model,
        [SimpleNamespace(content=context)],
        [tool],
        task,
        initial_tool_calls=prefetched,
    )

    assert len(model.calls) == 1
    assert len(tool.calls) == 1
    assert len(calls) == 1


def test_prefetch_inventory_uses_safe_title_without_mutating_provenance(monkeypatch):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(rag_max_queries=3)
        ),
    )
    internal_path = (
        "/srv/cache/users/user-17/conversations/conversation-29/jobs/job-41/"
        "process-guide.docx"
    )
    tool_calls = [
        {
            "tool": "chemical_knowledge_base_tool",
            "parameters": {"query": "聚乙烯工艺"},
            "success": True,
            "prefetched": True,
            "full_result": {
                "evidence": [
                    {
                        "title": "",
                        "source": internal_path,
                        "section_path": "§5",
                        "content": "温度影响熔融指数。",
                    }
                ]
            },
        }
    ]
    original = deepcopy(tool_calls)

    context = AutonomousToolNode._evidence_context_for_generation(tool_calls)

    assert internal_path not in context
    assert "/srv/cache" not in context
    assert "users/user-17" not in context
    assert "conversations/conversation-29" not in context
    assert "jobs/job-41" not in context
    assert '"title": "process-guide.docx"' in context
    assert tool_calls == original


def test_same_normalized_prefetch_query_is_rejected_without_adaptive_retrieval(
    monkeypatch,
):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(
                rag_max_queries=3, rag_adaptive_reserve=1
            )
        ),
    )

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def __init__(self):
            self.calls = []

        def invoke(self, parameters):
            self.calls.append(parameters)
            return {"success": True, "evidence": []}

    responses = [
        SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "same-q1",
                    "name": "chemical_knowledge_base_tool",
                    "args": {"query": "聚乙烯 质量异常"},
                }
            ],
        ),
        SimpleNamespace(content="基于已有证据完成。" + "正文" * 100, tool_calls=[]),
    ]

    seen_messages = []

    class Model:
        def invoke(self, messages, **_kwargs):
            seen_messages.append(messages)
            return responses.pop(0)

    task = {"task_id": "T1", "use_rag": True, "query": "  聚乙烯   质量异常 "}
    tool = KnowledgeTool()
    prefetched = AutonomousToolNode._prefetch_rag(task, [tool])
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=3, MAX_CHARTS_PER_TASK=0)
    node.llm_client = Model()

    _content, calls, _stats = node._execute_tool_loop(
        node.llm_client, [], [tool], task, initial_tool_calls=prefetched
    )

    assert len(tool.calls) == 1
    assert len(calls) == 1
    assert any(
        "已跳过重复的知识库查询" in message.content
        for message in seen_messages[1]
        if hasattr(message, "content")
    )


def test_textually_distinct_q2_runs_one_adaptive_retrieval(monkeypatch):
    monkeypatch.setattr(
        worker_graph_module,
        "get_app_config",
        lambda: SimpleNamespace(
            concept_graph_settings=SimpleNamespace(
                rag_max_queries=3, rag_adaptive_reserve=1
            )
        ),
    )

    class KnowledgeTool:
        name = "chemical_knowledge_base_tool"

        def __init__(self):
            self.calls = []

        def invoke(self, parameters):
            self.calls.append(parameters)
            return {"success": True, "evidence": []}

    q2 = "催化剂注入量 灰分"
    responses = [
        SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "new-q2",
                    "name": "chemical_knowledge_base_tool",
                    "args": {"query": q2},
                }
            ],
        ),
        SimpleNamespace(content="补充证据后完成。" + "正文" * 100, tool_calls=[]),
    ]

    class Model:
        def invoke(self, _messages, **_kwargs):
            return responses.pop(0)

    task = {"task_id": "T1", "use_rag": True, "query": "聚乙烯 质量异常"}
    tool = KnowledgeTool()
    prefetched = AutonomousToolNode._prefetch_rag(task, [tool])
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=3, MAX_CHARTS_PER_TASK=0)
    node.llm_client = Model()

    _content, calls, _stats = node._execute_tool_loop(
        node.llm_client, [], [tool], task, initial_tool_calls=prefetched
    )

    assert [call["query"] for call in tool.calls] == [task["query"], q2]
    assert calls[-1]["parameters"]["query"] == q2
    assert calls[-1]["iteration"] == 1


def test_worker_clean_loop_uses_prefetched_stable_id_without_binding_call():
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=2)
    prefetched_calls = [
        {
            "tool": "chemical_knowledge_base_tool",
            "parameters": {"query": "聚乙烯工艺"},
            "success": True,
            "full_result": {
                "evidence": [
                    {
                        "content": "温度影响熔融指数。",
                        "source": "/srv/docs/process.docx",
                        "title": "工艺证据",
                    }
                ]
            },
            "prefetched": True,
        }
    ]
    evidence_context = node._evidence_context_for_generation(prefetched_calls)

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages, **_kwargs):
            self.calls += 1
            assert '"evidence_id": "E1"' in messages[0].content
            return SimpleNamespace(
                content="温度影响熔融指数。[E1]" + "正文" * 100,
                tool_calls=[],
            )

    model = Model()
    node.llm_client = model
    content, calls, _stats = node._execute_tool_loop(
        model,
        [SimpleNamespace(content=evidence_context)],
        [],
        {"task_id": "T1", "task_name": "工艺概述"},
        initial_tool_calls=prefetched_calls,
    )
    evidence = normalize_rag_tool_calls(calls)
    revised = node._bind_claims_to_evidence(
        {"task_id": "T1", "task_description": "工艺概述"},
        content,
        evidence,
    )

    assert model.calls == 1
    assert revised == content


def test_worker_task_prompt_preserves_all_covered_section_headings(monkeypatch):
    node = AutonomousToolNode.__new__(AutonomousToolNode)
    node.config = SimpleNamespace(MAX_CHARTS_PER_TASK=6)

    class Prompt:
        def format_messages(self, **values):
            return [SimpleNamespace(content=values["covered_sections_section"])]

    monkeypatch.setattr(
        worker_graph_module.ChatPromptTemplate,
        "from_template",
        lambda _template: Prompt(),
    )

    prompt = node._build_task_prompt(
        {
            "task_description": "分析温度与压力对产品质量的影响。",
            "query": "聚乙烯 温度 压力 质量",
            "use_resources": [],
            "covers_sections": ["3.1 聚合反应温度", "3.2 聚合压力"],
        }
    )

    assert "3.1 聚合反应温度" in prompt
    assert "3.2 聚合压力" in prompt
    assert "H3 (###)" in prompt
    assert "必须按上述顺序保留" in prompt

    template = (
        Path(worker_graph_module.__file__).parents[3]
        / "prompts"
        / "worker_task_template.md"
    ).read_text(encoding="utf-8")
    assert "任务主标题使用二级标题" in template
    assert "覆盖章节标题使用“章节覆盖要求”中指定的 Markdown 层级" in template

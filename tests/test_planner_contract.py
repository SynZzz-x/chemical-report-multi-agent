import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from src.nodes import intake as intake_module
from src.nodes import planner as planner_module
from src.nodes import verifier_manual as verifier_manual_module


def _task(**overrides):
    task = {
        "task_id": "T1",
        "task_name": "工艺概述",
        "task_description": "基于知识库梳理聚乙烯生产工艺并注明来源。",
        "task_type": "analysis",
        "use_rag": True,
        "use_web": False,
        "query": "聚乙烯 生产工艺",
        "use_resources": [],
        "generate_figure": False,
        "generate_table": False,
        "visualization": None,
    }
    task.update(overrides)
    return task


class _SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=next(self.responses))


def _patch_model(monkeypatch, responses):
    model = _SequenceModel(responses)
    modes = []

    def fake_get_llm(config, *, json_mode):
        modes.append(json_mode)
        return model

    monkeypatch.setattr(planner_module, "get_llm", fake_get_llm)
    return model, modes


def _intake_summary(**overrides):
    payload = {
        "from": "Intake",
        "to": "Planner",
        "type": "INTAKE_SUMMARY",
        "title": "聚乙烯质量分析",
        "user_intent": "分析聚乙烯工艺参数及质量异常",
        "task_type": "工程报告",
        "doc_length": "5000字",
        "constraints": ["严格基于知识库"],
        "sections": ["工艺概述"],
        "core_content": ["聚乙烯生产", "质量异常排查"],
        "style": "formal",
        "output_format": "PDF",
        "web_authorized": False,
        "resources": [],
    }
    payload.update(overrides)
    return payload


def test_initial_planner_accepts_json_object_envelope_and_enables_json_mode(monkeypatch):
    model, modes = _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [_task()]}, ensure_ascii=False)],
    )

    tasks = planner_module._build_tasks_with_llm(_intake_summary(), {})

    assert tasks == [_task()]
    assert len(model.calls) == 1
    assert modes == [True]


def test_initial_planner_receives_read_only_knowledge_catalog_separately(
    monkeypatch,
):
    catalog = [
        {
            "resource_id": "doc-1",
            "file_name": "聚乙烯生产技术手册.pdf",
            "file_type": "pdf",
            "path": "/private/knowledge/聚乙烯生产技术手册.pdf",
            "sha256": "hash-1",
            "indexed": True,
            "summary": "聚乙烯生产路线及质量控制资料。",
            "topics": ["聚乙烯", "质量控制"],
            "content_type": "technical_document",
            "has_structured_data": False,
            "supports": ["rag", "citation"],
            "catalog_version": "1",
        }
    ]
    captured = {}
    monkeypatch.setattr(planner_module, "load_active_catalog", lambda: catalog)

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return [_task()]

    monkeypatch.setattr(planner_module, "_invoke_plan_generation", fake_invoke)

    planner_module._build_tasks_with_llm(_intake_summary(resources=[]), {})

    visible = captured["prompt_values"]["knowledge_catalog"]
    assert visible == [
        {
            "resource_id": "doc-1",
            "file_name": "聚乙烯生产技术手册.pdf",
            "file_type": "pdf",
            "indexed": True,
            "summary": "聚乙烯生产路线及质量控制资料。",
            "topics": ["聚乙烯", "质量控制"],
            "content_type": "technical_document",
            "has_structured_data": False,
            "supports": ["rag", "citation"],
            "catalog_version": "1",
        }
    ]
    assert "/private/knowledge" not in json.dumps(visible, ensure_ascii=False)
    assert captured["resources"] == []


def test_knowledge_catalog_entry_cannot_be_used_as_job_resource(monkeypatch):
    catalog = [
        {
            "resource_id": "doc-1",
            "file_name": "聚乙烯生产技术手册.pdf",
            "file_type": "pdf",
            "indexed": True,
            "summary": "聚乙烯生产资料。",
            "topics": ["聚乙烯"],
            "content_type": "technical_document",
            "has_structured_data": False,
            "supports": ["rag", "citation"],
            "catalog_version": "1",
        }
    ]
    task = _task(use_resources=["聚乙烯生产技术手册.pdf"])
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    monkeypatch.setattr(planner_module, "load_active_catalog", lambda: catalog)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="unknown resource"):
        planner_module._build_tasks_with_llm(_intake_summary(resources=[]), {})


def test_importing_planner_does_not_initialize_full_rag_runtime():
    script = textwrap.dedent(
        """
        import sys
        import src.nodes.planner
        forbidden = {
            'src.rag.service',
            'src.rag.tokenizer',
            'src.rag.vector_store',
        }
        loaded = sorted(forbidden.intersection(sys.modules))
        if loaded:
            raise SystemExit('unexpected RAG imports: ' + ', '.join(loaded))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_initial_planner_repairs_one_invalid_response(monkeypatch):
    model, _ = _patch_model(
        monkeypatch,
        ["not-json", json.dumps({"tasks": [_task()]}, ensure_ascii=False)],
    )

    tasks = planner_module._build_tasks_with_llm(_intake_summary(), {})

    assert tasks == [_task()]
    assert len(model.calls) == 2
    repair_text = str(model.calls[1][0][-1].content)
    assert "validation" in repair_text.lower() or "校验" in repair_text


def test_planner_validation_log_includes_error_detail_and_long_response_snippet(
    monkeypatch,
    caplog,
):
    invalid = "x" * 700 + "TAIL_MARKER"
    _patch_model(monkeypatch, [invalid, invalid])

    with caplog.at_level("WARNING", logger="src.nodes.planner"):
        with pytest.raises(ValueError):
            planner_module._build_tasks_with_llm(_intake_summary(), {})

    log_text = caplog.text
    assert "JSONDecodeError" in log_text
    assert "Expecting value" in log_text
    assert "TAIL_MARKER" in log_text


def test_initial_planner_two_invalid_responses_fail_without_placeholder_tasks(monkeypatch):
    _patch_model(monkeypatch, ["not-json", json.dumps({"tasks": []})])

    with pytest.raises(ValueError, match="initial plan generation failed"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


@pytest.mark.parametrize(
    "payload",
    [
        [_task()],
        {"foo": [_task()]},
        {"tasks": [_task()], "extra": True},
        {"tasks": []},
        {"tasks": [{key: value for key, value in _task().items() if key != "task_name"}]},
        {"tasks": [_task(use_rag="yes")]},
        {"tasks": [_task(task_id="   ")]},
    ],
)
def test_initial_planner_rejects_invalid_plan_contract(monkeypatch, payload):
    encoded = json.dumps(payload, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


def test_initial_planner_rejects_generated_task_with_extra_field(monkeypatch):
    task = _task(tool_requirements=[])
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="exactly"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


def test_initial_planner_requires_sequential_task_ids(monkeypatch):
    tasks = [
        _task(task_id="T1", task_name="工艺概述"),
        _task(task_id="T3", task_name="质量异常"),
    ]
    encoded = json.dumps({"tasks": tasks}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="T1.*T2"):
        planner_module._build_tasks_with_llm(
            _intake_summary(sections=["工艺概述", "质量异常"]),
            {},
        )


def test_initial_planner_rejects_task_id_with_surrounding_whitespace(monkeypatch):
    task = _task(task_id=" T1 ")
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="sequential"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


@pytest.mark.parametrize("query", ["不应执行的检索词", "   "])
def test_initial_planner_rejects_query_when_rag_is_disabled(monkeypatch, query):
    task = _task(
        task_description="梳理聚乙烯生产工艺。",
        use_rag=False,
        query=query,
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="use_rag=false"):
        planner_module._build_tasks_with_llm(
            _intake_summary(constraints=[]),
            {},
        )


def test_initial_planner_allows_knowledge_base_context_without_rag(monkeypatch):
    task = _task(
        task_name="引言",
        task_description="介绍本报告基于企业知识库开展分析，并说明背景和结构。",
        use_rag=False,
        query="",
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    assert planner_module._build_tasks_with_llm(
        _intake_summary(constraints=[], sections=["引言"]),
        {},
    ) == [task]


def test_initial_planner_does_not_reinfer_rag_from_task_wording(monkeypatch):
    task = _task(
        task_description="分析工艺参数并提供出处和引用依据。",
        use_rag=False,
        query="",
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    assert planner_module._build_tasks_with_llm(
        _intake_summary(constraints=[]),
        {},
    ) == [task]


@pytest.mark.parametrize(
    "task_description",
    ["汇总已有结论，不新增引用。", "引用格式采用 GB/T 7714。"],
)
def test_initial_planner_does_not_treat_non_retrieval_citation_text_as_rag(
    monkeypatch,
    task_description,
):
    task = _task(
        task_description=task_description,
        use_rag=False,
        query="",
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    assert planner_module._build_tasks_with_llm(
        _intake_summary(constraints=[]),
        {},
    ) == [task]


def test_initial_planner_rejects_statistical_work_without_data(monkeypatch):
    task = _task(
        task_description="计算 Pearson 相关系数并生成回归模型与热力图。",
        use_rag=False,
        query="",
        generate_figure=True,
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="data resource"):
        planner_module._build_tasks_with_llm(_intake_summary(constraints=[]), {})


@pytest.mark.parametrize(
    "task_description",
    [
        "计算关键参数的定量操作窗口。",
        "计算装置转化率。",
        "定量评估装置能耗。",
    ],
)
def test_initial_planner_rejects_quantitative_work_without_csv(
    monkeypatch,
    task_description,
):
    task = _task(
        task_description=task_description,
        use_rag=False,
        query="",
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="data resource"):
        planner_module._build_tasks_with_llm(
            _intake_summary(constraints=[]),
            {},
        )


@pytest.mark.parametrize(
    "task_description",
    [
        "基于知识库分析反应温度对转化率的影响机理。",
        "基于知识库分析不同工艺路线的能耗影响因素。",
    ],
)
def test_initial_planner_accepts_qualitative_metric_analysis_without_csv(
    monkeypatch,
    task_description,
):
    task = _task(task_description=task_description)
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    assert planner_module._build_tasks_with_llm(_intake_summary(), {}) == [task]


def test_initial_planner_rejects_unknown_resource_even_when_name_looks_like_csv(
    monkeypatch,
):
    task = _task(use_resources=["invented.csv"])
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="unknown resource"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


def test_initial_planner_accepts_assigned_data_resource_by_declared_type(monkeypatch):
    task = _task(
        task_description="计算 Pearson 相关系数并生成回归模型。",
        use_rag=False,
        query="",
        use_resources=["生产数据"],
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    tasks = planner_module._build_tasks_with_llm(
        _intake_summary(
            constraints=[],
            resources=[
                {
                    "name": "生产数据",
                    "path": "/uploads/resource-123.csv",
                    "type": "csv",
                }
            ],
        ),
        {},
    )

    assert tasks[0]["use_resources"] == ["/uploads/resource-123.csv"]


def test_initial_planner_rejects_declared_data_resource_without_usable_path(
    monkeypatch,
):
    task = _task(
        task_description="根据生产数据生成趋势图。",
        use_rag=False,
        query="",
        use_resources=["生产数据"],
        generate_figure=True,
        visualization=None,
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="data resource"):
        planner_module._build_tasks_with_llm(
            _intake_summary(
                constraints=[],
                resources=[{"name": "生产数据", "path": None, "type": "csv"}],
            ),
            {},
        )


def test_initial_planner_rejects_non_csv_resource_for_ordinary_figure(monkeypatch):
    task = _task(
        task_description="根据生产数据生成趋势图。",
        use_rag=False,
        query="",
        use_resources=["生产数据"],
        generate_figure=True,
        visualization=None,
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="CSV"):
        planner_module._build_tasks_with_llm(
            _intake_summary(
                constraints=[],
                resources=[
                    {
                        "name": "生产数据",
                        "path": "/uploads/data.parquet",
                        "type": "parquet",
                    }
                ],
            ),
            {},
        )


def test_initial_planner_accepts_ordinary_data_figure_without_visualization(monkeypatch):
    task = _task(
        task_description="根据生产数据生成趋势图。",
        use_rag=False,
        query="",
        use_resources=["生产数据"],
        generate_figure=True,
        visualization=None,
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    tasks = planner_module._build_tasks_with_llm(
        _intake_summary(
            constraints=[],
            resources=[
                {"name": "生产数据", "path": "/uploads/data.csv", "type": "csv"}
            ],
        ),
        {},
    )

    assert tasks[0]["generate_figure"] is True
    assert tasks[0]["visualization"] is None


def test_initial_planner_rejects_ordinary_figure_without_data(monkeypatch):
    task = _task(
        task_description="生成产品指标趋势图。",
        use_rag=False,
        query="",
        generate_figure=True,
        visualization=None,
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="data resource"):
        planner_module._build_tasks_with_llm(
            _intake_summary(constraints=[]),
            {},
        )


def test_initial_planner_rejects_ambiguous_resource_alias(monkeypatch):
    task = _task(use_resources=["production.csv"])
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])
    resources = [
        {"name": "production.csv", "path": "/plant-a/production.csv", "type": "csv"},
        {"name": "production.csv", "path": "/plant-b/production.csv", "type": "csv"},
    ]

    with pytest.raises(ValueError, match="ambiguous resource"):
        planner_module._build_tasks_with_llm(
            _intake_summary(resources=resources),
            {},
        )


def test_initial_planner_accepts_unique_full_resource_path(monkeypatch):
    task = _task(use_resources=["/plant-a/production.csv"])
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )
    resources = [
        {"name": "production.csv", "path": "/plant-a/production.csv", "type": "csv"},
        {"name": "production.csv", "path": "/plant-b/production.csv", "type": "csv"},
    ]

    tasks = planner_module._build_tasks_with_llm(
        _intake_summary(resources=resources),
        {},
    )

    assert tasks[0]["use_resources"] == ["/plant-a/production.csv"]


def test_initial_planner_rejects_web_without_explicit_authorization(monkeypatch):
    task = _task(
        task_description="检索公开网络资料并补充行业信息。",
        use_web=True,
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="web authorization"):
        planner_module._build_tasks_with_llm(
            _intake_summary(web_authorized=False),
            {},
        )


def test_initial_planner_rejects_visualization_web_queries_without_authorization(
    monkeypatch,
):
    task = _task(
        generate_figure=True,
        visualization={
            "kind": "causal",
            "title": "关系图",
            "required_concepts": ["温度"],
            "web_queries": ["公开资料"],
            "allow_web_fallback": False,
        }
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="web authorization"):
        planner_module._build_tasks_with_llm(
            _intake_summary(web_authorized=False),
            {},
        )


@pytest.mark.parametrize(
    "visualization",
    [
        {},
        {
            "kind": "pie",
            "title": "关系图",
            "required_concepts": ["温度"],
            "web_queries": [],
            "allow_web_fallback": False,
        },
        {
            "kind": "causal",
            "title": "关系图",
            "required_concepts": ["温度"],
            "web_queries": [],
            "allow_web_fallback": False,
            "extra": True,
        },
    ],
)
def test_initial_planner_rejects_invalid_concept_visualization(
    monkeypatch,
    visualization,
):
    task = _task(generate_figure=True, visualization=visualization)
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="visualization"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


def test_initial_planner_rejects_visualization_when_figure_is_disabled(monkeypatch):
    task = _task(
        generate_figure=False,
        visualization={
            "kind": "causal",
            "title": "关系图",
            "required_concepts": ["温度"],
            "web_queries": [],
            "allow_web_fallback": False,
        },
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="generate_figure"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


def test_initial_planner_accepts_supported_concept_visualization(monkeypatch):
    task = _task(
        generate_figure=True,
        visualization={
            "kind": "causal",
            "title": "概念关系图",
            "required_concepts": ["温度", "产品质量"],
            "web_queries": [],
            "allow_web_fallback": False,
        },
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    assert planner_module._build_tasks_with_llm(_intake_summary(), {}) == [task]


def test_initial_planner_rejects_more_than_six_required_concepts(monkeypatch):
    task = _task(
        generate_figure=True,
        visualization={
            "kind": "causal",
            "title": "概念关系图",
            "required_concepts": [f"概念{i}" for i in range(1, 8)],
            "web_queries": [],
            "allow_web_fallback": False,
        },
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="at most 6"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


@pytest.mark.parametrize(
    "compound_concept",
    ["氢气/共聚单体比例", "温度与压力", "催化剂、添加剂"],
)
def test_initial_planner_rejects_non_atomic_required_concepts(
    monkeypatch,
    compound_concept,
):
    task = _task(
        generate_figure=True,
        visualization={
            "kind": "causal",
            "title": "概念关系图",
            "required_concepts": [compound_concept],
            "web_queries": [],
            "allow_web_fallback": False,
        },
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="atomic"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


@pytest.mark.parametrize(
    "required_concepts",
    [
        [f"概念{i}" for i in range(1, 8)],
        ["温度与压力"],
    ],
)
def test_replacement_planner_enforces_concept_scope(required_concepts):
    candidate = {
        "task_name": "工艺参数",
        "task_description": "分析关键工艺参数。",
        "visualization": {
            "kind": "causal",
            "title": "概念关系图",
            "required_concepts": required_concepts,
            "web_queries": [],
            "allow_web_fallback": False,
        },
    }

    with pytest.raises(ValueError, match="at most 6|atomic"):
        planner_module._validate_replacement_task_schema([candidate])


@pytest.mark.parametrize(
    "visualization",
    [
        {
            "kind": "flowchart",
            "title": "流程图",
            "required_concepts": ["温度"],
            "web_queries": [],
            "allow_web_fallback": False,
        },
        {
            "kind": "causal",
            "title": "缺失概念的关系图",
            "web_queries": [],
            "allow_web_fallback": False,
        },
    ],
)
def test_replacement_planner_requires_complete_causal_visualization(visualization):
    candidate = {
        "task_name": "工艺参数",
        "task_description": "分析关键工艺参数。",
        "visualization": visualization,
    }

    with pytest.raises(ValueError, match="visualization|causal|required_concepts"):
        planner_module._validate_replacement_task_schema([candidate])


def test_initial_planner_rejects_concept_graph_without_evidence_channel(monkeypatch):
    task = _task(
        task_description="绘制温度与产品质量关系图。",
        use_rag=False,
        query="",
        generate_figure=True,
        visualization={
            "kind": "causal",
            "title": "概念关系图",
            "required_concepts": ["温度", "产品质量"],
            "web_queries": [],
            "allow_web_fallback": False,
        },
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="evidence"):
        planner_module._build_tasks_with_llm(
            _intake_summary(constraints=[]),
            {},
        )


@pytest.mark.parametrize("kind", ["flowchart", "fault_tree"])
def test_initial_planner_rejects_unimplemented_concept_visualizations(
    monkeypatch,
    kind,
):
    task = _task(
        generate_figure=True,
        visualization={
            "kind": kind,
            "title": "概念关系图",
            "required_concepts": ["温度", "产品质量"],
            "web_queries": [],
            "allow_web_fallback": False,
        },
    )
    encoded = json.dumps({"tasks": [task]}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="visualization.kind"):
        planner_module._build_tasks_with_llm(_intake_summary(), {})


def test_initial_planner_accepts_explicitly_authorized_web_task(monkeypatch):
    task = _task(
        task_description="检索公开网络资料并补充行业信息。",
        use_web=True,
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    tasks = planner_module._build_tasks_with_llm(
        _intake_summary(web_authorized=True),
        {},
    )

    assert tasks == [task]


def test_global_knowledge_constraint_requires_some_rag_not_every_task(monkeypatch):
    tasks = [
        _task(task_id="T1", task_name="工艺概述"),
        _task(
            task_id="T2",
            task_name="结论",
            task_description="归纳前述任务已形成的结论。",
            use_rag=False,
            query="",
        ),
    ]
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": tasks}, ensure_ascii=False)],
    )

    planned = planner_module._build_tasks_with_llm(
        _intake_summary(
            constraints=["所有内容严格基于知识库"],
            sections=["工艺概述", "结论"],
        ),
        {},
    )

    assert planned == tasks


def test_validator_does_not_reinfer_rag_from_global_wording(monkeypatch):
    task = _task(
        task_description="梳理聚乙烯生产工艺。",
        use_rag=False,
        query="",
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    assert planner_module._build_tasks_with_llm(
        _intake_summary(constraints=["所有内容严格基于知识库"]),
        {},
    ) == [task]


@pytest.mark.parametrize(
    "constraint",
    [
        "不使用知识库，只分析上传的 CSV",
        "不得使用知识库，只分析上传的 CSV",
        "不可使用知识库，只分析上传的 CSV",
    ],
)
def test_negative_global_knowledge_constraint_does_not_require_rag(
    monkeypatch,
    constraint,
):
    task = _task(
        task_description="只分析上传的生产数据。",
        use_rag=False,
        query="",
    )
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": [task]}, ensure_ascii=False)],
    )

    assert planner_module._build_tasks_with_llm(
        _intake_summary(
            constraints=[constraint],
        ),
        {},
    ) == [task]


def test_initial_planner_follows_intake_sections_one_to_one(monkeypatch):
    tasks = [
        _task(task_id="T1", task_name="引言"),
        _task(task_id="T2", task_name="结论"),
    ]
    encoded = json.dumps({"tasks": tasks}, ensure_ascii=False)
    _patch_model(monkeypatch, [encoded, encoded])

    with pytest.raises(ValueError, match="sections"):
        planner_module._build_tasks_with_llm(
            _intake_summary(sections=["引言", "工艺参数", "结论"]),
            {},
        )


@pytest.mark.parametrize(
    "abstract_section",
    [
        "摘要",
        "Abstract",
        "一、摘要",
        "第一章 摘要",
        "第1章 摘要",
        "## 摘要",
        "（一）摘要",
        "(一) 摘要",
        "第一部分 摘要",
        "第一篇 摘要",
    ],
)
def test_initial_planner_ignores_abstract_when_matching_sections(
    monkeypatch,
    abstract_section,
):
    tasks = [
        _task(task_id="T1", task_name="引言"),
        _task(task_id="T2", task_name="结论"),
    ]
    _patch_model(
        monkeypatch,
        [json.dumps({"tasks": tasks}, ensure_ascii=False)],
    )

    planned = planner_module._build_tasks_with_llm(
        _intake_summary(sections=[abstract_section, "引言", "结论"]),
        {},
    )

    assert planned == tasks


def test_initial_planner_passes_complete_intake_context_to_prompt(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        planner_module,
        "_invoke_plan_generation",
        lambda **kwargs: (captured.update(kwargs), [_task()])[1],
    )

    planner_module._build_tasks_with_llm(_intake_summary(), {})

    assert captured["prompt_values"]["core_content"] == [
        "聚乙烯生产",
        "质量异常排查",
    ]
    assert captured["prompt_values"]["style"] == "formal"
    assert captured["prompt_values"]["output_format"] == "PDF"


def test_replan_and_refine_accept_the_same_json_object_contract(monkeypatch):
    _patch_model(
        monkeypatch,
        [
            json.dumps({"tasks": [_task()]}, ensure_ascii=False),
            json.dumps({"tasks": [_task()]}, ensure_ascii=False),
        ],
    )
    intake = AIMessage(content=json.dumps(_intake_summary(), ensure_ascii=False))
    state = {
        "messages": [intake],
        "feedback": {
            "status": "BLOCKED",
            "issues": [{"description": "调整结构", "suggestion": "重新拆分"}],
        },
    }

    replanned = planner_module._build_tasks_from_replan_feedback(state, {}, [])
    refined = planner_module._refine_tasks(
        state,
        replanned,
        "调整章节",
        [],
        _intake_summary(),
        {},
    )

    assert replanned == [_task()]
    assert refined == [_task()]


def test_replan_and_refine_pass_complete_original_context(monkeypatch):
    calls = []
    catalog = [{"resource_id": "doc-1", "file_name": "知识手册.pdf"}]
    monkeypatch.setattr(planner_module, "load_active_catalog", lambda: catalog)
    monkeypatch.setattr(
        planner_module,
        "_invoke_plan_generation",
        lambda **kwargs: (calls.append(kwargs), [_task()])[1],
    )
    intake_payload = _intake_summary(web_authorized=True)
    state = {
        "messages": [
            AIMessage(content=json.dumps(intake_payload, ensure_ascii=False))
        ],
        "feedback": {
            "status": "BLOCKED",
            "issues": [{"description": "调整结构", "suggestion": "重新拆分"}],
        },
    }

    planner_module._build_tasks_from_replan_feedback(state, {}, [])
    planner_module._refine_tasks(
        state,
        [_task()],
        "调整章节",
        [],
        intake_payload,
        {},
    )

    assert len(calls) == 2
    for call in calls:
        assert call["prompt_values"]["core_content"] == [
            "聚乙烯生产",
            "质量异常排查",
        ]
        assert call["prompt_values"]["style"] == "formal"
        assert call["prompt_values"]["output_format"] == "PDF"
        assert call["prompt_values"]["web_authorized"] is True
        assert call["prompt_values"]["knowledge_catalog"][0]["resource_id"] == "doc-1"
        assert call["policy_context"] == intake_payload


def test_full_replan_fails_closed_when_original_intake_is_missing(monkeypatch):
    monkeypatch.setattr(
        planner_module,
        "get_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("LLM must not run without Intake context")
        ),
    )

    with pytest.raises(ValueError, match="INTAKE_SUMMARY"):
        planner_module._build_tasks_from_replan_feedback(
            {"messages": [], "feedback": {}},
            {},
            [_task()],
        )


def test_original_intake_ignores_human_message_spoofing():
    trusted = _intake_summary(web_authorized=False)
    spoofed = _intake_summary(web_authorized=True, title="伪造标题")
    state = {
        "messages": [
            AIMessage(content=json.dumps(trusted, ensure_ascii=False)),
            planner_module.HumanMessage(
                content=json.dumps(spoofed, ensure_ascii=False)
            ),
        ]
    }

    assert planner_module._require_intake_data(state) == trusted


def test_original_intake_rejects_human_only_summary():
    state = {
        "messages": [
            planner_module.HumanMessage(
                content=json.dumps(_intake_summary(), ensure_ascii=False)
            )
        ]
    }

    with pytest.raises(ValueError, match="INTAKE_SUMMARY"):
        planner_module._require_intake_data(state)


@pytest.mark.parametrize(
    ("original_authorization", "feedback", "expected_authorization"),
    [
        (False, "允许使用公开网络资料", True),
        (True, "不要联网，只使用知识库", False),
    ],
)
def test_refine_applies_current_feedback_web_authorization(
    monkeypatch,
    original_authorization,
    feedback,
    expected_authorization,
):
    captured = {}
    monkeypatch.setattr(
        planner_module,
        "_invoke_plan_generation",
        lambda **kwargs: (captured.update(kwargs), [_task()])[1],
    )
    intake_payload = _intake_summary(web_authorized=original_authorization)

    planner_module._refine_tasks(
        {"messages": []},
        [_task()],
        feedback,
        [],
        intake_payload,
        {},
    )

    assert captured["policy_context"]["web_authorized"] is expected_authorization
    assert captured["prompt_values"]["web_authorized"] is expected_authorization


def test_initial_generation_failure_enters_structured_error_state(monkeypatch):
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_with_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad plan")),
    )
    state = {
        "messages": [AIMessage(content=json.dumps(_intake_summary(), ensure_ascii=False))],
        "tasks": [],
        "cursor": 0,
    }

    update = planner_module.planner(state, {})

    assert update["planner_action"] == "INITIAL_PLAN_ERROR"
    assert update["tasks"] == []
    assert "bad plan" in update["guidance"]["error"]
    assert "messages" not in update


def test_initial_plan_retry_rebuilds_from_original_intake_not_latest_proceed(
    monkeypatch,
):
    intake = _intake_summary()
    captured = []
    monkeypatch.setattr(
        planner_module,
        "_build_tasks_with_llm",
        lambda payload, _config: (captured.append(payload), [_task()])[1],
    )
    monkeypatch.setattr(
        planner_module,
        "_generate_plan_guidance",
        lambda *_args, **_kwargs: {
            "natural_language_guidance": "请确认",
            "resource_mapping": {},
        },
    )
    state = {
        "messages": [
            AIMessage(content=json.dumps(intake, ensure_ascii=False)),
            AIMessage(
                content=json.dumps(
                    {"from": "Verifier", "to": "Planner", "type": "PROCEED"}
                )
            ),
        ],
        "planner_action": "INITIAL_PLAN_RETRY",
        "tasks": [],
    }

    update = planner_module.planner(state, {})

    assert captured == [intake]
    assert update["planner_action"] == "INTAKE_SUMMARY"
    assert update["tasks"] == [_task()]


def test_initial_plan_retry_fails_closed_when_original_intake_is_missing():
    state = {
        "messages": [
            AIMessage(
                content=json.dumps(
                    {"from": "Verifier", "to": "Planner", "type": "PROCEED"}
                )
            )
        ],
        "planner_action": "INITIAL_PLAN_RETRY",
        "tasks": [],
    }

    update = planner_module.planner(state, {})

    assert update["planner_action"] == "INITIAL_PLAN_ERROR"
    assert update["tasks"] == []
    assert "INTAKE_SUMMARY" in update["guidance"]["error"]


@pytest.mark.parametrize(
    ("resume", "expected_action"),
    [
        ({"action": "RETRY_INITIAL_PLAN"}, "INITIAL_PLAN_RETRY"),
        ({"action": "CANCEL"}, "INITIAL_PLAN_CANCELLED"),
    ],
)
def test_initial_plan_error_offers_retry_or_cancel(monkeypatch, resume, expected_action):
    payloads = []
    monkeypatch.setattr(
        planner_module,
        "interrupt",
        lambda payload: (payloads.append(payload), resume)[1],
    )

    update = planner_module.planner_confirm(
        {
            "planner_action": "INITIAL_PLAN_ERROR",
            "tasks": [],
            "guidance": {
                "natural_language_guidance": "规划失败",
                "error": "bad plan",
            },
        },
        {},
    )

    assert payloads[0]["accepted_choices"] == ["RETRY_INITIAL_PLAN", "CANCEL"]
    assert update["planner_action"] == expected_action
    assert update.get("tasks", []) == []


def test_initial_plan_error_routes_never_reach_worker():
    script = textwrap.dedent(
        '''
        import json
        from src.graph import route_planner_confirm

        print(json.dumps({
            "retry": route_planner_confirm({"planner_action": "INITIAL_PLAN_RETRY"}),
            "cancel": route_planner_confirm({"planner_action": "INITIAL_PLAN_CANCELLED"}),
        }))
        '''
    )
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-agent"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    routes = json.loads(completed.stdout.splitlines()[-1])
    assert routes == {"retry": "Planner", "cancel": "Exit"}


def test_intake_propagates_output_format_with_legacy_fallback():
    modern = intake_module.build_task_spec({"output_format": "PDF"}, [])
    legacy = intake_module.build_task_spec({"format": "Word"}, [])

    modern_payload = json.loads(modern["messages"][0].content)
    legacy_payload = json.loads(legacy["messages"][0].content)
    assert modern_payload["output_format"] == "PDF"
    assert legacy_payload["output_format"] == "Word"
    assert modern_payload["web_authorized"] is False


@pytest.mark.parametrize(
    ("raw_request", "expected"),
    [
        ("请结合知识库和公开网络资料撰写报告", True),
        ("除了参考 RAG，还需要参考网络上的公开消息", True),
        ("仅使用知识库，不要联网", False),
        ("撰写一份工艺报告", False),
    ],
)
def test_intake_derives_web_authorization_only_from_explicit_request(
    raw_request,
    expected,
):
    assert intake_module.explicit_web_authorization(raw_request) is expected


@pytest.mark.parametrize(
    ("raw_request", "expected"),
    [
        ("允许使用公开网络资料", True),
        ("不要联网，只使用知识库", False),
        ("不要使用公开网络资料", False),
        ("不需要网络资料", False),
        ("不需要联网", False),
        ("可以联网补充最新资料", True),
        ("调整章节顺序", None),
    ],
)
def test_web_authorization_directive_is_tristate(raw_request, expected):
    assert intake_module.web_authorization_directive(raw_request) is expected


@pytest.mark.parametrize(
    ("feedback", "expected"),
    [
        ("重新规划，并允许使用公开网络资料", True),
        ("重新规划，不要使用公开网络资料", False),
    ],
)
def test_manual_verifier_persists_web_policy_changes_for_full_replan(
    monkeypatch,
    feedback,
    expected,
):
    monkeypatch.setattr(
        verifier_manual_module,
        "interrupt",
        lambda _payload: {"text": feedback, "docs": []},
    )
    monkeypatch.setattr(
        verifier_manual_module,
        "_analyze_feedback",
        lambda *_args, **_kwargs: {
            "decision": "FULL_REPLAN",
            "reason": "用户要求重规划",
            "suggestions": feedback,
        },
    )

    update = verifier_manual_module.verifier_manual(
        {
            "tasks": [{"task_id": "T1", "task_name": "章节一"}],
            "cursor": 0,
            "current_result": {"task_id": "T1", "text_output": "正文"},
            "results": [],
        },
        {},
    )

    assert update["decision"] == "FULL_REPLAN"
    assert update["web_authorized"] is expected


def test_planner_prompts_use_one_valid_json_object_contract():
    root = Path(__file__).parents[1]
    for name in (
        "planner_to_worker.md",
        "planner_replan.md",
        "planner_intake_replan.md",
    ):
        prompt = (root / "src" / "prompts" / name).read_text(encoding="utf-8")
        assert "JSON Array" not in prompt
        assert "JSON 数组" not in prompt
        assert '"tasks"' in prompt
        assert "//" not in prompt
        assert "{core_content}" in prompt
        assert "{style}" in prompt
        assert "{output_format}" in prompt
        assert "{knowledge_catalog}" in prompt


def test_initial_planner_prompt_matches_deterministic_contract():
    root = Path(__file__).parents[1]
    prompt = (root / "src" / "prompts" / "planner_to_worker.md").read_text(
        encoding="utf-8"
    )

    assert "6 至 10" not in prompt
    assert "一一对应" in prompt
    assert "use_rag=false" in prompt
    assert "query" in prompt
    assert "causal" in prompt
    assert "flowchart" not in prompt
    assert "fault_tree" not in prompt
    assert "当前任务是否需要新增知识库证据" in prompt
    assert "知识库依据与说明" in prompt
    assert "不得自行创建" in prompt


def test_all_planner_prompts_share_rag_semantics_and_catalog_boundary():
    root = Path(__file__).parents[1]
    for name in (
        "planner_to_worker.md",
        "planner_replan.md",
        "planner_intake_replan.md",
    ):
        prompt = (root / "src" / "prompts" / name).read_text(encoding="utf-8")
        assert "当前任务是否需要新增知识库证据" in prompt
        assert "知识目录" in prompt
        assert "不得自行创建" in prompt
        assert "不能填写到 `use_resources`" in prompt


def test_plan_guidance_explicitly_uses_json_mode(monkeypatch):
    modes = []

    class Model:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "natural_language_guidance": "请确认",
                        "resource_mapping": {},
                    },
                    ensure_ascii=False,
                )
            )

    def fake_get_llm(_config, *, json_mode):
        modes.append(json_mode)
        return Model()

    monkeypatch.setattr(planner_module, "get_llm", fake_get_llm)

    planner_module._generate_plan_guidance([_task()], [], {})

    assert modes == [True]

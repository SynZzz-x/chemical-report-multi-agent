from copy import deepcopy

from src.nodes import asset_recovery as asset_recovery_module
from src.concept_graph.attempts import concept_graph_attempt_key


def _state(*, task=None, result=None, issues=None):
    current_task = {
        "task_id": "T2",
        "task_name": "工艺参数影响",
        "generate_table": False,
        "generate_figure": False,
        **(task or {}),
    }
    current_result = {
        "task_id": "T2",
        "text_output": "正文保持不变。",
        "tables": [],
        "figures": [],
        "citations": [],
        "tool_calls": [],
        "sources_used": ["source.docx"],
        "evidence_coverage": {"status": "sufficient"},
        "graph_spec": {},
        "figures_generated": 0,
        **(result or {}),
    }
    return {
        "user_id": "u1",
        "conversation_id": "c1",
        "job_id": "j1",
        "tasks": [current_task],
        "cursor": 0,
        "current_result": current_result,
        "assessment": {
            "status": "FAILED",
            "issues": list(issues or []),
        },
        "worker_state": {"retained": True},
    }


def test_markdown_table_is_materialized_without_rewriting_body():
    text = """正文保持不变。

| 参数 | 影响 |
| --- | --- |
| 温度 | 熔融指数 [E1] |
"""
    state = _state(
        task={"generate_table": True},
        result={"text_output": text},
        issues=[{"code": "MISSING_TABLE"}],
    )
    original = deepcopy(state["current_result"])

    update = asset_recovery_module.asset_recovery(state, {})

    assert update["workflow_action"] == "RETRY_VERIFIER"
    assert update["current_result"]["text_output"] == original["text_output"]
    assert update["current_result"]["citations"] == original["citations"]
    assert update["current_result"]["tool_calls"] == original["tool_calls"]
    assert update["current_result"]["tables"][0]["headers"] == ["参数", "影响"]


def test_missing_table_content_routes_to_focused_worker_rework():
    state = _state(
        task={"generate_table": True},
        issues=[{"code": "MISSING_TABLE", "description": "缺少要求的表格"}],
    )
    original = deepcopy(state["current_result"])

    update = asset_recovery_module.asset_recovery(state, {})

    assert update["workflow_action"] == "REWORK"
    assert update["current_result"] == original
    feedback = update["worker_state"]["execution_feedback"]
    assert feedback["mode"] == "asset_content_rework"
    assert "Markdown 表格" in feedback["instructions"]


def test_stale_result_from_another_task_is_never_repaired():
    state = _state(
        task={"task_id": "T2", "generate_table": True},
        result={
            "task_id": "T1",
            "text_output": (
                "旧任务正文。\n\n"
                "| 参数 | 影响 |\n"
                "| --- | --- |\n"
                "| 温度 | 指标 |\n"
            ),
        },
        issues=[{"code": "MISSING_TABLE"}],
    )
    original = deepcopy(state["current_result"])

    update = asset_recovery_module.asset_recovery(state, {})

    assert update["workflow_action"] == "REWORK"
    assert update["current_result"] == original
    assert "任务不一致" in update["asset_recovery_error"]


def test_causal_figure_reuses_existing_evidence_and_preserves_result(monkeypatch):
    captured = {}

    class FakeTool:
        def execute(self, task, evidence, output_dir):
            captured["task"] = task
            captured["evidence"] = evidence
            captured["output_dir"] = output_dir
            return {
                "success": True,
                "figure": {"path": "/tmp/recovered.png", "evidence_ids": ["E1"]},
                "graph_spec": {"graph_type": "causal"},
            }

    monkeypatch.setattr(asset_recovery_module, "ConceptGraphTool", FakeTool)
    monkeypatch.setattr(
        asset_recovery_module, "get_job_subdir", lambda *args, **kwargs: "/tmp/charts"
    )
    state = _state(
        task={
            "generate_figure": True,
            "visualization": {
                "kind": "causal",
                "title": "温度与熔融指数",
                "required_concepts": ["反应温度", "熔融指数"],
            },
        },
        result={
            "citations": [
                {
                    "evidence_id": "E1",
                    "source_type": "rag",
                    "title": "工艺说明",
                    "supporting_text": "反应温度升高会提高熔融指数。",
                    "file_path": "/srv/process.docx",
                }
            ],
            "tool_calls": [{"tool": "other_tool", "success": True}],
        },
        issues=[{"code": "MISSING_FIGURE"}],
    )
    original = deepcopy(state["current_result"])

    update = asset_recovery_module.asset_recovery(state, {})

    candidate = update["current_result"]
    assert update["workflow_action"] == "RETRY_VERIFIER"
    assert candidate["text_output"] == original["text_output"]
    assert candidate["citations"] == original["citations"]
    assert candidate["tool_calls"] == original["tool_calls"]
    assert candidate["sources_used"] == original["sources_used"]
    assert candidate["figures"] == [
        {"path": "/tmp/recovered.png", "evidence_ids": ["E1"]}
    ]
    assert candidate["graph_spec"] == {"graph_type": "causal"}
    assert candidate["figures_generated"] == 1
    assert [record.evidence_id for record in captured["evidence"].records] == ["E1"]


def test_failed_causal_figure_recovery_keeps_original_result(monkeypatch):
    class FailingTool:
        def execute(self, task, evidence, output_dir):
            return {"success": False, "error": "graph validation failed"}

    monkeypatch.setattr(asset_recovery_module, "ConceptGraphTool", FailingTool)
    monkeypatch.setattr(
        asset_recovery_module, "get_job_subdir", lambda *args, **kwargs: "/tmp/charts"
    )
    state = _state(
        task={
            "generate_figure": True,
            "visualization": {
                "kind": "causal",
                "required_concepts": ["反应温度", "熔融指数"],
            },
        },
        result={
            "citations": [
                {
                    "evidence_id": "E1",
                    "source_type": "rag",
                    "title": "工艺说明",
                    "supporting_text": "反应温度升高会提高熔融指数。",
                    "file_path": "/srv/process.docx",
                }
            ]
        },
        issues=[{"code": "MISSING_FIGURE"}],
    )
    original = deepcopy(state["current_result"])

    update = asset_recovery_module.asset_recovery(state, {})

    assert update["workflow_action"] == "RETRY_VERIFIER"
    assert update["current_result"] == original
    assert update["asset_recovery_error"] == "graph validation failed"


def test_asset_recovery_never_repeats_completed_concept_graph_semantic_attempt(
    monkeypatch,
):
    monkeypatch.setattr(
        asset_recovery_module,
        "ConceptGraphTool",
        lambda: (_ for _ in ()).throw(
            AssertionError("asset recovery must not repeat semantic extraction")
        ),
    )
    state = _state(
        task={
            "generate_figure": True,
            "visualization": {
                "kind": "causal",
                "required_concepts": ["反应温度", "熔融指数"],
            },
        },
        result={
            "citations": [
                {
                    "evidence_id": "E1",
                    "source_type": "rag",
                    "title": "工艺说明",
                    "supporting_text": "反应温度升高会提高熔融指数。",
                    "file_path": "/srv/process.docx",
                }
            ],
            "graph_spec": {},
        },
        issues=[{"code": "MISSING_FIGURE"}],
    )
    state["task_revisions"] = {"T2": 1}
    state["concept_graph_attempts"] = {
        concept_graph_attempt_key("T2", 1): 1
    }

    update = asset_recovery_module.asset_recovery(state, {})

    assert update["workflow_action"] == "RETRY_VERIFIER"
    assert update["current_result"] == state["current_result"]
    assert "semantic attempt limit" in update["asset_recovery_error"]


def test_multi_asset_failure_does_not_commit_partially_materialized_table(monkeypatch):
    class FailingTool:
        def execute(self, task, evidence, output_dir):
            return {"success": False, "error": "graph validation failed"}

    monkeypatch.setattr(asset_recovery_module, "ConceptGraphTool", FailingTool)
    monkeypatch.setattr(
        asset_recovery_module, "get_job_subdir", lambda *args, **kwargs: "/tmp/charts"
    )
    state = _state(
        task={
            "generate_table": True,
            "generate_figure": True,
            "visualization": {
                "kind": "causal",
                "required_concepts": ["反应温度", "熔融指数"],
            },
        },
        result={
            "text_output": (
                "正文。\n\n| 参数 | 指标 |\n| --- | --- |\n"
                "| 反应温度 | 熔融指数 [E1] |\n"
            ),
            "citations": [
                {
                    "evidence_id": "E1",
                    "source_type": "rag",
                    "title": "工艺说明",
                    "supporting_text": "反应温度升高会提高熔融指数。",
                    "file_path": "/srv/process.docx",
                }
            ],
        },
        issues=[{"code": "MISSING_TABLE"}, {"code": "MISSING_FIGURE"}],
    )
    original = deepcopy(state["current_result"])

    update = asset_recovery_module.asset_recovery(state, {})

    assert update["workflow_action"] == "RETRY_VERIFIER"
    assert update["current_result"] == original


def test_non_causal_figure_routes_to_worker_without_calling_concept_graph(monkeypatch):
    monkeypatch.setattr(
        asset_recovery_module,
        "ConceptGraphTool",
        lambda: (_ for _ in ()).throw(
            AssertionError("generic figures must not use the causal graph tool")
        ),
    )
    state = _state(
        task={
            "generate_figure": True,
            "visualization": {"kind": "heatmap"},
        },
        issues=[{"code": "MISSING_FIGURE"}],
    )

    update = asset_recovery_module.asset_recovery(state, {})

    assert update["workflow_action"] == "REWORK"
    assert update["worker_state"]["execution_feedback"]["mode"] == (
        "asset_content_rework"
    )

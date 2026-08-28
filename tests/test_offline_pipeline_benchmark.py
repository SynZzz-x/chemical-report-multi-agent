"""Characterize the current offline pipeline before optimization work begins."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.nodes import intake as intake_module
from src.nodes import planner as planner_module
from src.nodes import verifier as verifier_module
from src.nodes.worker.agent import graph as worker_graph_module
from src.nodes.worker.agent.graph import AutonomousToolNode
from tests.benchmark_support import (
    BenchmarkRecorder,
    FAILED_SEMANTIC_VERIFIER_RESPONSE,
    PASS_VERIFIER_RESPONSE,
    SCENARIO_A_TASK,
    SCENARIO_A_WORKER_RESPONSE,
    SCENARIO_B_FINAL_WORKER_RESPONSE,
    SCENARIO_B_PREFETCH_RESPONSE,
    SCENARIO_B_QUERY,
    SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE,
    SCENARIO_B_TASK,
    measure_serialized_messages,
    serialized_chars,
)
from src.verifier_contract import parse_verifier_assessment


class _SequenceRecorder:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[Any] = []

    def invoke(self, value: Any, **kwargs: Any) -> Any:
        self.calls.append(value)
        return self.responses.pop(0)

    @property
    def prompt_chars(self) -> int:
        return sum(serialized_chars(value) for value in self.calls)


class _KnowledgeTool:
    name = "chemical_knowledge_base_tool"
    description = "检索离线化工知识库。"

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(parameters)
        return SCENARIO_B_PREFETCH_RESPONSE


def _fake_invoke(model: Any, value: Any, **kwargs: Any) -> Any:
    return model.invoke(value)


def _planner_response() -> str:
    return json.dumps(
        {
            "tasks": [
                {
                    "task_id": "T1",
                    "task_name": "聚乙烯质量异常排查",
                    "task_description": "基于知识库证据分析聚乙烯常见质量异常并提出排查建议。",
                    "task_type": "analysis",
                    "use_rag": True,
                    "use_web": False,
                    "query": SCENARIO_B_QUERY,
                    "use_resources": [],
                    "generate_figure": False,
                    "generate_table": False,
                    "visualization": None,
                    "covers_sections": ["聚乙烯质量异常排查"],
                    "requirement_ids": [],
                    "depends_on_task_ids": [],
                }
            ]
        },
        ensure_ascii=False,
    )


def _worker_initial_messages(
    node: AutonomousToolNode, task: dict[str, Any], tools: list[Any], prefetched: list[dict[str, Any]]
) -> list[Any]:
    """Build the production Worker prompt path, including prefetch context."""

    messages = [
        worker_graph_module.SystemMessage(content=node._build_system_prompt(task, tools)),
        worker_graph_module.HumanMessage(content=node._build_task_prompt(task)),
    ]
    if prefetched:
        evidence_context = node._evidence_context_for_generation(prefetched)
        if evidence_context:
            messages.append(worker_graph_module.HumanMessage(content=evidence_context))
    return messages


def _collect_pipeline_metrics_in_process() -> dict[str, int]:
    """Run the real production prompt construction with offline fakes only."""

    intake_response = {
        "is_chat": False,
        "user_intent": "排查聚乙烯常见质量异常并提出建议。",
        "task_type": "工程分析报告",
        "title": "聚乙烯质量异常排查",
        "doc_length": "不限",
        "constraints": ["仅使用离线知识库"],
        "style": "formal",
        "output_format": "Markdown",
        "web_authorized": False,
        "sections": ["聚乙烯质量异常排查"],
        "core_content": ["熔融指数", "灰分", "凝胶含量"],
    }
    intake_recorder = BenchmarkRecorder(
        SimpleNamespace(content=json.dumps(intake_response, ensure_ascii=False))
    )
    planner_recorder = BenchmarkRecorder(
        SimpleNamespace(content=_planner_response())
    )
    verifier_recorder = BenchmarkRecorder(
        SimpleNamespace(content=json.dumps(PASS_VERIFIER_RESPONSE, ensure_ascii=False))
    )
    worker_a_recorder = BenchmarkRecorder(SCENARIO_A_WORKER_RESPONSE)
    worker_b_recorder = _SequenceRecorder([SCENARIO_B_FINAL_WORKER_RESPONSE])
    knowledge_tool = _KnowledgeTool()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(intake_module, "get_llm", lambda *args, **kwargs: intake_recorder)
        monkeypatch.setattr(intake_module, "invoke_llm", _fake_invoke)
        parsed_intake = intake_module.llm_parse_user_need(
            "请分析聚乙烯质量异常，给出排查建议。", {"configurable": {}}
        )

        monkeypatch.setattr(planner_module, "get_llm", lambda *args, **kwargs: planner_recorder)
        monkeypatch.setattr(planner_module, "invoke_llm", _fake_invoke)
        monkeypatch.setattr(planner_module, "load_active_catalog", lambda: [])
        planner_module._build_tasks_with_llm(
            {**parsed_intake, "requirements": []}, {"configurable": {}}
        )

        monkeypatch.setattr(
            worker_graph_module,
            "get_app_config",
            lambda: SimpleNamespace(
                concept_graph_settings=SimpleNamespace(rag_max_queries=3, rag_adaptive_reserve=1)
            ),
        )
        monkeypatch.setattr(worker_graph_module, "invoke_llm", _fake_invoke)
        node = AutonomousToolNode.__new__(AutonomousToolNode)
        node.config = SimpleNamespace(MAX_TOOL_ITERATIONS=3, MAX_CHARTS_PER_TASK=0)
        node.llm_client = worker_a_recorder
        scenario_a_messages = _worker_initial_messages(node, SCENARIO_A_TASK, [], [])
        node._execute_tool_loop(worker_a_recorder, scenario_a_messages, [], SCENARIO_A_TASK)

        prefetched = AutonomousToolNode._prefetch_rag(SCENARIO_B_TASK, [knowledge_tool])
        node.llm_client = worker_b_recorder
        scenario_b_messages = _worker_initial_messages(
            node, SCENARIO_B_TASK, [knowledge_tool], prefetched
        )
        node._execute_tool_loop(
            worker_b_recorder,
            scenario_b_messages,
            [knowledge_tool],
            SCENARIO_B_TASK,
            prefetched,
        )

        monkeypatch.setattr(verifier_module, "get_llm", lambda *args, **kwargs: verifier_recorder)
        monkeypatch.setattr(verifier_module, "invoke_llm", _fake_invoke)
        verifier_update = verifier_module.verifier(
            {
                "tasks": [
                    {
                        "task_id": "T1",
                        "task_name": "聚乙烯质量异常排查",
                        "task_description": "基于知识库证据分析聚乙烯常见质量异常。",
                        "task_type": "analysis",
                        "use_rag": True,
                        "use_web": False,
                        "requirement_ids": [],
                    }
                ],
                "cursor": 0,
                "current_result": {
                    "status": "COMPLETED",
                    "text_output": "已结合离线证据完成质量异常排查。",
                    "tables": [],
                    "figures": [],
                    "citations": [],
                    "report_sources": [],
                },
                "requirement_registry": [],
            },
            {"configurable": {"use_llm": True}},
        )

    assert verifier_update["assessment"]["status"] == "PASS"
    attempted_adaptive_retrievals = [
        tool_call
        for response in (SCENARIO_B_FINAL_WORKER_RESPONSE,)
        for tool_call in getattr(response, "tool_calls", [])
        if tool_call.get("name") == "chemical_knowledge_base_tool"
    ]
    prefetch_retrieval_calls = len(prefetched)
    adaptive_retrieval_calls = len(knowledge_tool.calls) - prefetch_retrieval_calls
    duplicate_retrievals = len(attempted_adaptive_retrievals) - adaptive_retrieval_calls
    assert len(attempted_adaptive_retrievals) == 0
    assert adaptive_retrieval_calls == 0
    assert duplicate_retrievals == 0
    failed_semantic_assessment = parse_verifier_assessment(
        json.dumps(FAILED_SEMANTIC_VERIFIER_RESPONSE, ensure_ascii=False)
    )
    failed_semantic_verifier_issues = sum(
        issue.code == "CLAIM_PARTIALLY_SUPPORTED"
        for issue in failed_semantic_assessment.issues
    )
    assert failed_semantic_verifier_issues == 1
    prompt_recorders = (
        intake_recorder,
        planner_recorder,
        worker_a_recorder,
        worker_b_recorder,
        verifier_recorder,
    )
    completion_responses = (
        intake_recorder.response,
        planner_recorder.response,
        worker_a_recorder.response,
        SCENARIO_B_FINAL_WORKER_RESPONSE,
        verifier_recorder.response,
    )
    serialized_prompt_chars = sum(
        measure_serialized_messages(recorder.calls)["serialized_prompt_chars"]
        for recorder in prompt_recorders
    )
    mock_completion_chars = sum(
        serialized_chars(getattr(response, "content", response))
        for response in completion_responses
    )
    return {
        "intake_llm_calls": len(intake_recorder.calls),
        "planner_llm_calls": len(planner_recorder.calls),
        "worker_llm_calls": len(worker_a_recorder.calls) + len(worker_b_recorder.calls),
        "verifier_llm_calls": len(verifier_recorder.calls),
        "worker_generations": len(worker_a_recorder.calls) + len(worker_b_recorder.calls),
        "worker_tool_loop_iterations": len(worker_a_recorder.calls) + len(worker_b_recorder.calls),
        "prefetch_retrieval_calls": prefetch_retrieval_calls,
        "adaptive_retrieval_calls": adaptive_retrieval_calls,
        "duplicate_retrievals": duplicate_retrievals,
        "failed_semantic_verifier_issues": failed_semantic_verifier_issues,
        "total_llm_calls": sum(len(recorder.calls) for recorder in prompt_recorders),
        "serialized_prompt_chars": serialized_prompt_chars,
        "mock_completion_chars": mock_completion_chars,
    }


def collect_pipeline_metrics() -> dict[str, int]:
    """Collect with installed production LangChain prompts, outside pytest stubs."""

    if os.environ.get("OFFLINE_PIPELINE_BENCHMARK_CHILD") == "1":
        return _collect_pipeline_metrics_in_process()
    command = (
        "import json; "
        "from tests.test_offline_pipeline_benchmark import collect_pipeline_metrics; "
        "print('__OFFLINE_PIPELINE_METRICS__' + json.dumps(collect_pipeline_metrics()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OFFLINE_PIPELINE_BENCHMARK_CHILD": "1"},
    )
    marker = "__OFFLINE_PIPELINE_METRICS__"
    metrics_line = next(
        line for line in reversed(result.stdout.splitlines()) if line.startswith(marker)
    )
    return json.loads(metrics_line.removeprefix(marker))


def _verifier_benchmark_state() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": "T1",
                "task_name": "聚乙烯质量异常排查",
                "task_description": "基于知识库证据分析聚乙烯常见质量异常。",
                "task_type": "analysis",
                "use_rag": True,
                "use_web": False,
                "requirement_ids": [],
            }
        ],
        "cursor": 0,
        "current_result": {
            "status": "COMPLETED",
            "text_output": "氢气是第一优先排查项并直接决定熔融指数。[E1]",
            "tables": [],
            "figures": [],
            "citations": [
                {
                    "evidence_id": "E1",
                    "title": "聚乙烯质量控制手册",
                    "locator": "§3.2",
                    "supporting_text": "氢气用量会影响聚乙烯熔融指数。",
                }
            ],
            "report_sources": ["聚乙烯质量控制手册"],
        },
        "requirement_registry": [],
    }


def _run_verifier_fixture(response_payload: dict[str, Any]):
    recorder = BenchmarkRecorder(
        SimpleNamespace(content=json.dumps(response_payload, ensure_ascii=False))
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            verifier_module, "get_llm", lambda *args, **kwargs: recorder
        )
        monkeypatch.setattr(verifier_module, "invoke_llm", _fake_invoke)
        update = verifier_module.verifier(
            _verifier_benchmark_state(),
            {"configurable": {"use_llm": True}},
        )
    return recorder, update


def test_compact_verifier_pass_and_failed_outputs_keep_one_semantic_call():
    failed_payload = json.loads(
        json.dumps(FAILED_SEMANTIC_VERIFIER_RESPONSE, ensure_ascii=False)
    )
    failed_payload["issues"][0].update(
        {
            "description": (
                "论断‘氢气是第一优先排查项并直接决定熔融指数’增加了未获支持的"
                "优先级和直接因果。"
            ),
            "retrieval_query": "聚乙烯 氢气 熔融指数 优先级 直接因果",
        }
    )

    pass_recorder, pass_update = _run_verifier_fixture(PASS_VERIFIER_RESPONSE)
    failed_recorder, failed_update = _run_verifier_fixture(failed_payload)

    assert len(pass_recorder.calls) == 1
    assert len(failed_recorder.calls) == 1
    assert set(PASS_VERIFIER_RESPONSE) == {
        "status",
        "current_section",
        "issues",
        "requirements_met",
        "requirements_missing",
    }
    assert PASS_VERIFIER_RESPONSE["issues"] == []
    assert PASS_VERIFIER_RESPONSE["requirements_met"] == []
    assert PASS_VERIFIER_RESPONSE["requirements_missing"] == []
    assert pass_update["assessment"]["status"] == "PASS"

    issue = failed_update["assessment"]["issues"][0]
    assert failed_update["assessment"]["status"] == "FAILED"
    assert issue["code"] == "CLAIM_PARTIALLY_SUPPORTED"
    assert "氢气是第一优先排查项" in issue["description"]
    assert issue["suggestion"] == "缩小结论范围或补充直接证据。"
    assert issue["retrieval_query"] == "聚乙烯 氢气 熔融指数 优先级 直接因果"
    assert serialized_chars(PASS_VERIFIER_RESPONSE) < serialized_chars(failed_payload)


def test_offline_benchmark_metrics_are_deterministic():
    first = collect_pipeline_metrics()
    second = collect_pipeline_metrics()
    baseline_path = Path(__file__).parents[1] / "docs/benchmarks/2026-08-28-pipeline-baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert first == second
    baseline_metrics = baseline["measurement"]["metrics"]
    optimized_metrics = {
        "worker_llm_calls",
        "worker_generations",
        "worker_tool_loop_iterations",
        "duplicate_retrievals",
        "total_llm_calls",
    }
    character_metrics = {"serialized_prompt_chars", "mock_completion_chars"}
    assert {
        key: value
        for key, value in first.items()
        if key not in character_metrics | optimized_metrics
    } == {
        key: value
        for key, value in baseline_metrics.items()
        if key not in character_metrics | optimized_metrics
    }
    assert first["serialized_prompt_chars"] <= baseline_metrics["serialized_prompt_chars"]
    assert first["mock_completion_chars"] <= baseline_metrics["mock_completion_chars"]
    assert first["worker_llm_calls"] == 2
    assert first["worker_generations"] == 2
    assert first["worker_tool_loop_iterations"] == 2
    assert first["prefetch_retrieval_calls"] == 1
    assert first["adaptive_retrieval_calls"] == 0
    assert first["duplicate_retrievals"] == 0
    assert first["total_llm_calls"] == 5
    assert first["worker_llm_calls"] == baseline_metrics["worker_llm_calls"] - 1
    assert first["worker_tool_loop_iterations"] == (
        baseline_metrics["worker_tool_loop_iterations"] - 1
    )
    assert first["duplicate_retrievals"] == baseline_metrics["duplicate_retrievals"] - 1
    assert first["total_llm_calls"] >= 1
    assert first["serialized_prompt_chars"] > 0
    assert first["mock_completion_chars"] > 0

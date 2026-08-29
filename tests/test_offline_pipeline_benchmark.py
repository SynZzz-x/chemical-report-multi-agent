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
from src.evidence.query_identity import normalize_query_identity
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
    VERIFIER_PASS_STATE,
    measure_serialized_messages,
    serialize_emitted_response,
    serialized_chars,
)
from src.verifier_contract import parse_verifier_assessment


class _PrefetchAwareWorkerRecorder:
    """Deterministically model the effect of completed first-turn prefetch context."""

    _REQUIRED_MARKERS = (
        "预检索已完成",
        "具体的新证据缺口",
        '"prefetched_queries"',
        '"query_fingerprints"',
        '"evidence"',
        '"evidence_id": "E1"',
        '"supporting_text_excerpt"',
        '"prefetch_queries_used"',
        '"adaptive_queries_remaining"',
        "聚乙烯 质量异常 排查 建议 常见异常",
        "熔融指数、灰分和凝胶含量异常",
    )

    def __init__(self):
        self.calls: list[Any] = []
        self.emitted_responses: list[Any] = []

    def invoke(self, value: Any, **kwargs: Any) -> Any:
        self.calls.append(value)
        if isinstance(value, (list, tuple)):
            serialized = "\n".join(
                str(getattr(message, "content", message)) for message in value
            )
        else:
            serialized = str(value)
        if len(self.calls) == 1 and all(
            marker in serialized for marker in self._REQUIRED_MARKERS
        ):
            response = SCENARIO_B_FINAL_WORKER_RESPONSE
        elif len(self.calls) == 1:
            response = SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE
        else:
            response = SCENARIO_B_FINAL_WORKER_RESPONSE
        self.emitted_responses.append(response)
        return response

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
    worker_b_recorder = _PrefetchAwareWorkerRecorder()
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
        for response in worker_b_recorder.emitted_responses
        for tool_call in getattr(response, "tool_calls", [])
        if tool_call.get("name") == "chemical_knowledge_base_tool"
    ]
    prefetch_retrieval_calls = len(prefetched)
    adaptive_retrieval_calls = len(knowledge_tool.calls) - prefetch_retrieval_calls
    prefetch_query_identity = normalize_query_identity(SCENARIO_B_QUERY)
    duplicate_query_requests = sum(
        normalize_query_identity((tool_call.get("args") or {}).get("query"))
        == prefetch_query_identity
        for tool_call in attempted_adaptive_retrievals
    )
    adaptive_tool_parameters = knowledge_tool.calls[prefetch_retrieval_calls:]
    duplicate_retrieval_executions = sum(
        normalize_query_identity(parameters.get("query")) == prefetch_query_identity
        for parameters in adaptive_tool_parameters
    )
    duplicate_guard_rejections = (
        duplicate_query_requests - duplicate_retrieval_executions
    )
    assert len(attempted_adaptive_retrievals) == 0
    assert adaptive_retrieval_calls == 0
    assert duplicate_query_requests == 0
    assert duplicate_guard_rejections == 0
    assert duplicate_retrieval_executions == 0
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
        *worker_b_recorder.emitted_responses,
        verifier_recorder.response,
    )
    serialized_prompt_chars = sum(
        measure_serialized_messages(recorder.calls)["serialized_prompt_chars"]
        for recorder in prompt_recorders
    )
    mock_completion_chars = sum(
        len(serialize_emitted_response(response)) for response in completion_responses
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
        "duplicate_query_requests": duplicate_query_requests,
        "duplicate_guard_rejections": duplicate_guard_rejections,
        "duplicate_retrieval_executions": duplicate_retrieval_executions,
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
            VERIFIER_PASS_STATE,
            {"configurable": {"use_llm": True}},
        )
    return recorder, update


def collect_verifier_pass_metrics() -> dict[str, int]:
    """Measure the real production prompt renderer with the fixed PASS fake."""

    if os.environ.get("OFFLINE_VERIFIER_BENCHMARK_CHILD") == "1":
        recorder, update = _run_verifier_fixture(PASS_VERIFIER_RESPONSE)
        assert update["assessment"]["status"] == "PASS"
        return {
            "serialized_prompt_chars": measure_serialized_messages(recorder.calls)[
                "serialized_prompt_chars"
            ],
            "mock_completion_chars": len(
                serialize_emitted_response(recorder.response)
            ),
            "semantic_llm_calls": len(recorder.calls),
        }
    command = (
        "import json; "
        "from tests.test_offline_pipeline_benchmark import collect_verifier_pass_metrics; "
        "print('__OFFLINE_VERIFIER_METRICS__' + "
        "json.dumps(collect_verifier_pass_metrics()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OFFLINE_VERIFIER_BENCHMARK_CHILD": "1"},
    )
    marker = "__OFFLINE_VERIFIER_METRICS__"
    metrics_line = next(
        line for line in reversed(result.stdout.splitlines()) if line.startswith(marker)
    )
    return json.loads(metrics_line.removeprefix(marker))


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


def test_mock_completion_serialization_includes_structured_tool_calls():
    serialized = serialize_emitted_response(SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE)

    assert '"content":""' in serialized
    assert '"tool_calls"' in serialized
    assert "repeat-prefetch-query" in serialized
    assert len(serialized) > len(SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE.content)


def test_verifier_pass_benchmark_records_exact_snapshot_comparison():
    baseline_path = Path(__file__).parents[1] / "docs/benchmarks/2026-08-28-pipeline-baseline.json"
    artifact = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparison = artifact["verifier_pass_comparison"]
    current = collect_verifier_pass_metrics()

    assert comparison["fixture"] == "VERIFIER_PASS_STATE + PASS_VERIFIER_RESPONSE"
    assert comparison["baseline"]["commit"] == (
        "3ba9fd3eb3ad84b193f699e72e15bc40bea40446"
    )
    assert comparison["baseline"]["semantic_llm_calls"] == 1
    assert comparison["optimized"] == current
    assert comparison["optimized"]["serialized_prompt_chars"] < (
        comparison["baseline"]["serialized_prompt_chars"]
    )


def test_worker_benchmark_fake_requires_completed_prefetch_inventory():
    recorder = _PrefetchAwareWorkerRecorder()

    without_inventory = recorder.invoke(
        [SimpleNamespace(content="请完成任务，知识库可用。")]
    )
    with_inventory = _PrefetchAwareWorkerRecorder().invoke(
        [
            SimpleNamespace(
                content=(
                    "知识库预检索已完成；只有具体的新证据缺口才可检索。"
                    ' {"prefetched_queries": ["聚乙烯 质量异常 排查 建议 常见异常"], '
                    '"query_fingerprints": ["abc"], '
                    '"evidence": [{"evidence_id": "E1", '
                    '"supporting_text_excerpt": "熔融指数、灰分和凝胶含量异常"}], '
                    '"prefetch_queries_used": 1, "adaptive_queries_remaining": 2}'
                )
            )
        ]
    )

    assert without_inventory is SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE
    assert with_inventory is SCENARIO_B_FINAL_WORKER_RESPONSE


def test_offline_benchmark_metrics_are_deterministic():
    first = collect_pipeline_metrics()
    second = collect_pipeline_metrics()
    baseline_path = Path(__file__).parents[1] / "docs/benchmarks/2026-08-28-pipeline-baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert first == second
    baseline_metrics = baseline["measurement"]["metrics"]
    optimized = baseline["optimized"]
    optimized_metrics = {
        "worker_llm_calls",
        "worker_generations",
        "worker_tool_loop_iterations",
        "duplicate_query_requests",
        "duplicate_guard_rejections",
        "duplicate_retrieval_executions",
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
    assert first["serialized_prompt_chars"] < 14500
    assert first["mock_completion_chars"] <= baseline_metrics["mock_completion_chars"]
    assert baseline_metrics["mock_completion_chars"] - first["mock_completion_chars"] == len(
        serialize_emitted_response(SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE)
    )
    assert first["worker_llm_calls"] == 2
    assert first["worker_generations"] == 2
    assert first["worker_tool_loop_iterations"] == 2
    assert first["prefetch_retrieval_calls"] == 1
    assert first["adaptive_retrieval_calls"] == 0
    assert first["duplicate_query_requests"] == 0
    assert first["duplicate_guard_rejections"] == 0
    assert first["duplicate_retrieval_executions"] == 0
    assert first["total_llm_calls"] == 5
    assert first["worker_llm_calls"] == baseline_metrics["worker_llm_calls"] - 1
    assert first["worker_tool_loop_iterations"] == (
        baseline_metrics["worker_tool_loop_iterations"] - 1
    )
    assert first["duplicate_query_requests"] == baseline_metrics["duplicate_query_requests"] - 1
    assert first["duplicate_guard_rejections"] == baseline_metrics["duplicate_guard_rejections"] - 1
    assert baseline_metrics["duplicate_query_requests"] == 1
    assert baseline_metrics["duplicate_guard_rejections"] == 1
    assert baseline_metrics["duplicate_retrieval_executions"] == 0
    assert first["total_llm_calls"] >= 1
    assert first["serialized_prompt_chars"] > 0
    assert first["mock_completion_chars"] > 0
    assert optimized["metrics"] == first
    assert optimized["offline_local_execution"]["offline_only"] is True
    assert optimized["offline_local_execution"]["not_an_online_latency_claim"] is True
    assert optimized["provider_tokens"] is None
    assert optimized["online_latency_seconds"] is None
    assert optimized["requires_real_run"] is True

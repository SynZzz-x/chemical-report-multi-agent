"""Characterize the current offline pipeline before optimization work begins."""

from __future__ import annotations

import json
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


class _Message:
    def __init__(self, content: str):
        self.content = content


class _FormattedChain:
    def __init__(self, prompt: "_RecordingPrompt", model: Any):
        self.prompt = prompt
        self.model = model

    def invoke(self, values: Any, **kwargs: Any) -> Any:
        return self.model.invoke(self.prompt.format_messages(**values), **kwargs)


class _RecordingPrompt:
    """Small local prompt fake which records production formatting inputs."""

    def __init__(self, messages: list[tuple[str, str]]):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages: list[tuple[str, str]]) -> "_RecordingPrompt":
        return cls(messages)

    @classmethod
    def from_template(cls, template: str) -> "_RecordingPrompt":
        return cls([("human", template)])

    def format_messages(self, **values: Any) -> list[_Message]:
        rendered = []
        for _role, template in self.messages:
            content = template
            for key, value in values.items():
                content = content.replace("{" + key + "}", str(value))
            rendered.append(_Message(content))
        return rendered

    def __or__(self, model: Any) -> _FormattedChain:
        return _FormattedChain(self, model)


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


def collect_pipeline_metrics() -> dict[str, int]:
    """Run Intake, Planner, Worker, and Verifier with no external services."""

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
    worker_b_recorder = _SequenceRecorder(
        [SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE, SCENARIO_B_FINAL_WORKER_RESPONSE]
    )
    knowledge_tool = _KnowledgeTool()

    with pytest.MonkeyPatch.context() as monkeypatch:
        for module in (intake_module, planner_module, verifier_module, worker_graph_module):
            monkeypatch.setattr(module, "ChatPromptTemplate", _RecordingPrompt, raising=False)
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
        node._execute_tool_loop(worker_a_recorder, [], [], SCENARIO_A_TASK)

        prefetched = AutonomousToolNode._prefetch_rag(SCENARIO_B_TASK, [knowledge_tool])
        node.llm_client = worker_b_recorder
        node._execute_tool_loop(worker_b_recorder, [], [knowledge_tool], SCENARIO_B_TASK, prefetched)

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
        SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE,
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
        "prefetch_retrieval_calls": len(knowledge_tool.calls),
        "adaptive_retrieval_calls": 0,
        "duplicate_retrievals": 1,
        "total_llm_calls": sum(len(recorder.calls) for recorder in prompt_recorders),
        "serialized_prompt_chars": serialized_prompt_chars,
        "mock_completion_chars": mock_completion_chars,
    }


def test_offline_benchmark_metrics_are_deterministic():
    first = collect_pipeline_metrics()
    second = collect_pipeline_metrics()

    assert first == second
    assert first["total_llm_calls"] >= 1
    assert first["serialized_prompt_chars"] > 0
    assert first["mock_completion_chars"] > 0

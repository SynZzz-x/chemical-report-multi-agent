from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
from types import SimpleNamespace
import json

import pytest

from src.evidence.models import EvidenceBundle, EvidenceRecord
from src.llm import extract_token_usage, invoke_llm
from src.nodes import intake as intake_module
from src.nodes import planner as planner_module
from src.nodes import verifier as verifier_module
from src.nodes.worker.agent.graph import AutonomousToolNode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StubRunnable:
    model_name = "deepseek-chat"

    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, value, **kwargs):
        self.calls.append((value, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                }
            ),
            (11, 7, 18),
        ),
        (
            SimpleNamespace(
                usage_metadata=None,
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 13,
                        "completion_tokens": 5,
                        "total_tokens": 18,
                    }
                },
            ),
            (13, 5, 18),
        ),
        (SimpleNamespace(), (None, None, None)),
    ],
)
def test_extract_token_usage_supports_langchain_shapes(response, expected):
    assert extract_token_usage(response) == expected


def test_invoke_llm_logs_success_and_preserves_response_identity(caplog):
    caplog.set_level("INFO", logger="src.llm.observability")
    response = SimpleNamespace(
        content="secret response",
        usage_metadata={
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        },
    )
    runnable = StubRunnable(response=response)
    messages = ["secret prompt"]
    config = {
        "configurable": {"thread_id": "thread-1"},
        "metadata": {"job_id": "job-1"},
    }

    actual = invoke_llm(
        runnable,
        messages,
        config=config,
        node="Planner",
        purpose="plan_generation",
        attempt=2,
        task_id="T2",
        plan_revision=3,
        task_revision=4,
        json_mode=True,
    )

    assert actual is response
    assert runnable.calls == [(messages, {"config": config})]
    assert len(caplog.messages) == 2
    start, end = caplog.messages
    assert start.startswith("LLM_CALL_START ")
    assert end.startswith("LLM_CALL_END ")
    for field in (
        "call_id=",
        "job_id=job-1",
        "thread_id=thread-1",
        "node=Planner",
        "purpose=plan_generation",
        "task_id=T2",
        "plan_revision=3",
        "task_revision=4",
        "attempt=2",
        "iteration=-",
        "model=deepseek-chat",
        "json_mode=true",
    ):
        assert field in start
    for field in (
        "call_id=",
        "status=ok",
        "latency_ms=",
        "input_tokens=3",
        "output_tokens=4",
        "total_tokens=7",
    ):
        assert field in end
    assert start.split("call_id=", 1)[1].split()[0] == end.split(
        "call_id=", 1
    )[1].split()[0]
    assert re.search(r"\bcall_id=[0-9a-f]{32}\b", start)
    assert "secret prompt" not in caplog.text
    assert "secret response" not in caplog.text


def test_invoke_llm_logs_error_without_swallowing_or_leaking(caplog):
    caplog.set_level("INFO", logger="src.llm.observability")
    error = RuntimeError("secret API key and prompt")
    runnable = StubRunnable(error=error)

    with pytest.raises(RuntimeError) as raised:
        invoke_llm(
            runnable,
            ["secret message"],
            node="Verifier",
            purpose="assessment",
        )

    assert raised.value is error
    assert len(caplog.messages) == 2
    assert caplog.messages[0].startswith("LLM_CALL_START ")
    assert caplog.messages[1].startswith("LLM_CALL_ERROR ")
    assert "error_type=RuntimeError" in caplog.messages[1]
    assert "latency_ms=" in caplog.messages[1]
    assert "secret" not in caplog.text


def test_invoke_llm_preserves_omitted_config_call_shape():
    response = object()
    runnable = StubRunnable(response=response)

    assert invoke_llm(
        runnable,
        "input",
        node="Intake",
        purpose="request_parse",
    ) is response
    assert runnable.calls == [("input", {})]


def test_all_phase_one_llm_call_sites_use_the_observability_taxonomy():
    expected = Counter(
        {
            ("Intake", "canonical_intake_generation"): 1,
            ("Planner", "plan_generation"): 1,
            ("Worker", "task_generation"): 3,
            ("Worker", "citation_binding"): 1,
            ("Verifier", "assessment"): 1,
            ("Verifier", "assessment_contract_repair"): 1,
            ("ManualVerifier", "feedback_analysis"): 1,
            ("EvidenceRecovery", "plan_patch"): 1,
            ("Synthesis", "report_synthesis"): 1,
            ("Summarizer", "report_evaluation"): 1,
            ("ConceptGraph", "concept_graph_extraction"): 1,
            ("ChartGenerator", "chart_description"): 1,
            ("ChartGenerator", "chart_description_fallback"): 1,
            ("ChartGenerator", "table_description"): 1,
        }
    )
    actual: Counter[tuple[str, str]] = Counter()

    for source_path in (PROJECT_ROOT / "src").rglob("*.py"):
        if source_path == PROJECT_ROOT / "src" / "llm.py":
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "invoke_llm"
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            node_name = ast.literal_eval(keywords["node"])
            purpose = ast.literal_eval(keywords["purpose"])
            actual[(node_name, purpose)] += 1

    assert sum(actual.values()) == 16
    assert actual == expected


def test_phase_one_clean_path_removes_three_runtime_calls(monkeypatch, caplog):
    caplog.set_level("INFO", logger="src.llm.observability")

    class IntakeModel:
        model_name = "deepseek-chat"

        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "is_chat": False,
                        "user_intent": "生成工艺报告",
                        "task_type": "工程报告",
                        "title": "工艺报告",
                        "doc_length": "500字",
                        "constraints": [],
                        "sections": ["工艺概述"],
                        "core_content": ["温度影响"],
                    },
                    ensure_ascii=False,
                )
            )

    class VerifierModel:
        model_name = "deepseek-chat"

        def invoke(self, _payload, **_kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "status": "PASS",
                        "current_section": "工艺概述",
                        "issues": [],
                        "requirements_met": ["内容和引用符合要求"],
                        "requirements_missing": [],
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr(
        intake_module, "get_llm", lambda *_args, **_kwargs: IntakeModel()
    )
    monkeypatch.setattr(
        verifier_module, "get_llm", lambda *_args, **_kwargs: VerifierModel()
    )
    monkeypatch.setattr(
        planner_module,
        "get_llm",
        lambda *_args, **_kwargs: pytest.fail("guidance must be deterministic"),
    )

    intake_module.llm_parse_user_need("生成一份500字工艺报告", {})
    planner_module._generate_plan_guidance(
        [
            {
                "task_id": "T1",
                "task_name": "工艺概述",
                "task_description": "撰写不超过500字的工艺概述。",
                "tool_requirements": [],
                "use_resources": [],
                "generate_table": False,
                "generate_figure": False,
            }
        ],
        [],
        {},
    )
    worker = AutonomousToolNode.__new__(AutonomousToolNode)
    worker.llm_client = SimpleNamespace(
        invoke=lambda *_args, **_kwargs: pytest.fail(
            "valid citations must bypass citation binding LLM"
        )
    )
    worker._bind_claims_to_evidence(
        {"task_id": "T1", "task_description": "工艺概述"},
        "温度影响熔融指数。[E1]",
        EvidenceBundle(
            records=(
                EvidenceRecord(
                    evidence_id="E1",
                    source_type="rag",
                    title="工艺证据",
                    supporting_text="温度影响熔融指数。",
                    file_path="/srv/docs/process.docx",
                ),
            )
        ),
    )
    verifier_module.verifier(
        {
            "tasks": [
                {
                    "task_id": "T1",
                    "task_name": "工艺概述",
                    "task_description": "撰写不超过500字的工艺概述。",
                    "generate_table": False,
                    "generate_figure": False,
                }
            ],
            "cursor": 0,
            "current_result": {
                "text_output": "温度影响熔融指数。[E1]",
                "citations": [{"evidence_id": "E1"}],
                "tables": [],
                "figures": [],
            },
            "plan_revision": 1,
        },
        {"configurable": {"use_llm": True}},
    )

    starts = [
        message
        for message in caplog.messages
        if message.startswith("LLM_CALL_START ")
    ]
    assert len(starts) == 2
    assert "node=Intake purpose=canonical_intake_generation" in starts[0]
    assert "node=Verifier purpose=assessment" in starts[1]
    assert all("request_refine" not in message for message in starts)
    assert all("plan_guidance" not in message for message in starts)
    assert all("citation_binding" not in message for message in starts)

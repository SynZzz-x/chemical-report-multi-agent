from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from src.llm import extract_token_usage, invoke_llm


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


def test_all_18_llm_call_sites_use_the_observability_taxonomy():
    expected = Counter(
        {
            ("Intake", "request_parse"): 1,
            ("Intake", "request_refine"): 1,
            ("Planner", "plan_generation"): 1,
            ("Planner", "plan_guidance"): 1,
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

    assert sum(actual.values()) == 18
    assert actual == expected

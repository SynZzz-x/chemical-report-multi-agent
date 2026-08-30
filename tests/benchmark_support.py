"""Deterministic fixtures and counters for offline pipeline benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from types import SimpleNamespace
from typing import Any


def serialized_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(len(str(getattr(item, "content", item))) for item in value)
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def serialize_emitted_response(response: Any) -> str:
    """Serialize the complete deterministic fake response, including tool calls."""

    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif hasattr(response, "__dict__"):
        payload = vars(response)
    else:
        payload = response
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def measure_serialized_messages(messages: Any) -> dict[str, int]:
    """Return character-proxy prompt measurements, never provider token counts."""

    return {"serialized_prompt_chars": serialized_chars(messages)}


def measure_template_contributions(
    template: str,
    values: Mapping[str, str],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    """Measure independent template-field contributions as characters."""

    blank = {key: "" for key in values}
    base_chars = len(template.format(**blank))
    metrics = {"base_instructions_chars": base_chars}
    for group_name, fields in groups.items():
        isolated = dict(blank)
        for field in fields:
            isolated[field] = values[field]
        metrics[f"{group_name}_chars"] = len(template.format(**isolated)) - base_chars
    metrics["verifier_prompt_total_chars"] = len(template.format(**values))
    component_total = sum(
        value for key, value in metrics.items() if key != "verifier_prompt_total_chars"
    )
    assert component_total == metrics["verifier_prompt_total_chars"]
    return metrics


@dataclass
class BenchmarkRecorder:
    response: Any
    calls: list[Any] = field(default_factory=list)

    def invoke(self, value: Any, **kwargs: Any) -> Any:
        self.calls.append(value)
        return self.response

    def __call__(self, value: Any, **kwargs: Any) -> Any:
        """Allow a real LangChain prompt sequence to invoke this offline fake."""

        return self.invoke(value, **kwargs)

    @property
    def prompt_chars(self) -> int:
        return sum(serialized_chars(value) for value in self.calls)


SCENARIO_A_TASK = {
    "task_id": "T-A",
    "task_name": "聚乙烯质量异常概述",
    "task_description": "基于已给定上下文撰写聚乙烯质量异常概述。",
}
SCENARIO_A_WORKER_RESPONSE = SimpleNamespace(
    content="聚乙烯质量异常需要结合原料、反应条件和产品指标进行系统排查。" * 8,
    tool_calls=[],
)

SCENARIO_B_QUERY = "聚乙烯 质量异常 排查 建议 常见异常"
SCENARIO_B_TASK = {
    "task_id": "T-B",
    "task_name": "聚乙烯质量异常排查",
    "task_description": "基于知识库证据分析常见聚乙烯质量异常并提出排查建议。",
    "use_rag": True,
    "query": SCENARIO_B_QUERY,
}
SCENARIO_B_PREFETCH_RESPONSE = {
    "success": True,
    "evidence": [
        {
            "title": "聚乙烯质量控制手册",
            "content": "熔融指数、灰分和凝胶含量异常应结合催化剂、氢气和反应温度排查。",
            "source": "/knowledge/polyethylene-quality-control.pdf",
            "section_path": "3.2 常见质量异常",
        }
    ],
}
SCENARIO_B_REPEATED_ADAPTIVE_RESPONSE = SimpleNamespace(
    content="",
    tool_calls=[
        {
            "id": "repeat-prefetch-query",
            "name": "chemical_knowledge_base_tool",
            "args": {"query": SCENARIO_B_QUERY},
        }
    ],
)
SCENARIO_B_FINAL_WORKER_RESPONSE = SimpleNamespace(
    content="应先核对熔融指数、灰分和凝胶含量，再追溯催化剂、氢气与反应温度的变动。" * 8,
    tool_calls=[],
)

PASS_VERIFIER_RESPONSE = {
    "status": "PASS",
    "current_section": "聚乙烯质量异常排查",
    "issues": [],
    "requirements_met": [],
    "requirements_missing": [],
}
FAILED_SEMANTIC_VERIFIER_RESPONSE = {
    "status": "FAILED",
    "current_section": "聚乙烯质量异常排查",
    "issues": [
        {
            "code": "CLAIM_PARTIALLY_SUPPORTED",
            "category": "EVIDENCE_GAP",
            "description": "证据支持部分异常现象，但不足以支持完整因果链。",
            "suggestion": "缩小结论范围或补充直接证据。",
            "severity": "major",
            "requirement_ids": [],
        }
    ],
    "requirements_met": [],
    "requirements_missing": ["完整因果链证据"],
}

VERIFIER_PASS_STATE = {
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

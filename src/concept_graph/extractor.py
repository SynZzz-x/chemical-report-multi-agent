from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.evidence.models import EvidenceBundle
from src.llm import get_llm

from .models import ConceptGraphSpec


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(getattr(value, "content", value) or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("concept graph response must be a JSON object")
    return parsed


class ConceptGraphExtractor:
    def __init__(self, llm=None):
        self.llm = llm or get_llm({"configurable": {"temperature": 0.0}}, json_mode=True)
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "concept_graph_extractor.md"
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    def extract(
        self,
        *,
        title: str,
        evidence: EvidenceBundle,
        required_concepts: list[str] | tuple[str, ...],
        graph_type: str = "causal",
    ) -> ConceptGraphSpec:
        if graph_type != "causal":
            raise ValueError(f"graph type {graph_type!r} is reserved but not implemented")
        payload = {
            "title": title,
            "graph_type": graph_type,
            "required_concepts": list(required_concepts),
            "evidence": [record.model_dump(mode="json") for record in evidence.records],
        }
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
        return ConceptGraphSpec.model_validate(_parse_json_object(response))

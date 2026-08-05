"""Canonical tool identifiers shared by planning and execution."""

from __future__ import annotations

from typing import Any


_SUPPORTED_TOOL_NAMES = {
    "ChemicalKnowledgeBaseTool": "chemical_knowledge_base_tool",
    "chemical_knowledge_base_tool": "chemical_knowledge_base_tool",
    "CSVTool": "csv_analysis_tool",
    "csv_analysis_tool": "csv_analysis_tool",
    "ChartTool": "chart_generator_tool",
    "chart_generator_tool": "chart_generator_tool",
    "SpiderTool": "spider_tool",
    "spider_tool": "spider_tool",
}


def canonical_tool_name(value: Any) -> str | None:
    """Return an exact canonical identifier, or ``None`` for an invalid name."""
    if not isinstance(value, str):
        return None
    identifier = value.strip()
    if not identifier:
        return None
    return _SUPPORTED_TOOL_NAMES.get(identifier, identifier)

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ConceptNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    label: str
    category: str = "concept"
    description: str = ""


class ConceptEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    source: str
    target: str
    relation: str
    polarity: Literal["positive", "negative", "mixed", "unknown"] = "unknown"
    support: Literal["direct", "inferred"] = "direct"
    evidence_ids: tuple[str, ...]
    evidence_quotes: tuple[str, ...]


class ConceptGraphSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    graph_type: Literal["causal", "flowchart", "fault_tree"]
    title: str
    nodes: tuple[ConceptNode, ...]
    edges: tuple[ConceptEdge, ...]
    legend: bool = True


class GraphArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    figure_id: str
    png_path: str
    svg_path: str
    spec_path: str
    dot_path: str
    errors: tuple[str, ...] = ()

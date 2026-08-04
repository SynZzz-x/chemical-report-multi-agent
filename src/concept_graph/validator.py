from __future__ import annotations

from src.evidence.models import EvidenceBundle

from .models import ConceptGraphSpec


def validate_graph(
    spec: ConceptGraphSpec,
    evidence: EvidenceBundle,
    *,
    max_nodes: int = 24,
    max_edges: int = 40,
) -> ConceptGraphSpec:
    if spec.graph_type != "causal":
        raise ValueError(f"graph type {spec.graph_type!r} is reserved but not implemented")
    if not spec.nodes or not spec.edges:
        raise ValueError("concept graph must contain nodes and edges")
    if len(spec.nodes) > max_nodes or len(spec.edges) > max_edges:
        raise ValueError("concept graph exceeds configured size limits")

    node_ids = [node.node_id for node in spec.nodes]
    edge_ids = [edge.edge_id for edge in spec.edges]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate node id")
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("duplicate edge id")
    known_nodes = set(node_ids)
    nodes_by_id = {node.node_id: node for node in spec.nodes}
    evidence_by_id = {record.evidence_id: record for record in evidence.records}
    known_evidence = set(evidence_by_id)
    for node in spec.nodes:
        if "\x00" in node.label:
            raise ValueError(f"invalid node label: {node.node_id}")
    for edge in spec.edges:
        if edge.source not in known_nodes or edge.target not in known_nodes:
            raise ValueError(f"edge {edge.edge_id} refers to an unknown node")
        if not edge.evidence_ids:
            raise ValueError(f"edge {edge.edge_id} has no evidence")
        if not edge.evidence_quotes:
            raise ValueError(f"edge {edge.edge_id} has no evidence quotes")
        missing = [value for value in edge.evidence_ids if value not in known_evidence]
        if missing:
            raise ValueError(f"edge {edge.edge_id} has unknown evidence IDs: {', '.join(missing)}")
        cited_text = "\n".join(
            "\n".join(
                (
                    evidence_by_id[value].title,
                    evidence_by_id[value].locator,
                    evidence_by_id[value].supporting_text,
                )
            )
            for value in edge.evidence_ids
        ).casefold()
        for node_id in (edge.source, edge.target):
            label = nodes_by_id[node_id].label.strip()
            if label and label.casefold() not in cited_text:
                raise ValueError(
                    f"edge {edge.edge_id} cites evidence that does not mention node {label}"
                )
        normalized_cited_text = " ".join(cited_text.split())
        for quote in edge.evidence_quotes:
            normalized_quote = " ".join(quote.casefold().split())
            if not normalized_quote or normalized_quote not in normalized_cited_text:
                raise ValueError(
                    f"edge {edge.edge_id} contains an evidence quote absent from cited evidence"
                )
    return spec

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.concept_graph.models import ConceptEdge, ConceptGraphSpec, ConceptNode
from src.concept_graph.renderer import GraphvizRenderer, build_dot
from src.concept_graph.service import ConceptGraphService
from src.concept_graph.validator import validate_graph
from src.evidence.models import EvidenceBundle, EvidenceRecord


def _bundle():
    return EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="聚乙烯工艺说明",
                supporting_text="反应温度升高会提高熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )


def _spec(evidence_ids=("E1",)):
    return ConceptGraphSpec(
        schema_version=1,
        graph_type="causal",
        title="反应条件与质量指标",
        nodes=(
            ConceptNode(
                node_id="N1",
                label="反应温度",
                category="process_parameter",
            ),
            ConceptNode(
                node_id="N2",
                label="熔融指数",
                category="quality_indicator",
            ),
        ),
        edges=(
            ConceptEdge(
                edge_id="R1",
                source="N1",
                target="N2",
                relation="升高时提高",
                polarity="positive",
                support="direct",
                evidence_ids=evidence_ids,
                evidence_quotes=("反应温度升高会提高熔融指数。",),
            ),
        ),
    )


def test_validator_rejects_uncited_relations():
    with pytest.raises(ValueError, match="evidence"):
        validate_graph(_spec(evidence_ids=()), _bundle())


def test_validator_rejects_unknown_evidence_ids():
    with pytest.raises(ValueError, match="E404"):
        validate_graph(_spec(evidence_ids=("E404",)), _bundle())


def test_validator_rejects_relation_when_cited_text_does_not_mention_both_nodes():
    unrelated = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺说明",
                supporting_text="反应温度影响聚合速率。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    with pytest.raises(ValueError, match="熔融指数"):
        validate_graph(_spec(), unrelated)


def test_dot_contains_relation_direction_and_evidence_marker():
    validated = validate_graph(_spec(), _bundle())

    dot = build_dot(validated)

    assert '"N1" -> "N2"' in dot
    assert "升高时提高 [E1]" in dot
    assert "反应温度" in dot
    assert "熔融指数" in dot


def test_renderer_writes_svg_png_and_graph_spec(tmp_path):
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_bytes(b"rendered")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    renderer = GraphvizRenderer(runner=fake_runner)
    artifacts = renderer.render(_spec(), tmp_path, task_id="T3")

    assert Path(artifacts.png_path).read_bytes() == b"rendered"
    assert Path(artifacts.svg_path).read_bytes() == b"rendered"
    stored_spec = json.loads(Path(artifacts.spec_path).read_text(encoding="utf-8"))
    assert stored_spec["graph_type"] == "causal"
    assert {command[1] for command in commands} == {"-Tpng", "-Tsvg"}


def test_service_retains_validated_spec_when_graphviz_is_unavailable(tmp_path):
    class Extractor:
        def extract(self, **kwargs):
            return _spec()

    def missing_dot(command, **kwargs):
        raise FileNotFoundError("dot")

    settings = SimpleNamespace(
        enabled=True,
        renderer="graphviz",
        max_nodes=24,
        max_edges=40,
        font_family="Noto Sans CJK SC",
    )
    service = ConceptGraphService(
        settings,
        extractor=Extractor(),
        renderer=GraphvizRenderer(runner=missing_dot),
    )

    result = service.generate(
        task_id="T3",
        title="反应条件与质量指标",
        graph_type="causal",
        required_concepts=["反应温度", "熔融指数"],
        evidence=_bundle(),
        output_dir=tmp_path,
    )

    assert result["success"] is False
    assert result["graph_spec"]["edges"][0]["evidence_ids"] == ["E1"]
    assert Path(result["spec_path"]).exists()


def test_renderer_retains_png_when_svg_rendering_fails(tmp_path):
    def partial_runner(command, **kwargs):
        class Result:
            stderr = "svg unavailable"

        result = Result()
        if command[1] == "-Tpng":
            Path(command[command.index("-o") + 1]).write_bytes(b"png")
            result.returncode = 0
        else:
            result.returncode = 1
        return result

    artifacts = GraphvizRenderer(runner=partial_runner).render(
        _spec(), tmp_path, task_id="T3"
    )

    assert Path(artifacts.png_path).read_bytes() == b"png"
    assert artifacts.svg_path == ""
    assert artifacts.errors

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess

from src.config import ConceptGraphSettings
from src.evidence.models import EvidenceBundle

from .extractor import ConceptGraphExtractor
from .renderer import GraphvizRenderer
from .validator import validate_graph


class ConceptGraphService:
    """Deterministic façade around extraction, validation, and rendering."""

    def __init__(
        self,
        settings: ConceptGraphSettings,
        *,
        extractor: ConceptGraphExtractor | None = None,
        renderer: GraphvizRenderer | None = None,
    ):
        self.settings = settings
        self.extractor = extractor
        self.renderer = renderer or GraphvizRenderer(font_family=settings.font_family)

    def generate(
        self,
        *,
        task_id: str,
        title: str,
        graph_type: str,
        required_concepts: list[str],
        evidence: EvidenceBundle,
        output_dir: str | Path,
    ) -> dict:
        if not self.settings.enabled:
            return {"success": False, "error": "concept graph generation is disabled"}
        if self.settings.renderer != "graphviz":
            return {
                "success": False,
                "error": f"unsupported concept graph renderer: {self.settings.renderer}",
            }
        if not evidence.records:
            return {"success": False, "error": "concept graph evidence is unavailable"}
        extractor = self.extractor or ConceptGraphExtractor()
        spec = extractor.extract(
            title=title,
            evidence=evidence,
            required_concepts=required_concepts,
            graph_type=graph_type,
        )
        validated = validate_graph(
            spec,
            evidence,
            max_nodes=self.settings.max_nodes,
            max_edges=self.settings.max_edges,
        )
        evidence_ids = tuple(
            dict.fromkeys(value for edge in validated.edges for value in edge.evidence_ids)
        )
        try:
            artifacts = self.renderer.render(validated, output_dir, task_id=task_id)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            serialized = validated.model_dump_json()
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]
            spec_path = directory / f"concept_{task_id}_{digest}.json"
            spec_path.write_text(
                json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return {
                "success": False,
                "error": str(exc),
                "spec_path": str(spec_path),
                "graph_spec": validated.model_dump(mode="json"),
            }
        if not artifacts.png_path:
            return {
                "success": False,
                "error": "; ".join(artifacts.errors) or "PNG rendering failed",
                "svg_path": artifacts.svg_path,
                "spec_path": artifacts.spec_path,
                "graph_spec": validated.model_dump(mode="json"),
            }
        return {
            "success": True,
            "figure": {
                "figure_id": artifacts.figure_id,
                "path": artifacts.png_path,
                "svg_path": artifacts.svg_path,
                "spec_path": artifacts.spec_path,
                "description": title,
                "graph_type": validated.graph_type,
                "evidence_ids": list(evidence_ids),
            },
            "graph_spec": validated.model_dump(mode="json"),
            "warnings": list(artifacts.errors),
        }

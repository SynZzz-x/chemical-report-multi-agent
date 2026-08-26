from __future__ import annotations

from src.concept_graph.service import ConceptGraphService
from src.config import get_app_config
from src.evidence.models import EvidenceBundle


class ConceptGraphTool:
    """Dedicated post-retrieval tool for evidence-grounded conceptual diagrams."""

    name = "concept_graph_tool"

    def __init__(self, service: ConceptGraphService | None = None):
        settings = get_app_config().concept_graph_settings
        self.service = service or ConceptGraphService(settings)

    def execute(self, task: dict, evidence: EvidenceBundle, output_dir: str) -> dict:
        visualization = task.get("visualization") or {}
        return self.service.generate(
            task_id=str(task.get("task_id") or "task"),
            title=str(visualization.get("title") or task.get("task_name") or "概念关系图"),
            graph_type=str(visualization.get("kind") or "causal"),
            required_concepts=list(visualization.get("required_concepts") or []),
            evidence=evidence,
            output_dir=output_dir,
            job_id=task.get("_job_id"),
            plan_revision=task.get("_plan_revision"),
            task_revision=task.get("_task_revision"),
        )

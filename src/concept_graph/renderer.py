from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import ConceptGraphSpec, GraphArtifacts


RENDERER_VERSION = "1"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def build_dot(spec: ConceptGraphSpec, font_family: str = "Noto Sans CJK SC") -> str:
    lines = [
        "digraph ConceptGraph {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.25", nodesep="0.45", ranksep="0.75"];',
        f'  node [shape="box", style="rounded,filled", fillcolor="#F4F7FB", color="#49657A", fontname="{_escape(font_family)}"];',
        f'  edge [color="#49657A", fontname="{_escape(font_family)}", fontsize="10"];',
        f'  label="{_escape(spec.title)}"; labelloc="t"; fontsize="18"; fontname="{_escape(font_family)}";',
    ]
    colors = {
        "process_parameter": "#E8F2FF",
        "quality_indicator": "#FFF1D6",
        "material": "#E8F8EE",
        "mechanism": "#F4ECFF",
    }
    for node in spec.nodes:
        fill = colors.get(node.category, "#F4F7FB")
        lines.append(
            f'  "{_escape(node.node_id)}" [label="{_escape(node.label)}", fillcolor="{fill}"];'
        )
    for edge in spec.edges:
        evidence = ", ".join(edge.evidence_ids)
        label = f"{edge.relation} [{evidence}]"
        style = "dashed" if edge.support == "inferred" else "solid"
        color = {"positive": "#197642", "negative": "#B33A3A", "mixed": "#8A5A00"}.get(
            edge.polarity, "#49657A"
        )
        lines.append(
            f'  "{_escape(edge.source)}" -> "{_escape(edge.target)}" '
            f'[label="{_escape(label)}", style="{style}", color="{color}"];'
        )
    lines.append("}")
    return "\n".join(lines)


class GraphvizRenderer:
    def __init__(
        self,
        *,
        dot_binary: str = "dot",
        font_family: str = "Noto Sans CJK SC",
        runner: Callable[..., Any] = subprocess.run,
    ):
        self.dot_binary = dot_binary
        self.font_family = font_family
        self.runner = runner

    def render(self, spec: ConceptGraphSpec, output_dir: str | Path, *, task_id: str) -> GraphArtifacts:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        serialized = spec.model_dump_json()
        digest = hashlib.sha256(
            f"{RENDERER_VERSION}:{self.font_family}:{serialized}".encode("utf-8")
        ).hexdigest()[:10]
        figure_id = f"concept_{task_id}_{digest}"
        base = directory / figure_id
        dot_path = base.with_suffix(".dot")
        png_path = base.with_suffix(".png")
        svg_path = base.with_suffix(".svg")
        spec_path = base.with_suffix(".json")
        dot_path.write_text(build_dot(spec, self.font_family), encoding="utf-8")
        spec_path.write_text(
            json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        errors: list[str] = []
        for output_format, output_path in (("png", png_path), ("svg", svg_path)):
            try:
                result = self.runner(
                    [self.dot_binary, f"-T{output_format}", str(dot_path), "-o", str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    errors.append(
                        f"Graphviz {output_format} rendering failed: {result.stderr}"
                    )
                elif not output_path.exists():
                    errors.append(f"Graphviz did not create {output_path}")
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"Graphviz {output_format} rendering failed: {exc}")
        return GraphArtifacts(
            figure_id=figure_id,
            png_path=str(png_path) if png_path.exists() else "",
            svg_path=str(svg_path) if svg_path.exists() else "",
            spec_path=str(spec_path),
            dot_path=str(dot_path),
            errors=tuple(errors),
        )

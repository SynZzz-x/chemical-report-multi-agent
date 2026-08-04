# Evidence-Grounded Concept Graph Design

## Status

Approved architecture; detailed design pending implementation planning.

## Problem

The current Worker can retrieve textual evidence and can generate charts from
CSV or Excel data, but it cannot produce evidence-grounded conceptual diagrams
from technical documents. This creates a contract mismatch: the Planner may
request a causal diagram or fault tree while the available `ChartTool` only
accepts tabular datasets.

The current Worker boundary also exposes RAG maintenance operations to the
model, initializes tools that a task will not use, and does not reliably carry
retrieved source metadata into `TaskResult.citations`. A concept-graph feature
must correct these boundaries instead of adding more responsibilities to the
existing chart generator.

## Goals

- Add a general concept-graph capability with an initial `causal` graph type.
- Reserve the schema for later `flowchart` and `fault_tree` renderers.
- Prefer local RAG evidence and use public web evidence only when RAG coverage
  is insufficient.
- Require every rendered relation to be traceable to one or more evidence
  records.
- Generate both SVG and PNG artifacts and embed them in Markdown, DOCX, and
  PDF reports through the existing `TaskResult.figures` path.
- Keep retrieval, evidence assessment, graph construction, and rendering as
  separate components with testable interfaces.
- Fail closed when evidence is unavailable or insufficient; unsupported
  relations must not appear in a formal report figure.

## Non-Goals

- Generating free-form illustrations or photorealistic images.
- Converting qualitative evidence into invented quantitative heatmap values.
- Letting the concept-graph renderer search the web or mutate the RAG index.
- Replacing the existing CSV/Excel `ChartTool`.
- Supporting interactive browser-only diagrams in the first release.
- Implementing `flowchart` and `fault_tree` rendering in the first release.

## Architectural Decision

Use a structured evidence pipeline and a deterministic Graphviz renderer.
DeepSeek may extract a graph specification from supplied evidence, but it does
not draw the image directly and cannot introduce uncited relations.

The pipeline is:

```text
Planner
  -> EvidenceCoordinator
       -> ChemicalKnowledgeBaseTool (RAG, query only)
       -> EvidenceCoverageChecker
       -> WebEvidenceTool (conditional fallback)
       -> EvidenceNormalizer
  -> ConceptGraphBuilder
       -> RelationExtractor
       -> EvidenceValidator
       -> GraphvizRenderer
  -> TaskResult(figures, citations, graph_spec)
  -> Markdown / DOCX / PDF
```

This design deliberately keeps `ConceptGraphTool` focused on graph
construction. Evidence acquisition remains an upstream responsibility.

## Component Boundaries

### Planner contract

The Planner adds an explicit visualization request instead of relying only on
phrases embedded in `task_description`:

```json
{
  "generate_figure": true,
  "visualization": {
    "kind": "concept_graph",
    "graph_type": "causal",
    "title": "反应条件对聚乙烯质量指标的影响",
    "required_concepts": ["反应温度", "反应压力", "熔融指数", "密度"],
    "allow_web_fallback": true
  }
}
```

`generate_figure=true` alone is not sufficient to choose a tool. The
`visualization.kind` field distinguishes data charts from conceptual diagrams:

- `data_chart` routes to the existing `ChartTool` and requires CSV/Excel data.
- `concept_graph` routes to the new concept-graph pipeline and requires
  evidence records.

If the Planner requests a data chart without data, validation converts the
request to a concept graph only when the requested content is relational and
the user did not explicitly require a quantitative chart. Otherwise the task
is marked unsupported and returned for replan.

### RAG retrieval boundary

The Worker-facing `ChemicalKnowledgeBaseTool` exposes `query` only. RAG
maintenance remains available exclusively through operator APIs and the CLI:

- `python -m src.rag.cli rebuild /srv/chemical-docs/process-guide.pdf`
- `python -m src.rag.cli ingest /srv/chemical-docs/new-standard.docx`

The model cannot call `load`, `ingest`, `rebuild`, or `stats`. This prevents a
report task from mutating the active knowledge base or wasting tool-loop
iterations on maintenance operations.

### Web evidence boundary

`WebEvidenceTool` retrieves public material only after the coverage checker
requests fallback. It replaces the concept pipeline's use of the legacy
general-purpose spider; the legacy `SpiderTool` may remain temporarily for
unrelated browsing tasks.

Preferred source classes, in order, are:

1. government and regulator publications;
2. standards organizations;
3. peer-reviewed or institutional publications;
4. equipment, catalyst, or material supplier technical documentation;
5. other attributable technical sources.

Anonymous pages, search-result snippets, scraped aggregators, and content with
no stable URL are not valid support for a graph relation. Web evidence stores a
short supporting excerpt rather than a full copied page.

### Evidence coordinator

`EvidenceCoordinator` owns retrieval order and query budgets. It is not an LLM
agent. It receives a task and returns an immutable `EvidenceBundle`.

Default collection sequence:

1. Build up to three focused RAG queries from the task query, required
   concepts, and requested graph title.
2. Deduplicate normalized queries before execution.
3. Normalize RAG results and run coverage assessment.
4. If coverage is insufficient and `allow_web_fallback=true`, issue focused
   web queries only for uncovered concepts or relations.
5. Normalize, deduplicate, and rank the combined evidence.
6. Return the evidence bundle and a coverage report.

The coordinator enforces a default maximum of three RAG queries and three web
queries per task. These values are configuration settings, not LLM choices.

### Lazy tool initialization

Tool requirements are resolved before tool classes are instantiated. A
RAG-only task must not initialize CSV, chart, or spider dependencies. This
removes irrelevant startup errors such as a missing web-scraping dependency
during a knowledge-base-only task.

## Evidence Model

All local and web evidence is converted to one record shape:

```json
{
  "evidence_id": "E1",
  "source_type": "rag",
  "title": "聚乙烯生产工艺与质量控制概述",
  "locator": "第 4.2 节",
  "file_path": "/knowledge/polyethylene.docx",
  "url": null,
  "published_at": null,
  "accessed_at": null,
  "pages": {"start": 8, "end": 9},
  "chunk_ids": ["child_ab12cd34"],
  "supporting_text": "反应温度变化会影响聚合物分子量及熔融指数。",
  "retrieval_query": "反应温度 分子量 熔融指数",
  "source_quality": "internal_technical_document"
}
```

For web evidence, `url` and `accessed_at` are required. For RAG evidence,
`file_path` or another stable internal document identifier is required.
Evidence IDs are stable within one report job and are the only identifiers
shown inside a diagram.

Evidence deduplication uses normalized source identity plus locator and content
hash. Repeated retrieval of the same chunk or web passage must not create a new
evidence record.

## Coverage Assessment

Web fallback is conditional and deterministic. RAG evidence is insufficient
when any of the following is true:

- retrieval is unavailable or returned no evidence;
- one or more required concepts have no supporting evidence;
- the requested graph cannot form at least one supported relation;
- the relation extractor reports an unsupported required relation topic;
- all candidate evidence is duplicated or lacks a stable source locator.

Coverage output is explicit:

```json
{
  "status": "sufficient",
  "covered_concepts": ["反应温度", "熔融指数"],
  "uncovered_concepts": [],
  "supported_relation_count": 3,
  "web_fallback_required": false,
  "warnings": []
}
```

The checker does not require two sources for every relation. A single direct,
authoritative source may be sufficient. Multiple sources improve provenance
but must not be merged as if they were independent when they repeat the same
underlying publication.

If RAG and web evidence conflict, the graph may include neither relation until
the conflict is resolved, or it may render an explicit `conflicting` edge when
the task asks for comparison. The report must describe the conflict and cite
both sources.

## Concept Graph Schema

The extractor returns strict JSON validated by Pydantic before rendering:

```json
{
  "schema_version": 1,
  "graph_type": "causal",
  "title": "反应条件对聚乙烯质量指标的影响",
  "nodes": [
    {
      "node_id": "N1",
      "label": "反应温度",
      "category": "process_parameter",
      "description": "聚合反应温度"
    }
  ],
  "edges": [
    {
      "edge_id": "R1",
      "source": "N1",
      "target": "N2",
      "relation": "升高时通常降低",
      "polarity": "negative",
      "support": "direct",
      "evidence_ids": ["E1"]
    }
  ],
  "legend": true
}
```

Reserved graph types are `causal`, `flowchart`, and `fault_tree`. The first
release accepts `causal` only and rejects the other values with a structured
unsupported-type result.

Allowed causal polarities are `positive`, `negative`, `conditional`,
`conflicting`, and `unspecified`. Numerical strength is forbidden unless an
evidence record contains a directly cited quantitative value. Qualitative
language must not be converted into fabricated weights.

`support=direct` means the source explicitly states the relation.
`support=inferred` means the relation is a transparent synthesis of cited
facts. Inferred edges use a distinct dashed style and are identified as
inferred in the report legend.

## Validation Rules

Before rendering, `EvidenceValidator` enforces:

- all node IDs and edge IDs are unique;
- every edge references existing nodes;
- every edge has at least one valid evidence ID;
- every evidence ID exists in the current immutable evidence bundle;
- an edge cannot claim `direct` support when its evidence assessment is
  inferred;
- unsupported quantitative labels are rejected;
- empty graphs and disconnected single-node outputs are rejected;
- graph size stays within configured node and edge limits;
- graph title and labels contain no HTML, script, or Graphviz control syntax.

Validation failure does not fall back to an uncited image. The tool returns a
structured failure that the Worker and Verifier can report or replan.

## Rendering

Graphviz is the rendering backend. The renderer accepts only a validated graph
specification and never receives raw model output.

For causal graphs:

- process parameters, quality indicators, materials, and outcomes use distinct
  node styles;
- positive and negative relations use different colors and arrow labels;
- conditional relations use a dotted edge;
- inferred relations use a dashed edge;
- evidence IDs such as `[E1]` appear in the edge label;
- a compact legend is included when more than one relation style is present;
- Chinese fonts are selected from configurable server-safe font families.

Each render produces:

```text
cache/users/<user>/conversations/<conversation>/jobs/<job>/charts/
  concept_<task_id>_<hash>.svg
  concept_<task_id>_<hash>.png
  concept_<task_id>_<hash>.json
```

The JSON file is the validated graph specification and supports later
re-rendering without another LLM call. The content hash includes the graph
specification, renderer version, and style version so identical graphs can be
reused safely.

## Worker Integration

Concept graph generation is an explicit phase after evidence collection and
before the task result is finalized. It is not left to arbitrary tool-call
order.

The Worker sequence for a concept-graph task is:

1. collect and normalize evidence;
2. assess coverage and optionally add web evidence;
3. provide the same immutable evidence bundle to report-text generation;
4. extract and validate the graph specification;
5. render SVG and PNG;
6. create one `TaskResult` containing text, figures, citations, graph spec, and
   coverage diagnostics.

`TaskResult` gains:

```json
{
  "figures": [
    {
      "figure_id": "concept_T3_ab12cd34",
      "path": "cache/users/local-user/conversations/conv_1/jobs/job_1/charts/concept_T3_ab12cd34.png",
      "svg_path": "cache/users/local-user/conversations/conv_1/jobs/job_1/charts/concept_T3_ab12cd34.svg",
      "description": "反应条件对聚乙烯质量指标的影响关系图",
      "graph_type": "causal",
      "evidence_ids": ["E1", "E2"]
    }
  ],
  "citations": [],
  "graph_spec": {},
  "evidence_coverage": {}
}
```

`sources_used` is derived from the citation records, not from only
`task.use_resources`. This ensures prebuilt RAG sources and web sources survive
into the final report.

## Report Integration

The existing figure path remains the integration boundary:

- Markdown embeds the PNG and writes a caption followed by an evidence table.
- DOCX embeds the PNG at a bounded width and writes the caption and source
  list immediately below it.
- PDF uses the PNG for reliable font and layout behavior.
- SVG is retained as a downloadable high-resolution artifact and for future
  renderers.

The evidence table maps `[E1]` identifiers to document titles and locators or
to web titles and URLs. The Summarizer may improve surrounding prose but must
not remove evidence identifiers, alter graph claims, or invent additional
sources.

## Failure and Degraded Modes

- RAG unavailable, web allowed: use web evidence and state that the graph has
  no internal-source support.
- RAG insufficient, web unavailable or disabled: omit the graph and return an
  explicit evidence-insufficient result.
- Web retrieval failure after partial RAG coverage: render only relations fully
  supported by RAG when the remaining graph is still meaningful; otherwise
  omit it.
- Relation extraction failure: retry once with validation errors, then fail the
  graph phase without fabricating an image.
- Graphviz unavailable: retain the validated JSON spec, mark rendering failed,
  and do not claim that a figure was generated.
- One output format fails: retain successful artifacts and report the failed
  format explicitly.

No failure mode silently substitutes an unrelated chart or an LLM-generated
bitmap.

## Configuration

Configuration remains centralized in `src/config.py`. Proposed settings are:

```text
CONCEPT_GRAPH_ENABLED=true
CONCEPT_GRAPH_RENDERER=graphviz
CONCEPT_GRAPH_MAX_NODES=24
CONCEPT_GRAPH_MAX_EDGES=40
CONCEPT_GRAPH_FONT_FAMILY=Noto Sans CJK SC
EVIDENCE_RAG_MAX_QUERIES=3
EVIDENCE_WEB_MAX_QUERIES=3
EVIDENCE_WEB_FALLBACK=true
EVIDENCE_WEB_ALLOWED_SOURCE_CLASSES=government,standards,academic,vendor
```

Secrets for web providers remain environment variables represented by typed
configuration fields. No API key or provider URL is embedded in prompts or
tool implementations.

## Proposed Module Layout

```text
src/
  evidence/
    models.py                 # EvidenceRecord, EvidenceBundle, CoverageReport
    coordinator.py            # RAG-first orchestration and budgets
    coverage.py               # deterministic fallback decision
    normalizer.py             # RAG/web normalization and deduplication
    web.py                    # public web evidence adapter
  concept_graph/
    models.py                 # ConceptGraphSpec, nodes, edges
    extractor.py              # DeepSeek structured relation extraction
    validator.py              # provenance and graph validation
    renderer.py               # Graphviz SVG/PNG renderer
    service.py                # concept-graph pipeline façade
  nodes/worker/
    agent/graph.py            # integration only; no rendering implementation
```

This keeps the already large Worker module from accumulating another complete
subsystem.

## Testing Strategy

### Deterministic unit tests

- RAG evidence normalization and duplicate removal.
- Coverage decisions with complete, incomplete, unavailable, and conflicting
  evidence.
- Web fallback is not called when RAG is sufficient.
- Query budgets and duplicate-query suppression.
- Every accepted edge has valid evidence.
- Unsupported numerical strength is rejected.
- Graphviz escaping and Chinese labels.
- Stable artifact hashes and cache reuse.
- Citation and figure propagation into `TaskResult`.
- Lazy tool initialization does not import unused dependencies.
- Worker-facing RAG schema exposes query only.

### Integration tests

- RAG-only causal graph generation.
- RAG-insufficient then web-fallback graph generation.
- Conflicting RAG and web evidence.
- Evidence-insufficient failure with no generated image.
- PNG/SVG artifacts and graph JSON are written to the job directory.
- Markdown, DOCX, and PDF include the image and evidence table.
- SQLite checkpoint restoration does not regenerate an unchanged graph.

LLM and network behavior are represented by deterministic fixtures in the
default test suite. Optional live-provider tests are separate and opt-in.

## Delivery Phases

### Phase 1: Evidence and Worker boundaries

- Make Worker RAG query-only.
- Add lazy tool initialization.
- Introduce evidence records, normalization, coverage checking, and citation
  propagation.
- Add conditional public-web evidence retrieval.

### Phase 2: Causal graph generation

- Add strict causal graph schema and relation extraction.
- Add evidence validation and Graphviz rendering.
- Attach graph artifacts and citations to `TaskResult`.
- Embed images and evidence tables in all report formats.

### Phase 3: Additional graph types

- Add `flowchart` renderer and validation rules.
- Add `fault_tree` renderer and validation rules.
- Preserve the same evidence and report-integration contracts.

## Acceptance Criteria

- A RAG-supported causal task generates valid PNG, SVG, and graph JSON files.
- The resulting figure appears in Markdown, DOCX, and PDF outputs.
- Every rendered edge maps to one or more citations in the report.
- Web retrieval is skipped when RAG coverage is sufficient.
- When RAG coverage is insufficient, only uncovered concepts trigger web
  retrieval.
- Unsupported relations are omitted rather than invented.
- The Worker cannot invoke RAG ingestion or statistics operations.
- An unused spider or chart dependency cannot fail a RAG-only task.
- Conceptual diagrams do not require CSV or Excel resources.
- Existing data-chart behavior remains unchanged.

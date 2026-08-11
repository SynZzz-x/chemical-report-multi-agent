# Planner Contract Stabilization Design

## 1. Goal

Stabilize the existing Planner boundary before changing Worker, Verifier,
Summarizer, report state, evidence orchestration, or rendering. A Planner model
response may enter `state["tasks"]` only when its envelope and every task satisfy
the declared contract. Invalid output must never be replaced by executable
placeholder tasks.

This design is a focused first increment of the existing report-pipeline
stabilization design. It preserves the serial workflow, user-confirmed initial
plans, user-only full replanning, SQLite checkpoint compatibility, current web
authorization rules, and the existing Worker task representation.

## 2. Confirmed Failure

Planner currently enables DeepSeek JSON-object mode but asks three generation
paths to return a top-level JSON array. Initial-plan parsing rejects a non-list
response as `bad tasks`, then silently builds tasks whose descriptions contain
only `围绕 <section> 生成占位内容`. Those fallback tasks omit the report subject,
core content, evidence requirements, RAG query, web policy, and output
constraints, so they can reach Worker without any tools or adequate grounding.

The same array/object mismatch exists in:

- initial generation in `_build_tasks_with_llm`;
- user-requested full replan in `_build_tasks_from_replan_feedback`;
- plan-confirmation refinement in `_refine_tasks`.

The prompt-loading fallback also still requests a JSON array, and prompt
examples contain JavaScript-style comments that are invalid JSON.

## 3. Scope

### In scope

- one JSON-object envelope for all three Planner generation paths;
- strict model-payload validation before resource-path normalization;
- one bounded format-repair attempt after invalid model output;
- structured initial-plan failure and explicit retry/cancel behavior;
- removal of executable placeholder-plan fallback;
- complete Intake context propagation into all Planner prompts;
- correction of the Intake `output_format` field mapping;
- conditional planning rules for RAG, data analysis, visualization, and web use;
- focused contract, retry, routing, and compatibility tests.

### Out of scope

- Worker prompt or tool-loop changes;
- Verifier or DecisionPolicy changes;
- Summarizer and renderer changes;
- `report_spec` or new State architecture;
- TaskController, Artifact, or ReviewRecord implementation;
- changing the global default of `get_llm`;
- deleting or migrating existing SQLite databases.

## 4. Input Contract

Planner consumes the latest `INTAKE_SUMMARY` object. Initial, full-replan, and
confirmation-refinement prompts receive the following immutable request context:

```json
{
  "title": "string",
  "user_intent": "string",
  "task_type": "string",
  "doc_length": "string or number",
  "constraints": ["string"],
  "sections": ["string"],
  "core_content": ["string"],
  "style": "string or null",
  "output_format": "string or null",
  "resources": ["normalized resource descriptors"]
}
```

`build_task_spec` reads `output_format` first and accepts legacy `format` only
as a compatibility fallback. Full replan and refinement must use the original
Intake context rather than reconstructing scope from the current task list.

## 5. Output Contract

Every Planner model call uses `get_llm(config, json_mode=True)` and requests one
JSON object whose only top-level field is `tasks`:

```json
{
  "tasks": [
    {
      "task_id": "T1",
      "task_name": "章节名称",
      "task_description": "完整执行要求",
      "task_type": "analysis",
      "use_rag": true,
      "use_web": false,
      "query": "检索查询",
      "use_resources": [],
      "generate_figure": false,
      "generate_table": false,
      "visualization": null
    }
  ]
}
```

The envelope rejects extra top-level fields, a bare array, missing `tasks`, an
empty task list, and more than `MAX_PLAN_TASKS` tasks. The normal target remains
six to ten tasks, but code does not impose a minimum of six because explicit
short report outlines are valid.

Each newly generated task requires all fields shown above. Validation enforces:

- non-empty `task_id`, `task_name`, and `task_description` strings;
- unique task IDs and the existing stable-ID normalization rules;
- `task_type` in `analysis`, `summary`, or `inference`;
- actual booleans for RAG, web, table, and figure flags;
- a string query and a list of non-empty resource names;
- `visualization` as either `null` or an object satisfying existing nested
  visualization rules;
- no unsupported task fields.

Strict model-payload validation is kept separate from legacy-checkpoint
normalization so older persisted tasks are not rejected merely for omitting a
newly required model-output field.

## 6. Semantic Planning Rules

Schema validation checks shape and types. A separate deterministic semantic
validator checks cross-field meaning:

- a task explicitly requiring knowledge-base evidence or traceable citations
  must set `use_rag=true` and provide a non-empty query;
- a task with `use_rag=false` must not claim that RAG evidence is mandatory;
- statistical correlation, regression, time-series, heatmap, or numeric-window
  work may be planned only when an assigned data resource or explicit user data
  source exists;
- conceptual relationship diagrams may be planned without CSV data, using the
  existing `visualization` contract;
- web fields remain false unless current job authorization and task intent both
  permit web access;
- unsupported quantitative results must not be requested when no real data
  source exists.

These rules are based on explicit task requirements and available resources,
not a broad heuristic that treats every professional chapter as a RAG task.

## 7. Generation and Repair Flow

All three generation paths use one shared invocation/parsing routine:

```text
build prompt from complete request context
  -> invoke once in JSON-object mode
  -> parse envelope, task schema, and semantic rules
  -> valid: normalize resource paths and stable IDs
  -> invalid: invoke one format-repair attempt with concise validation errors
  -> valid: normalize and return
  -> invalid: return a typed Planner generation failure
```

The repair attempt is not a business replan and does not increment plan
revision. It receives the expected schema and concise validation errors; it does
not receive unrestricted execution instructions from the invalid response.
Errors log the generation path, attempt number, exception category, and a
bounded response snippet without credentials.

Prompt read failures are typed Planner failures. There is no generic prompt
fallback and no placeholder task fallback.

## 8. Initial-Plan Error State

Exhausted initial generation fails closed without crashing the whole job:

```text
Planner
  -> planner_action = INITIAL_PLAN_ERROR
  -> tasks remain empty/unmodified
  -> PlannerConfirm interrupt
       -> RETRY_INITIAL_PLAN -> Planner
       -> CANCEL -> Exit
```

Retry uses the original `INTAKE_SUMMARY` and does not accept a partial plan.
Cancellation returns an explicit unsuccessful/cancelled final result. Neither
branch may route to Worker with an empty or invalid task list.

Full-replan failure continues preserving the active plan and using the existing
`FULL_REPLAN_ERROR` confirmation path. Refinement failure preserves the last
confirmed candidate and requests user action; it never silently fabricates a
replacement task.

## 9. Prompt Changes

The three Planner prompts must:

- request the same JSON-object envelope;
- contain only valid JSON examples with no comments or Markdown wrappers;
- include `core_content`, `style`, and `output_format`;
- distinguish evidence-grounded professional writing from real-data analysis;
- prohibit invented statistics when no data resource is assigned;
- make figures conditional on user requirements, evidence, and available data;
- preserve original title, intent, core content, and constraints during replan
  and refinement.

## 10. Files

- `src/nodes/planner.py`: shared parser, validation, repair attempt, and typed
  failure handling for all three generation paths.
- `src/nodes/intake.py`: correct `output_format` propagation with legacy
  compatibility.
- `src/graph.py`: route initial-plan retry back to Planner and cancellation to
  Exit; never route an initial-plan error to Worker.
- `src/prompts/planner_to_worker.md`: initial-plan object contract and
  conditional task-design rules.
- `src/prompts/planner_replan.md`: synchronized full-replan contract and full
  original context.
- `src/prompts/planner_intake_replan.md`: synchronized refinement contract and
  scope-preservation rules.
- `tests/test_planner_contract.py`: new focused contract and failure tests.
- Existing Planner/recovery/graph tests: compatibility updates only where the
  intentional contract or route changes.

## 11. Verification

Tests must prove:

1. `{"tasks": [...]}` is accepted in all three generation paths.
2. A bare array, extra envelope field, missing/empty tasks, malformed fields,
   duplicate IDs, and oversized plans are rejected.
3. One invalid response followed by one valid repair response succeeds.
4. Two invalid responses produce a typed failure and no placeholder tasks.
5. Initial-plan exhaustion cannot reach Worker and exposes retry/cancel choices.
6. Retry regenerates from the original Intake context; cancel reaches Exit.
7. `core_content`, `style`, and `output_format` reach every Planner prompt.
8. Intake propagates `output_format` and accepts legacy `format`.
9. Explicit knowledge-base requirements enforce RAG plus a query.
10. Data-analysis methods requiring real data are rejected without an assigned
    data source, while evidence-backed qualitative work remains valid.
11. Full-replan failure preserves the active plan.
12. Legacy checkpoint tasks still restore through compatibility normalization.
13. Both manual and automatic graphs compile with the new routes.

## 12. Acceptance Criteria

- Planner model output has one contract across initial, full-replan, and refine
  paths.
- No Planner error creates executable placeholder content.
- No invalid or empty plan can enter Worker.
- Planner failure is visible and recoverable without resetting accepted work.
- Planner tasks preserve the original report scope and evidence/data policy.
- Existing SQLite checkpoint files remain usable.

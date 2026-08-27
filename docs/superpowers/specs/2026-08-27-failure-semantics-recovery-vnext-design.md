# Failure Semantics / Recovery vNext Design

**Date:** 2026-08-27
**Status:** Approved direction, pending written-spec review

## 1. Goal and scope

Refactor failure semantics and recovery policy without replacing the existing
LangGraph execution architecture. The workflow keeps the current Intake,
Planner, Worker, Verifier, DecisionPolicy, recovery, Summarizer, and Exit
boundaries. New graph nodes are not introduced unless an existing conditional
edge cannot express the required behavior.

The target behavior is:

```text
Verifier assessment
        ↓
DecisionPolicy classification
        ├── RETRYABLE_EXECUTION → bounded local retry → Verifier
        ├── REPAIRABLE_CONTRACT → bounded repair → Verifier
        ├── DEGRADABLE_QUALITY → CommitWithWarning → next runnable task
        ├── USER_DECISION_REQUIRED → register blocker → next runnable task
        └── FATAL_SYSTEM → fail job with diagnostic

all runnable tasks processed
        ↓
deterministic report admission
        ├── no unresolved blockers → Summarizer
        └── unresolved blockers → one consolidated NeedsUserInput
```

This work does not redesign persistence, RAG, rendering, or the Worker tool
loop. Existing deterministic citation/source validation, scoped ConceptGraph
and length-rewrite budgets, LLM observability, and blocker radio controls must
not regress.

## 2. Phase 0 findings

The automatic graph currently routes `Worker/Synthesis → Verifier →
DecisionPolicy`. DecisionPolicy can route to Worker, EvidenceRecovery,
AssetRecovery, PlanPatcher, NeedsUserInput, Verifier, Planner, Summarizer, or
Synthesis.

The existing taxonomy is execution-oriented but incomplete:

- `CONTENT_DEFECT`
- `EVIDENCE_GAP`
- `LOCAL_PLAN_DEFECT`
- `EXTERNAL_BLOCKER`
- `VERIFIER_FAILURE`

It does not distinguish retryability, contract repair, safe degradation, user
contract changes, and fatal system failures. As a result, ordinary evidence
gaps, optional asset failures, length-repair exhaustion, verifier service
failure, and general content retry exhaustion can all reach NeedsUserInput.

Current budgets are already bounded: content 2, asset 1, evidence 1, verifier
1, plan patch 1 per task and 3 per job, ConceptGraph semantic extraction 1 per
task/revision, and length rewrite 1 per task/revision. Recovery vNext preserves
or narrows these bounds.

The Intake contract stores constraints as strings. Planner tasks have no
stable requirement IDs or severity. State has one `pending_user_action`, so a
serial task blocker interrupts immediately and prevents blocker consolidation.

Baseline before behavioral changes: `679 passed`.

## 3. Chosen approach

Use a compatibility-first domain layer over the existing graph.

### Rejected alternatives

Extending `IssueCategory` in place would minimize the diff but retain mixed
assessment and workflow semantics. A scheduler/new-node rewrite would provide
cleaner topology but exceeds the requested scope and risks persistence and
resume regressions.

### Domain boundary

A focused failure-semantics module owns:

```python
class FailureClass(str, Enum):
    RETRYABLE_EXECUTION = "RETRYABLE_EXECUTION"
    REPAIRABLE_CONTRACT = "REPAIRABLE_CONTRACT"
    DEGRADABLE_QUALITY = "DEGRADABLE_QUALITY"
    USER_DECISION_REQUIRED = "USER_DECISION_REQUIRED"
    FATAL_SYSTEM = "FATAL_SYSTEM"


class FailureAction(str, Enum):
    COMMIT = "COMMIT"
    RETRY_VERIFIER = "RETRY_VERIFIER"
    RETRY_TASK = "RETRY_TASK"
    REPAIR_CONTRACT = "REPAIR_CONTRACT"
    RECOVER_EVIDENCE = "RECOVER_EVIDENCE"
    RECOVER_ASSET = "RECOVER_ASSET"
    PATCH_PLAN = "PATCH_PLAN"
    COMMIT_WITH_WARNING = "COMMIT_WITH_WARNING"
    REGISTER_BLOCKER = "REGISTER_BLOCKER"
    FAIL_JOB = "FAIL_JOB"
```

`FailureAction` is the policy-level action vocabulary. Existing
`WorkflowAction` values remain graph routing adapters until all consumers have
migrated; recovery nodes must not invent additional actions.

`FailureDecision` is serialized to a plain dictionary in LangGraph State with
this exact shape:

```python
class FailureDecision(TypedDict):
    failure_class: str          # FailureClass value
    subtype: str                # stable uppercase subtype
    reason: str                 # bounded, user-safe explanation
    task_id: str | None
    action: str                 # FailureAction value
    retryable: bool
    repair_attempt: int
    repair_budget: int
    user_blocker: bool
    requirement_ids: list[str]
    hard_requirement_ids: list[str]
    metadata: dict[str, JSONValue]
```

Verifier continues to judge whether a result satisfies its task. It does not
choose recovery actions or user interruption. DecisionPolicy is the sole owner
of failure classification and workflow action.

The old `IssueCategory` remains as an assessment compatibility input during
migration. It is not the authority for new workflow decisions.

The allowed action matrix is closed:

| Decision | Allowed `FailureAction` |
|---|---|
| successful assessment | `COMMIT` |
| `RETRYABLE_EXECUTION` | `RETRY_VERIFIER`, `RETRY_TASK`, `RECOVER_EVIDENCE` |
| `REPAIRABLE_CONTRACT` | `REPAIR_CONTRACT`, `RECOVER_ASSET`, `PATCH_PLAN` |
| `DEGRADABLE_QUALITY` | `COMMIT_WITH_WARNING` |
| `USER_DECISION_REQUIRED` | `REGISTER_BLOCKER` |
| `FATAL_SYSTEM` | `FAIL_JOB` |

Any other class/action pair is a State contract error. `PATCH_PLAN` is valid
only for the plan-defect subtypes listed in Section 6.

## 4. Requirement contract

### Canonical requirements

Intake creates a serializable requirement registry. Each record contains:

```python
{
    "requirement_id": "REQ-001",
    "text": "必须使用用户提供的质量控制规程",
    "severity": "hard",
    "kind": "resource",
    "status": "active",
    "contract_revision": 1,
    "provenance": {
        "origin": "explicit_user",
        "source_message_id": "msg-123",
        "source_field": "constraints",
        "source_index": 0,
        "derivation": "deterministic_explicit_marker",
    },
}
```

IDs are assigned once from canonical Intake ordering and stored in State. They
are not recomputed by comparing free text during recovery. Full replanning
preserves the registry and IDs.

Explicit non-negotiable language such as `必须`, `不得`, `不可`, `任何缺口不可接受`,
and explicit required user files creates hard requirements. Ordinary quality
goals, recommended length, nonessential visualizations, and inferred planner
preferences default to soft. An explicit maximum or minimum is hard only when
the user frames it as mandatory; otherwise it remains a soft target.

`provenance.origin` is one of `explicit_user`, `intake_inferred`,
`legacy_reconstructed`, or `legacy_task_contract`. Planner is not a permitted
requirement provenance because it may link requirements but may not invent
them. `source_message_id`, `source_field`, and `source_index` identify the
authoritative input without relying on later free-text matching. The
`derivation` value records the deterministic rule used to assign severity.

Planner model tasks gain a required `requirement_ids` list. Validation rejects
unknown IDs. Synthesis tasks may reference report-wide IDs. Legacy tasks with
no field link only to requirements that can be recovered from explicit
structural task fields; otherwise they normalize to an empty list. Missing
linkage never authorizes issue-prose matching or invention of a hard
requirement.

Verifier issues may include `requirement_ids`. Deterministic checks attach IDs
from the active task. DecisionPolicy resolves severity through the registry,
not issue prose.

### Minimum task dependency contract

Requirement linkage and task dependency are separate contracts. Requirement
overlap must never be used to infer execution dependency.

Every newly generated Planner task contains:

```python
{
    "requirement_ids": ["REQ-001"],
    "depends_on_task_ids": ["T1"],
}
```

`depends_on_task_ids` is a unique list of existing task IDs. It cannot contain
the task itself, must reference tasks in the same plan, and the complete plan
must be acyclic. Initial plans require dependencies to point to earlier tasks.
Replacement plans are topologically validated before commit. A synthesis task
must explicitly depend on every task result it consumes.

A dependency is satisfied only when the upstream task has a terminal
`COMMITTED` or `DEGRADED` outcome. An upstream user blocker leaves the dependency
unsatisfied. The deterministic runnable-task helper scans for tasks whose
explicit dependencies are satisfied; it never substitutes shared requirement
IDs for dependency edges.

## 5. Failure classification and actions

### RETRYABLE_EXECUTION

Timeouts, temporary API/tool/network failures, and transient structured-output
failures receive an explicit budget. Success returns to verification. Exhaustion
is reclassified according to available fallback:

- safe partial result exists → `DEGRADABLE_QUALITY`;
- only the user can provide a hard prerequisite → `USER_DECISION_REQUIRED`;
- core execution/state cannot proceed → `FATAL_SYSTEM`.

System retries never create a business blocker by themselves.

Verifier invocation or assessment-contract failure receives the existing one
bounded verifier retry. If deterministic preflight already found an actionable
contract issue, that issue continues through ordinary policy without requiring
semantic verification. Otherwise, exhaustion of the semantic Verifier is
classified as `FATAL_SYSTEM / VERIFIER_UNAVAILABLE` with action `FAIL_JOB`.
Unverified semantic content is neither degraded nor committed, and Verifier
exhaustion never becomes `USER_DECISION_REQUIRED`.

### REPAIRABLE_CONTRACT

Length, citation numbering, source projection, formatting, lightweight schema,
and caption/reference mismatches use deterministic repair or a bounded LLM
repair. They do not route to PlanPatcher or Human merely because repair failed.

Length behavior remains one semantic rewrite per task revision. If the result
is still too long, paragraph-safe deterministic reduction is attempted. A soft
target then commits with a warning. A mandatory hard bound creates a hard
constraint failure and, only if changing the contract is required, a user
blocker. The user is never asked how the system should shorten prose.

### DEGRADABLE_QUALITY

Soft evidence gaps, optional asset failures, noncritical missing resources, and
unmet soft quality targets commit supported content with a stable warning. The
result, valid tables, citations, and evidence are retained. Dangling figure
references and empty captions are removed deterministically.

This is a terminal outcome for the current task and does not re-enter a repair
loop or require user acceptance.

### USER_DECISION_REQUIRED

This class is limited to:

- a missing hard-required file or action only the user can provide;
- a necessary change to an explicit hard requirement;
- irreconcilable user requirements;
- explicit human approval;
- a choice that changes the user contract.

Each blocker has a stable ID, task ID, requirement ID, reason, attempted
repairs, required action, and available options. It is registered rather than
immediately interrupting execution.

### FATAL_SYSTEM

Unrecoverable state/persistence corruption, uninterpretable core State, and a
core dependency failure with no fallback fail the job. Diagnostics record the
exception type, component, operation, and safe bounded context without secrets
or raw LLM content. UI and CLI show an actionable system error and never show a
business blocker form.

Fatal ownership has two explicit layers:

- Graph-level DecisionPolicy owns failures represented inside a valid
  checkpoint State, including bounded core model/tool exhaustion, an invalid
  but readable State contract, and Verifier exhaustion. It writes
  `fatal_system_error`, selects `FAIL_JOB`, and routes to the existing Exit
  lifecycle without creating a blocker.
- Runner-level handling in `app.py` and `run.py` owns failures that prevent the
  graph from producing or persisting a valid decision, including checkpoint
  read/write failure, SQLite corruption, stream/runtime exceptions escaping a
  node, and Store projection failure. It marks the job failed when projection
  is available and surfaces a safe diagnostic. It must not inject a fabricated
  graph blocker or resume the graph after the fatal exception.

If the persistence backend itself is unavailable, runner-level diagnostics are
reported to UI/CLI/logs only; the implementation must not claim that the fatal
record was checkpointed.

## 6. Recovery component boundaries

EvidenceRecovery remains an execution helper that constructs one bounded local
research attempt and returns its outcome to DecisionPolicy. Exhausted soft
evidence recovery degrades; a hard evidence requirement can create a blocker.
EvidenceRecovery cannot directly interrupt.

AssetRecovery remains inside the asset pipeline:

```text
generate → deterministic validate → optional one-shot repair → fallback
```

Optional failure preserves text/tables and records degradation. A hard-required
asset creates a blocker only after the bounded repair fails.

PlanPatcher accepts only stable plan-defect subtypes: missing task, invalid
dependency/order, unexecutable task objective, task conflict, or invalid task
decomposition. Evidence, asset, length, citation, format, and transient errors
cannot route to it. Existing task/job patch budgets, revision reason, and
history remain.

## 7. Registries and stable identity

State gains these serializable fields:

- `requirement_registry`
- `failure_decision`
- `degraded_issue_registry`
- `pending_user_blockers`
- `blocker_resolution_registry`
- `resolved_user_blocker_ids`
- `task_outcome_registry`
- `fatal_system_error`

Registry items have these exact schemas:

```python
class RequirementProvenance(TypedDict):
    origin: str                 # explicit_user | intake_inferred |
                                # legacy_reconstructed | legacy_task_contract
    source_message_id: str | None
    source_field: str
    source_index: int
    derivation: str


class RequirementRecord(TypedDict):
    requirement_id: str
    text: str
    severity: str               # hard | soft
    kind: str
    status: str                 # active | modified | withdrawn
    contract_revision: int
    provenance: RequirementProvenance


class RepairAttemptRecord(TypedDict):
    repair_type: str
    attempt: int
    budget: int
    outcome: str
    diagnostic_code: str | None


class DegradedIssueRecord(TypedDict):
    issue_id: str
    task_id: str
    task_revision: int
    failure_class: str          # DEGRADABLE_QUALITY
    subtype: str
    reason: str
    affected_claims: list[str]
    affected_requirement_ids: list[str]
    attempted_repairs: list[RepairAttemptRecord]
    final_fallback: str
    status: str                 # active | superseded
    metadata: dict[str, JSONValue]


class UserBlockerRecord(TypedDict):
    blocker_id: str
    status: str                 # pending | retry_pending | resolved | cancelled
    task_id: str
    requirement_ids: list[str]
    affected_task_ids: list[str]
    reason: str
    required_user_action: str
    available_options: list[str]
    attempted_repairs: list[RepairAttemptRecord]
    metadata: dict[str, JSONValue]


class RequirementModification(TypedDict):
    requirement_id: str
    operation: str              # update | withdraw
    previous_text: str
    new_text: str | None
    previous_severity: str
    new_severity: str | None
    previous_contract_revision: int
    new_contract_revision: int


class BlockerResolutionRecord(TypedDict):
    resolution_id: str
    blocker_id: str
    action: str                 # UPLOAD_RESOURCES | MODIFY_REQUIREMENT |
                                # APPROVE_EXCEPTION | CANCEL_JOB
    status: str                 # applied | retry_pending | rejected
    resource_ids: list[str]
    requirement_modification: RequirementModification | None
    affected_task_ids: list[str]
    contract_revision: int
    metadata: dict[str, JSONValue]


class TaskOutcomeRecord(TypedDict):
    task_id: str
    status: str                 # pending | running | committed | degraded |
                                # blocked_dependency | blocked_user
    dependency_ids: list[str]
    blocker_ids: list[str]
    task_revision: int


class FatalSystemError(TypedDict):
    failure_id: str
    failure_class: str          # FATAL_SYSTEM
    subtype: str
    origin: str                 # graph | runner
    component: str
    operation: str
    task_id: str | None
    diagnostic_code: str
    retryable: bool             # always False for the final fatal record
    metadata: dict[str, JSONValue]
```

The State containers are exact: `requirement_registry` and
`degraded_issue_registry` are ordered lists; `pending_user_blockers` and
`blocker_resolution_registry` are ordered lists; `resolved_user_blocker_ids` is
a stable-deduplicated list of strings; `task_outcome_registry` is a dictionary
keyed by task ID; `failure_decision` is one `FailureDecision` or `{}`; and
`fatal_system_error` is one `FatalSystemError` or `{}`.

Degradation IDs are derived from task ID, task revision, subtype, and affected
requirement IDs. A blocker ID is derived from job scope, task ID, subtype,
sorted requirement IDs, and any canonical missing-resource identity. It never
contains `task_revision`, `plan_revision`, retry count, or repair attempt, so
the same unresolved contract problem upserts across task rewrites. Task
revision remains metadata on the failure/attempt rather than blocker identity.
`resolution_id` is derived from blocker ID plus the canonical action payload.
Upsert-by-ID prevents duplicate warnings, blockers, or resolutions after
retry/resume.

Existing `accepted_evidence_gaps`, `accepted_gap_fingerprints`,
`pending_user_action`, retry counters, and patch history are not removed. Old
accepted gaps are translated to a compatibility degradation view. The new
workflow does not add ordinary accepted-evidence-gap records.

## 8. Consolidated HITL and admission

When DecisionPolicy produces `USER_DECISION_REQUIRED`, it records the blocker,
marks the current section blocked, and advances to the next task that can run
without the missing hard prerequisite. The runnable-task helper follows only
`depends_on_task_ids`: a task continues when every dependency is committed or
degraded; a task with a blocked dependency becomes `blocked_dependency`.
Requirement overlap is used only to report affected contract scope, never to
make scheduling decisions. `affected_task_ids` is the deterministic transitive
closure of the blocker task through explicit dependency edges.

A deterministic admission helper runs when the current graph would otherwise
route to Summarizer. If unresolved blockers exist, it routes to the existing
NeedsUserInput node. No ReportGate node is added.

NeedsUserInput interrupts once with a payload containing all unresolved
blockers. Resume data maps each selected resolution to `blocker_id`. Resolved
IDs are persisted; unresolved blockers may be shown again only after a partial
submission, never as duplicate records. Resuming never re-executes committed
tasks.

### Blocker resolution and affected-task resume

Resolution is an explicit state transition, not a synonym for accepting an
issue:

- `UPLOAD_RESOURCES` ingests only the submitted resources, records their stable
  resource IDs, changes the blocker to `retry_pending`, and schedules the
  blocker task plus its explicit transitive dependents for re-execution. The
  blocker becomes `resolved` only after the originating hard requirement passes
  verification.
- `MODIFY_REQUIREMENT` creates a new contract revision and appends the prior
  requirement record to modification history. The stable requirement ID is
  retained when text/severity changes; removal marks it `withdrawn`. Planner
  revalidates task linkage and dependency edges against the new contract before
  affected tasks resume. Only the explicitly affected task and its dependency
  descendants are invalidated.
- `APPROVE_EXCEPTION` is available only when the original contract explicitly
  allows human approval/waiver. It records a user-authored hard-requirement
  exception, resolves the blocker, and converts affected terminal output to a
  visible warning. It is not the replacement for ordinary evidence-gap
  acceptance.
- `CANCEL_JOB` records the resolution and terminates the job without admitting
  a report.

Applying the same canonical resolution twice is idempotent through
`resolution_id`. A conflicting second resolution is rejected rather than
silently overwriting contract history. Unrelated committed/degraded tasks retain
their results and revisions. Affected committed tasks are invalidated only when
their input dependency or linked requirement actually changed. The scheduler
selects the earliest affected task whose dependencies are again terminal; its
descendants resume in dependency order. After verification, resolved blocker
IDs are removed from pending projection but retained in the resolution audit
registry.

## 9. UI behavior

The blocker UI renders only `USER_DECISION_REQUIRED` entries. A consolidated
payload is displayed as a list of blocker forms keyed by stable blocker ID.
Available actions are determined per blocker:

- upload resources only for a hard missing-file requirement;
- adjust requirement only when contract modification is valid;
- explicit approval only when the original requirement demands approval;
- retry is not offered for internal transient/system failures.

The legacy UI action `ADJUST_REQUIREMENT` is a compatibility alias for the
canonical resolution action `MODIFY_REQUIREMENT`; persisted new resolution
records always use the canonical action.

Ordinary degradation appears as nonblocking warning/status text. The existing
radio-based action selector pattern is retained for each real blocker. Legacy
single-blocker payloads continue to render and resume.

## 10. Report behavior

Summarizer consumes `degraded_issue_registry` deterministically. Each warning
states the supported boundary and missing capability without claiming the gap
was resolved or accepted by the user. Soft degradation produces a deliverable
report with warnings. An unresolved hard requirement never passes admission.

Legacy accepted gaps continue to produce compatible warning text, while new
soft gaps are described as system-recorded limitations rather than
user-approved incomplete drafts.

## 11. Persistence and migration

`SqliteSaver` remains the graph execution source of truth. `SqliteStore` remains
the job/UI/history projection. No SQLite schema change is required because new
LangGraph values are plain JSON/checkpoint-serializable structures.

Every reader uses explicit compatibility normalization for new fields.

For a legacy checkpoint without `requirement_registry`, normalization follows
this order:

1. Reconstruct canonical requirements from the latest persisted
   `INTAKE_SUMMARY` when available. Assign deterministic `LEGACY-REQ-*` IDs,
   persist them on the first successful state transition, and use provenance
   `legacy_reconstructed`.
2. Only unambiguous explicit hard markers and named mandatory user resources
   reconstruct as hard. Other reconstructed requirements default to soft.
3. If Intake authority is unavailable, create narrowly scoped
   `LEGACY-TASK-*` requirements only for structural obligations already encoded
   in task fields, such as a named `use_resources` entry or explicit
   `generate_figure=True`. Provenance is `legacy_task_contract`. Semantic issue
   prose is never used to invent a hard requirement.
4. If neither source exists, use an empty registry. Existing committed results,
   accepted waivers, and resolved blockers remain valid; a new issue cannot be
   escalated to hard solely because legacy context is missing.

For legacy tasks without `depends_on_task_ids`, preserve the historical serial
contract by assigning each nonsynthesis task a dependency on the immediately
preceding task and assigning synthesis a dependency on all preceding report
tasks. This conservative chain prevents previously sequential work from being
treated as newly independent. New Planner output must always carry explicit
dependencies.

Compatibility tests cover checkpoints with no requirement, dependency,
failure, degradation, outcome, or blocker registries. Existing cursor-keyed
retry counters continue to normalize to task IDs. Completed results, warnings,
blockers, and resolutions are upserted by stable identity during resume.

Persistence exceptions outside graph execution are runner-level
`FATAL_SYSTEM`, as defined in Section 5. They are classified for UI/job
projection when possible. If the checkpoint database itself is unavailable,
the diagnostic is surfaced without pretending it was saved to the broken
database.

## 12. Observability

Every policy decision emits structured metadata:

- `task_id`
- `failure_class`
- `subtype`
- `policy_action`
- `repair_attempt`
- `repair_budget`
- `degraded_issue_id`
- `blocker_id`
- `requirement_id`
- `hard_or_soft`

Logs contain no prompts, responses, knowledge-base content, documents, API
keys, or exception text. Existing LLM observability and HTTP logging remain
unchanged.

## 13. Test strategy

Implementation follows red-green-refactor in bounded phases. Required scenario
coverage includes:

1. optional ConceptGraph failure degrades, preserves text/table, removes
   dangling references, and never interrupts;
2. soft length overflow rewrites once, reduces deterministically, then warns and
   commits without Human;
3. hard length failure follows hard-contract policy without asking the user how
   to shorten content;
4. exhausted soft evidence recovery commits with warning;
5. hard evidence gap produces a blocker containing requirement and attempted
   repair details;
6. missing hard-required user file blocks only dependent work;
7. transient tool error retries within budget and succeeds without Human;
8. fatal state/persistence error surfaces a system diagnostic and no blocker
   actions;
9. two blockers are collected into one interrupt;
10. degradation, consolidated interrupt, resolution, and old-checkpoint resume
    are idempotent;
11. PlanPatcher rejects non-plan failure routes;
12. degradation warnings appear in final report and UI;
13. structured decision logs contain the required metadata and no sensitive
    bodies;
14. new plans reject missing/unknown/cyclic dependencies and runtime scheduling
    follows explicit dependency edges rather than requirement overlap;
15. legacy plans receive conservative serial dependencies;
16. legacy checkpoints reconstruct only authoritative requirements and never
    promote issue prose to hard constraints;
17. the same blocker across task revisions retains one blocker ID;
18. upload, requirement modification, approval, duplicate resolution, and
    affected-task resume follow the state transitions in Section 8;
19. semantic Verifier exhaustion becomes `FATAL_SYSTEM /
    VERIFIER_UNAVAILABLE`, not degradation or Human;
20. graph-level fatal and runner-level persistence/stream fatal paths produce
    their respective safe diagnostics without blocker forms;
21. all registry records satisfy the exact schemas in Section 7;
22. the complete pre-existing suite remains green.

## 14. Delivery sequence

Implementation is split into independently testable phases:

1. failure taxonomy and compatibility normalization;
2. requirement registry, provenance, and Planner requirement/dependency
   linkage;
3. DecisionPolicy mapping and bounded terminal outcomes;
4. degradation registry and local Evidence/Asset recovery boundaries;
5. PlanPatcher route narrowing;
6. consolidated blocker admission, canonical resolution, and dependency-based
   affected-task resume;
7. UI and Summarizer projections;
8. fatal-system handling, observability, compatibility tests, and documentation;
9. full regression and branch integration.

Each phase starts with failing tests, changes only the files required for that
behavior, and runs its focused test set before the next phase.

## 15. Definition of done

The work is complete only when optional asset failure, soft length failure, and
ordinary evidence gaps never require Human; NeedsUserInput accepts only
`USER_DECISION_REQUIRED`; Verifier and DecisionPolicy responsibilities are
separate; retries and repairs are bounded; degradation is terminal; hard/soft
requirements are persisted and linked to tasks; multiple blockers consolidate;
EvidenceRecovery and AssetRecovery have local authority only; PlanPatcher fixes
only plan defects; final reports expose degradation; new State survives old and
new checkpoint resume; persistence authority remains unchanged; and the full
test suite passes.

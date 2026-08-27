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
```

`FailureDecision` is serialized to a plain dictionary in LangGraph State. It
contains:

- `failure_class`
- `subtype`
- `reason`
- `task_id`
- `action`
- `retryable`
- `repair_attempt`
- `repair_budget`
- `user_blocker`
- `hard_requirement_ids`
- `metadata`

Verifier continues to judge whether a result satisfies its task. It does not
choose recovery actions or user interruption. DecisionPolicy is the sole owner
of failure classification and workflow action.

The old `IssueCategory` remains as an assessment compatibility input during
migration. It is not the authority for new workflow decisions.

## 4. Requirement contract

### Canonical requirements

Intake creates a serializable requirement registry. Each record contains:

```python
{
    "requirement_id": "REQ-001",
    "text": "必须使用用户提供的质量控制规程",
    "severity": "hard",
    "kind": "resource",
    "source": "explicit_user_constraint",
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

Planner model tasks gain a required `requirement_ids` list. Validation rejects
unknown IDs. Synthesis tasks may reference report-wide IDs. Legacy tasks with
no field normalize to an empty list; this means no requirement may be invented
as hard during checkpoint recovery.

Verifier issues may include `requirement_ids`. Deterministic checks attach IDs
from the active task. DecisionPolicy resolves severity through the registry,
not issue prose.

## 5. Failure classification and actions

### RETRYABLE_EXECUTION

Timeouts, temporary API/tool/network failures, and transient structured-output
failures receive an explicit budget. Success returns to verification. Exhaustion
is reclassified according to available fallback:

- safe partial result exists → `DEGRADABLE_QUALITY`;
- only the user can provide a hard prerequisite → `USER_DECISION_REQUIRED`;
- core execution/state cannot proceed → `FATAL_SYSTEM`.

System retries never create a business blocker by themselves.

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
- `resolved_user_blocker_ids`
- `fatal_system_error`

Degradation IDs are derived from task ID, task revision, subtype, and affected
requirement IDs. Blocker IDs use the same stable scope. Upsert-by-ID prevents
duplicate warnings or blockers after retry/resume.

Existing `accepted_evidence_gaps`, `accepted_gap_fingerprints`,
`pending_user_action`, retry counters, and patch history are not removed. Old
accepted gaps are translated to a compatibility degradation view. The new
workflow does not add ordinary accepted-evidence-gap records.

## 8. Consolidated HITL and admission

When DecisionPolicy produces `USER_DECISION_REQUIRED`, it records the blocker,
marks the current section blocked, and advances to the next task that can run
without the missing hard prerequisite. Because the current planner has no
dependency graph, continuation is conservative:

- tasks with no overlap with the blocker requirement IDs continue;
- synthesis/report-wide tasks may execute only from committed inputs but final
  admission remains blocked;
- tasks sharing affected hard requirement IDs remain blocked without execution.

A deterministic admission helper runs when the current graph would otherwise
route to Summarizer. If unresolved blockers exist, it routes to the existing
NeedsUserInput node. No ReportGate node is added.

NeedsUserInput interrupts once with a payload containing all unresolved
blockers. Resume data maps each selected resolution to `blocker_id`. Resolved
IDs are persisted; unresolved blockers may be shown again only after a partial
submission, never as duplicate records. Resuming never re-executes committed
tasks.

## 9. UI behavior

The blocker UI renders only `USER_DECISION_REQUIRED` entries. A consolidated
payload is displayed as a list of blocker forms keyed by stable blocker ID.
Available actions are determined per blocker:

- upload resources only for a hard missing-file requirement;
- adjust requirement only when contract modification is valid;
- explicit approval only when the original requirement demands approval;
- retry is not offered for internal transient/system failures.

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

Every reader uses default-empty normalization for new fields. Compatibility
tests cover old checkpoints with no requirement, failure, degradation, or
blocker registries. Existing cursor-keyed retry counters continue to normalize
to task IDs. Completed results, warnings, blockers, and resolutions are upserted
by stable identity during resume.

Persistence exceptions outside graph execution are classified for UI/job
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
14. the complete pre-existing suite remains green.

## 14. Delivery sequence

Implementation is split into independently testable phases:

1. failure taxonomy and compatibility normalization;
2. requirement registry and Planner requirement linkage;
3. DecisionPolicy mapping and bounded terminal outcomes;
4. degradation registry and local Evidence/Asset recovery boundaries;
5. PlanPatcher route narrowing;
6. consolidated blocker admission and resume;
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

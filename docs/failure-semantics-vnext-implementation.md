# Failure Semantics / Recovery vNext Implementation

Implemented on `codex/sqlite-checkpoint-store` on 2026-08-27.

## Runtime contract

- `FailureDecision` is the canonical DecisionPolicy output. Existing
  `workflow_action` values remain routing projections for graph compatibility.
- New Planner tasks carry both `requirement_ids` and
  `depends_on_task_ids`. Requirement overlap is never interpreted as an
  execution dependency.
- Soft evidence, optional asset, and soft quality exhaustion commit supported
  content as `DEGRADABLE_QUALITY`; warnings are persisted in
  `degraded_issue_registry` and rendered in draft artifacts.
- Only explicit hard-contract choices create `USER_DECISION_REQUIRED` records.
  Blockers are collected, keyed by revision-independent `blocker_id`, and shown
  in one consolidated interrupt after independent runnable work completes.
- Semantic Verifier exhaustion is graph-level
  `FATAL_SYSTEM / VERIFIER_UNAVAILABLE`. Escaped stream/persistence failures are
  runner-level fatal diagnostics in `app.py` and `run.py`; neither path creates
  a business blocker.

## Resolution semantics

- `UPLOAD_RESOURCES` ingests only resources submitted for that blocker, marks
  it `retry_pending`, and invalidates the origin plus explicit dependency
  descendants.
- `MODIFY_REQUIREMENT` retains the stable requirement ID, increments its
  contract revision, records the before/after contract, and resumes only the
  affected dependency closure.
- `APPROVE_EXCEPTION` is accepted only for a requirement whose contract kind
  explicitly grants approval/waiver authority.
- `CANCEL_JOB` routes directly to Exit without report admission.
- Canonically identical resolutions are idempotent; a different second
  resolution for the same blocker is rejected.

## Legacy checkpoint migration

No SQLite schema migration is required. New fields are ordinary checkpoint
values and are normalized on policy entry. A missing requirement registry is
reconstructed from the latest authoritative `INTAKE_SUMMARY`; only explicit
hard markers and named mandatory resources become hard. If Intake authority is
absent, only structural task obligations are recovered. Issue or assessment
prose never creates hard authority. Missing dependency lists receive the
historical serial dependency contract.

## Observability and privacy

Each failure decision logs attribution, class, subtype, policy action, repair
attempt/budget, stable degradation/blocker IDs, requirement IDs, and severity.
Logs exclude prompts, model responses, documents, issue descriptions,
exception messages, and credentials. Existing LLM call observability remains
unchanged.

## Verification

The final implementation suite passed `748` tests. The delivery sequence was
committed as independently testable Phase 1–8 checkpoints.

# SQLite Checkpoint and Store Persistence Design

## Goal

Replace process-local LangGraph state with durable SQLite persistence and make
historical report jobs recoverable after the Streamlit application or CLI
process restarts.

The implementation must persist:

- LangGraph state, checkpoint history, pending writes, and interrupts.
- A per-user index of report jobs.
- Streamlit UI messages needed to rebuild the page.
- Report paths and recovery metadata.

## Current State

`app.py` and `run.py` currently compile the workflow with `MemorySaver`.
Checkpoints therefore disappear when the Python process exits. The graph does
not currently receive a LangGraph Store.

The existing scope model is retained:

- `user_id` identifies the local application user.
- `conversation_id` groups related report jobs.
- `job_id` is also the LangGraph `thread_id`.

## User Identity

The local application uses `AGENT_USER_ID` when it is set. Otherwise it uses
the stable development identity `local-user`.

This is intentionally a local-development identity adapter. A production
authentication system must replace it with its own stable user identifier.

## Architecture

Persistence is isolated in `src/persistence.py`. It owns the SQLite
connections, schema setup, connection settings, and the LangGraph persistence
objects exposed to callers.

Two separate databases are used:

- `cache/langgraph/checkpoints.sqlite` is managed by LangGraph
  `SqliteSaver`.
- `cache/langgraph/store.sqlite` is managed by LangGraph `SqliteStore`.

The separation keeps thread-level execution state independent from
application-level job discovery data. It also permits either database to be
migrated to a different backend later.

Both Streamlit and the CLI compile the graph with:

```python
workflow.compile(
    checkpointer=persistence.checkpointer,
    store=persistence.store,
)
```

Streamlit keeps one cached persistence resource for the application process.
The CLI keeps the resource open for the duration of the command and closes it
on exit.

## Store Data Model

Job records use the namespace:

```python
(user_id, "jobs")
```

The Store key is `job_id`. The value contains:

```text
user_id
conversation_id
job_id
title
verifier_mode
status
created_at
updated_at
pending_interrupt
ui_messages
report_paths
```

Valid statuses are:

```text
created
running
waiting
completed
failed
```

Historical jobs are sorted by `updated_at` in descending order. Store access
always includes the active `user_id`; a job from another namespace cannot be
restored.

## Streamlit Data Flow

### Startup

1. Resolve the stable local `user_id`.
2. Initialize both SQLite databases.
3. Read the user's job records from Store.
4. Display historical jobs in the sidebar.
5. Compile the selected manual or automatic workflow with both persistence
   objects.

### New Job

1. Generate a `conversation_id` and `job_id`.
2. Wait until the first valid user submission before creating a Store record,
   so empty jobs do not pollute history.
3. Save the initial UI message and mark the job `running`.
4. Invoke the graph with `job_id` as `configurable.thread_id`.

### Execution

LangGraph writes graph state after each super-step. The application updates the
Store record when UI messages, status, interrupt information, or report paths
change.

An interrupt changes the Store status to `waiting`. A normal graph completion
changes it to `completed`. An application-visible execution failure changes it
to `failed` without deleting existing checkpoints.

### Restore

Historical jobs are not restored automatically. The user selects a job and
clicks a restore action.

The application then:

1. Loads the Store record from the current user's namespace.
2. Restores `conversation_id`, `job_id`, verifier mode, and UI messages into
   Streamlit session state.
3. Reads the latest graph snapshot using the restored `job_id`.
4. Treats the checkpoint as the source of truth for graph state and pending
   interrupts.
5. Restores the interrupt input state when the graph is waiting.
6. Rebuilds report download controls from the latest state and stored report
   paths.

## CLI Data Flow

The CLI adds optional `--thread-id` and `--user-id` arguments.

- Without `--thread-id`, it creates a new job.
- With `--thread-id`, it loads the existing checkpoint.
- On startup it inspects the latest graph snapshot for interrupts instead of
  relying only on an in-process boolean.
- `--user-id` overrides `AGENT_USER_ID`; otherwise the CLI uses `local-user`.

## Failure Handling

- Database setup failure blocks graph compilation and reports the affected
  path. There is no silent fallback to `MemorySaver`.
- A Store job record is created before graph execution starts, reducing the
  chance of an undiscoverable checkpoint.
- A Store update failure does not delete or overwrite an existing checkpoint.
- A Store record with no corresponding checkpoint is reported as incomplete
  and cannot be resumed.
- Restore rejects records outside the current user's namespace.
- Corrupt or unreadable SQLite files produce an actionable error rather than
  creating replacement databases over them.

## Concurrency and File Safety

The two databases enable WAL mode, foreign-key checks, and a finite busy
timeout. SQLite connections allow Streamlit worker-thread access, while
persistence-layer locks serialize shared connection operations.

The `cache/langgraph/` directory and database files remain ignored by Git.
They are runtime data and may contain user prompts, generated content, and
internal graph state.

## Dependencies and Serialization Security

The implementation upgrades `langgraph` from `1.0.1` to `1.0.10` and adds
`langgraph-checkpoint-sqlite==3.0.3`.

`LANGGRAPH_STRICT_MSGPACK=true` is documented and enabled for local startup to
restrict checkpoint deserialization. LangGraph cache backends are not enabled
as part of this change.

References:

- <https://docs.langchain.com/oss/python/langgraph/persistence>
- <https://pypi.org/project/langgraph/1.0.10/>
- <https://pypi.org/project/langgraph-checkpoint-sqlite/3.0.3/>

## Scope Exclusions

- Production authentication and authorization.
- PostgreSQL or other networked persistence backends.
- Semantic/vector search over Store records.
- Automatic checkpoint retention or cleanup.
- Automated test implementation or execution, per the user's explicit
  direction for this change.

## Acceptance Criteria

- Restarting Streamlit does not destroy LangGraph checkpoints.
- A historical job can be selected and its graph state, interrupt, UI
  messages, status, and report paths are restored.
- The CLI can resume a known `thread_id`.
- Jobs are isolated by `user_id`.
- Runtime SQLite files are not tracked by Git.
- Neither application entry point uses `MemorySaver`.

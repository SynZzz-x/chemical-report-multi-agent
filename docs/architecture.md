# Architecture

## Workflow

The system uses LangGraph to coordinate a stateful report-generation pipeline:

1. `Intake` extracts the user's request and uploaded resources.
2. `Planner` converts the request into a task list and maps tasks to resources.
3. `Worker` executes the current task with tools such as CSV analysis, chart generation, crawling, or knowledge-base retrieval.
4. `Verifier` decides whether to accept the result, retry the worker, replan, or finish.
5. `Summarizer` merges verified results into final report artifacts.
6. `Exit` archives output paths and recovery information.

## State Model

The shared state includes:

- `messages`: conversation and internal routing messages.
- `docs`: uploaded resources.
- `tasks`: planned task objects.
- `cursor`: index of the current task.
- `current_result`: worker output for the active task.
- `results`: accepted task results.
- `decision`: verifier route decision.
- `feedback`: actionable retry or replan information.
- `metadata`: session and request identifiers.

## Verification Modes

Manual verification interrupts the graph and asks the user to approve or revise each result. Automatic verification uses an LLM-based assessment and routes to retry, replan, next task, or summarization.

## Design Tradeoffs

The project favors explicit state and visible checkpoints over fully autonomous execution. That makes the system easier to debug, demo, and discuss in interviews.

## Persistence

`job_id` is the LangGraph `thread_id`. `SqliteSaver` stores thread-level graph
state and interrupts in `cache/langgraph/checkpoints.sqlite`. `SqliteStore`
stores user-scoped job records, UI messages, statuses, and report paths in
`cache/langgraph/store.sqlite`.

Streamlit restores historical jobs only after an explicit user selection.
Checkpoint state is authoritative for workflow execution; Store records are
the discovery and presentation layer. The CLI accepts `--thread-id` to reopen
the same checkpoint.

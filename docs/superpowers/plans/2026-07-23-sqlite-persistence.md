# SQLite Checkpoint and Store Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist LangGraph checkpoints and application job history in SQLite so Streamlit and the CLI can recover report jobs after a process restart.

**Architecture:** A new persistence module owns separate SQLite connections for `SqliteSaver` and `SqliteStore`. A focused job repository stores per-user recovery records, while Streamlit and the CLI continue using `job_id` as the LangGraph `thread_id`.

**Tech Stack:** Python 3.10+, LangGraph 1.0.10, langgraph-checkpoint-sqlite 3.0.3, SQLite, Streamlit.

## Global Constraints

- Work on branch `codex/sqlite-checkpoint-store`.
- Use `AGENT_USER_ID`, defaulting to the stable local identity `local-user`.
- Store checkpoints in `cache/langgraph/checkpoints.sqlite`.
- Store job records in `cache/langgraph/store.sqlite`.
- Compile every top-level graph with both `checkpointer` and `store`.
- Do not silently fall back to `MemorySaver`.
- Set `LANGGRAPH_STRICT_MSGPACK=true` unless the caller explicitly overrides it.
- Do not enable a LangGraph cache backend.
- Keep all SQLite runtime files ignored by Git.
- Do not add or execute automated tests, per the user's explicit direction.

---

### Task 1: Add compatible persistence dependencies and configuration

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `src/config.py`

**Interfaces:**
- Consumes: Existing `get_env()` and `get_cache_root()` helpers.
- Produces: `get_local_user_id() -> str` and `get_langgraph_data_dir() -> Path`.

- [ ] **Step 1: Update the pinned LangGraph dependencies**

Replace:

```text
langgraph==1.0.1
```

with:

```text
langgraph==1.0.10
langgraph-checkpoint-sqlite==3.0.3
```

- [ ] **Step 2: Add local identity and strict deserialization settings**

Append to `.env.example`:

```dotenv

# Local persistence identity and checkpoint hardening.
AGENT_USER_ID=local-user
LANGGRAPH_STRICT_MSGPACK=true
```

- [ ] **Step 3: Add persistence configuration helpers**

Add to `src/config.py` after `get_cache_root()`:

```python
def get_local_user_id() -> str:
    """Return the stable local identity used to namespace persisted jobs."""

    return get_env("AGENT_USER_ID", "local-user") or "local-user"


def get_langgraph_data_dir() -> Path:
    """Return the private runtime directory for LangGraph SQLite files."""

    directory = get_cache_root() / "langgraph"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory
```

- [ ] **Step 4: Review the dependency/configuration diff**

Run:

```bash
git diff -- requirements.txt .env.example src/config.py
```

Expected: only the two dependency pins, two environment settings, and two
configuration helpers are present.

- [ ] **Step 5: Commit the configuration change**

```bash
git add requirements.txt .env.example src/config.py
git commit -m "chore: configure SQLite persistence"
```

---

### Task 2: Implement SQLite resources and the job repository

**Files:**
- Create: `src/persistence.py`
- Create: `src/job_store.py`

**Interfaces:**
- Consumes: `get_langgraph_data_dir()`.
- Produces: `SQLitePersistence.open()`, `SQLitePersistence.close()`,
  `JobStore.create_job()`, `JobStore.get_job()`, `JobStore.list_jobs()`,
  `JobStore.update_job()`, and `interrupt_from_snapshot()`.

- [ ] **Step 1: Create the SQLite resource owner**

Create `src/persistence.py`:

```python
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from .config import get_langgraph_data_dir


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(path),
        timeout=30,
        check_same_thread=False,
        isolation_level=None,
    )
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _restrict_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.chmod(0o600)
    except OSError:
        pass


@dataclass
class SQLitePersistence:
    root: Path
    checkpoint_connection: sqlite3.Connection
    store_connection: sqlite3.Connection
    checkpointer: SqliteSaver
    store: SqliteStore
    _closed: bool = False

    @classmethod
    def open(cls, root: Path | None = None) -> "SQLitePersistence":
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        data_dir = (root or get_langgraph_data_dir()).expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = data_dir / "checkpoints.sqlite"
        store_path = data_dir / "store.sqlite"
        checkpoint_connection = _connect(checkpoint_path)

        try:
            store_connection = _connect(store_path)
            checkpointer = SqliteSaver(checkpoint_connection)
            store = SqliteStore(store_connection)
            checkpointer.setup()
            store.setup()
        except Exception:
            checkpoint_connection.close()
            if "store_connection" in locals():
                store_connection.close()
            raise

        _restrict_file(checkpoint_path)
        _restrict_file(store_path)
        return cls(
            root=data_dir,
            checkpoint_connection=checkpoint_connection,
            store_connection=store_connection,
            checkpointer=checkpointer,
            store=store,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.store_connection.close()
        self.checkpoint_connection.close()
        self._closed = True

    def __enter__(self) -> "SQLitePersistence":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
```

- [ ] **Step 2: Create the Store-backed job repository**

Create `src/job_store.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from langgraph.store.base import BaseStore


JobStatus = Literal["created", "running", "waiting", "completed", "failed"]
VALID_JOB_STATUSES = {"created", "running", "waiting", "completed", "failed"}


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def job_namespace(user_id: str) -> tuple[str, str]:
    normalized = user_id.strip()
    if not normalized:
        raise ValueError("user_id must not be empty")
    return normalized, "jobs"


def interrupt_from_snapshot(snapshot: Any) -> Any | None:
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return getattr(interrupts[0], "value", interrupts[0])
    return None


class JobStore:
    def __init__(self, store: BaseStore):
        self.store = store

    def create_job(
        self,
        *,
        user_id: str,
        conversation_id: str,
        job_id: str,
        title: str,
        verifier_mode: str,
        ui_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        record = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "job_id": job_id,
            "title": " ".join(title.split())[:120] or "未命名任务",
            "verifier_mode": verifier_mode,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "pending_interrupt": None,
            "ui_messages": list(ui_messages or []),
            "report_paths": [],
        }
        self.store.put(job_namespace(user_id), job_id, record, index=False)
        return record

    def get_job(self, user_id: str, job_id: str) -> dict[str, Any] | None:
        item = self.store.get(job_namespace(user_id), job_id)
        if item is None:
            return None
        value = dict(item.value)
        if value.get("user_id") != user_id:
            return None
        return value

    def list_jobs(self, user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        items = self.store.search(job_namespace(user_id), limit=limit)
        records = [
            dict(item.value)
            for item in items
            if item.value.get("user_id") == user_id
        ]
        return sorted(
            records,
            key=lambda record: record.get("updated_at", ""),
            reverse=True,
        )

    def update_job(
        self,
        user_id: str,
        job_id: str,
        **changes: Any,
    ) -> dict[str, Any]:
        record = self.get_job(user_id, job_id)
        if record is None:
            raise KeyError(f"Unknown job_id for user {user_id}: {job_id}")

        status = changes.get("status")
        if status is not None and status not in VALID_JOB_STATUSES:
            raise ValueError(f"Invalid job status: {status}")

        record.update(changes)
        record["updated_at"] = utc_now_iso()
        self.store.put(job_namespace(user_id), job_id, record, index=False)
        return record
```

- [ ] **Step 3: Inspect the new persistence boundary**

Run:

```bash
git diff -- src/persistence.py src/job_store.py
rg -n "MemorySaver|SqliteSaver|SqliteStore|job_namespace|interrupt_from_snapshot" src
```

Expected: SQLite construction is confined to `src/persistence.py`; Store job
semantics are confined to `src/job_store.py`.

- [ ] **Step 4: Commit the persistence services**

```bash
git add src/persistence.py src/job_store.py
git commit -m "feat: add SQLite persistence services"
```

---

### Task 3: Integrate persistence and historical restore into Streamlit

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `SQLitePersistence`, `JobStore`, `get_local_user_id()`, and
  `interrupt_from_snapshot()`.
- Produces: a cached durable graph runtime, a historical-job selector, and
  recovery of scope, UI messages, interrupts, and report paths.

- [ ] **Step 1: Replace in-memory imports and initialize durable resources**

Remove:

```python
from langgraph.checkpoint.memory import MemorySaver
```

Add:

```python
from src.config import get_cache_root, get_local_user_id, missing_key_message
from src.job_store import JobStore, interrupt_from_snapshot
from src.persistence import SQLitePersistence
```

Replace the existing `src.config` import rather than adding a duplicate.

After `CACHE_ROOT = get_cache_root()`, add:

```python
@st.cache_resource
def _open_persistence() -> SQLitePersistence:
    return SQLitePersistence.open()


try:
    PERSISTENCE = _open_persistence()
    JOBS = JobStore(PERSISTENCE.store)
except Exception as exc:
    st.error(f"无法初始化 LangGraph SQLite 持久化：{exc}")
    st.stop()
```

- [ ] **Step 2: Make the Streamlit identity stable**

In `_ensure_session_scope()`, replace the random guest assignment:

```python
if "user_id" not in st.session_state:
    st.session_state["user_id"] = get_local_user_id()
```

Also initialize:

```python
if "job_record_created" not in st.session_state:
    st.session_state["job_record_created"] = False

if "last_run_failed" not in st.session_state:
    st.session_state["last_run_failed"] = False
```

- [ ] **Step 3: Centralize graph compilation and job persistence**

Add after `_job_metadata()`:

```python
def _compile_workflow(mode: str) -> None:
    workflow = WorkFlowAuto() if mode == "auto" else WorkFlow()
    st.session_state["app"] = workflow.compile(
        checkpointer=PERSISTENCE.checkpointer,
        store=PERSISTENCE.store,
    )
    st.session_state["compiled_mode"] = mode


def _current_job() -> dict[str, Any] | None:
    scope = _scope()
    return JOBS.get_job(scope["user_id"], scope["job_id"])


def _ensure_job_record(title: str, verifier_mode: str) -> dict[str, Any]:
    record = _current_job()
    if record is not None:
        st.session_state["job_record_created"] = True
        return record

    scope = _scope()
    record = JOBS.create_job(
        **scope,
        title=title,
        verifier_mode=verifier_mode,
        ui_messages=st.session_state["ui_messages"],
    )
    st.session_state["job_record_created"] = True
    return record


def _update_job(**changes: Any) -> None:
    if not st.session_state.get("job_record_created"):
        return
    scope = _scope()
    try:
        JOBS.update_job(scope["user_id"], scope["job_id"], **changes)
    except Exception as exc:
        st.warning(f"任务恢复信息保存失败：{exc}")
```

- [ ] **Step 4: Persist UI messages and reset new-job state**

At the end of `_append_ui_message()`, add:

```python
_update_job(ui_messages=list(st.session_state["ui_messages"]))
```

Update `_start_new_job()` to include:

```python
st.session_state["ui_messages"] = []
st.session_state["job_record_created"] = False
st.session_state["last_run_failed"] = False
```

- [ ] **Step 5: Add historical job restore helpers**

Add after `_start_new_conversation()`:

```python
def _restore_job(job_id: str) -> None:
    user_id = st.session_state["user_id"]
    record = JOBS.get_job(user_id, job_id)
    if record is None:
        raise ValueError("任务不存在或不属于当前用户。")

    mode = record.get("verifier_mode") or "manual"
    st.session_state["conversation_id"] = record["conversation_id"]
    st.session_state["active_job_id"] = record["job_id"]
    st.session_state["active_job_created_at"] = record["created_at"]
    st.session_state["ui_messages"] = list(record.get("ui_messages") or [])
    st.session_state["job_record_created"] = True
    st.session_state["verifier_mode"] = mode
    _compile_workflow(mode)

    if PERSISTENCE.checkpointer.get_tuple(_graph_config()) is None:
        raise ValueError("任务索引存在，但对应 checkpoint 缺失。")

    snapshot = st.session_state["app"].get_state(_graph_config())
    pending = interrupt_from_snapshot(snapshot)
    st.session_state["pending_interrupt"] = pending
    _update_job(pending_interrupt=pending)
```

- [ ] **Step 6: Add the sidebar history selector**

Give the verifier radio a stable key:

```python
verifier_mode = st.radio(
    "审核模式",
    options=["manual", "auto"],
    key="verifier_mode",
    format_func=lambda value: "人工审核" if value == "manual" else "自动审核",
    horizontal=True,
)
```

Replace direct compilation with:

```python
_compile_workflow(verifier_mode)
```

Before the environment configuration expander, add:

```python
st.divider()
st.subheader("历史任务")
job_records = JOBS.list_jobs(st.session_state["user_id"])

if job_records:
    job_by_id = {record["job_id"]: record for record in job_records}
    selected_job_id = st.selectbox(
        "选择任务",
        options=list(job_by_id),
        format_func=lambda job_id: (
            f"{job_by_id[job_id].get('title', '未命名任务')} · "
            f"{job_by_id[job_id].get('status', '-')}"
        ),
    )
    if st.button("恢复选中任务", use_container_width=True):
        try:
            _restore_job(selected_job_id)
        except Exception as exc:
            st.error(f"恢复失败：{exc}")
        else:
            st.rerun()
else:
    st.caption("暂无可恢复任务。")
```

- [ ] **Step 7: Persist execution status and interrupt state**

At the beginning of `_handle_interrupt()`, after assigning
`pending_interrupt`, add:

```python
_update_job(status="waiting", pending_interrupt=payload)
```

In `_safe_stream_updates()`, update both exception handlers:

```python
except RuntimeError as exc:
    st.session_state["last_run_failed"] = True
    _update_job(status="failed")
    st.error(str(exc))
    st.info(missing_key_message("OPENAI_API_KEY"))
except Exception as exc:
    st.session_state["last_run_failed"] = True
    _update_job(status="failed")
    st.exception(exc)
```

Immediately before the first `_append_ui_message()` in the chat submission
block, add:

```python
_ensure_job_record(graph_text, verifier_mode)
_update_job(status="running", pending_interrupt=None)
st.session_state["last_run_failed"] = False
```

After the stream loop, when no interrupt is pending and no run failure was
recorded, persist completion:

```python
if (
    st.session_state.get("pending_interrupt") is None
    and not st.session_state.get("last_run_failed")
):
    report_paths = [str(path) for path in _report_paths_from_state()]
    _update_job(
        status="completed",
        pending_interrupt=None,
        report_paths=report_paths,
    )
```

- [ ] **Step 8: Add Store report-path fallback**

In `_report_paths_from_state()`, before the compatibility fallback, add:

```python
record = _current_job()
if not candidates and record:
    candidates.extend(Path(path) for path in record.get("report_paths") or [])
```

- [ ] **Step 9: Review the complete Streamlit change**

Run:

```bash
git diff -- app.py
rg -n "MemorySaver|_compile_workflow|历史任务|_restore_job|status=\"waiting\"|status=\"completed\"" app.py
```

Expected: `MemorySaver` is absent, compilation passes both persistence
objects, and history/status integration is present.

- [ ] **Step 10: Commit the Streamlit integration**

```bash
git add app.py
git commit -m "feat: restore persisted Streamlit jobs"
```

---

### Task 4: Add durable thread recovery to the CLI

**Files:**
- Modify: `run.py`

**Interfaces:**
- Consumes: `SQLitePersistence`, `JobStore`, `get_local_user_id()`, and
  `interrupt_from_snapshot()`.
- Produces: `--thread-id` and `--user-id` recovery behavior.

- [ ] **Step 1: Replace imports and add CLI arguments**

Remove the `MemorySaver` import and add:

```python
from src.config import (
    configure_langsmith_from_env,
    get_local_user_id,
    missing_key_message,
)
from src.job_store import JobStore, interrupt_from_snapshot
from src.persistence import SQLitePersistence
```

Add parser arguments:

```python
parser.add_argument("--thread-id", help="Resume an existing LangGraph thread")
parser.add_argument("--user-id", help="Override AGENT_USER_ID for Store scope")
```

- [ ] **Step 2: Resolve stable CLI scope**

Replace the unconditional UUID assignment with:

```python
thread_id = args.thread_id or f"job_{uuid.uuid4().hex}"
user_id = args.user_id or get_local_user_id()
conversation_id = f"conv_cli_{thread_id}"
```

- [ ] **Step 3: Compile with SQLite and detect a persisted interrupt**

Replace the `MemorySaver` block with:

```python
persistence = SQLitePersistence.open()
jobs = JobStore(persistence.store)
app = workflow.compile(
    checkpointer=persistence.checkpointer,
    store=persistence.store,
)

config = {
    "configurable": {"thread_id": thread_id},
    "metadata": {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "job_id": thread_id,
    },
}
snapshot = app.get_state(config)
last_interrupt_value = interrupt_from_snapshot(snapshot)
is_interrupted = last_interrupt_value is not None
job_record_created = jobs.get_job(user_id, thread_id) is not None
```

- [ ] **Step 4: Create and update the CLI job record**

After reading a non-empty, non-quit user input and before streaming, add:

```python
if not job_record_created:
    jobs.create_job(
        user_id=user_id,
        conversation_id=conversation_id,
        job_id=thread_id,
        title=user_input,
        verifier_mode="auto" if args.auto_verify else "manual",
    )
    job_record_created = True

jobs.update_job(
    user_id,
    thread_id,
    status="running",
    pending_interrupt=None,
)
```

When an interrupt is found, add:

```python
jobs.update_job(
    user_id,
    thread_id,
    status="waiting",
    pending_interrupt=last_interrupt_value,
)
```

After a stream completes without an interrupt, add:

```python
if not is_interrupted:
    jobs.update_job(
        user_id,
        thread_id,
        status="completed",
        pending_interrupt=None,
    )
```

At the start of each exception handler, add:

```python
if job_record_created:
    jobs.update_job(user_id, thread_id, status="failed")
```

- [ ] **Step 5: Close SQLite resources on exit**

Insert `try:` immediately before the existing `while True:` input loop and
indent that complete loop by one level:

```python
try:
    while True:
        # The existing input, resume, stream, and status-update statements
        # remain in this loop at one additional indentation level.
        user_input = input(f"\nUser [{thread_id}]> ").strip()
        # Continue with the existing loop body from the input handling.
```

Append the matching cleanup block immediately after the loop:

```python
finally:
    persistence.close()
```

- [ ] **Step 6: Review the CLI change**

Run:

```bash
git diff -- run.py
rg -n "MemorySaver|thread-id|user-id|SQLitePersistence|interrupt_from_snapshot" run.py
```

Expected: the CLI has no in-memory saver and can select an existing thread.

- [ ] **Step 7: Commit the CLI integration**

```bash
git add run.py
git commit -m "feat: resume persisted CLI jobs"
```

---

### Task 5: Document the delivered persistence behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/development-workflow.md`

**Interfaces:**
- Consumes: The completed Streamlit and CLI behavior.
- Produces: Operator-facing setup, storage-path, recovery, and branch guidance.

- [ ] **Step 1: Update README persistence guidance**

Add to Highlights:

```markdown
- SQLite-backed checkpoint and Store persistence with historical job recovery.
```

Add after Quick Start:

```markdown
### Local persistence

LangGraph checkpoints and job history are stored under `cache/langgraph/`.
Set `AGENT_USER_ID` to a stable authenticated identifier when moving beyond
the default single-user `local-user` setup. Keep
`LANGGRAPH_STRICT_MSGPACK=true` enabled.

Resume a CLI thread with:

```bash
python run.py --thread-id job_<id>
```
```

Remove the completed Future Improvements item:

```markdown
- Add persistent checkpoint storage beyond in-memory debugging.
```

- [ ] **Step 2: Extend the architecture document**

Append to `docs/architecture.md`:

```markdown
## Persistence

`job_id` is the LangGraph `thread_id`. `SqliteSaver` stores thread-level graph
state and interrupts in `cache/langgraph/checkpoints.sqlite`. `SqliteStore`
stores user-scoped job records, UI messages, statuses, and report paths in
`cache/langgraph/store.sqlite`.

Streamlit restores historical jobs only after an explicit user selection.
Checkpoint state is authoritative for workflow execution; Store records are
the discovery and presentation layer. The CLI accepts `--thread-id` to reopen
the same checkpoint.
```

- [ ] **Step 3: Correct the development branch example**

Replace `codex/sqlite-checkpoint` with:

```text
codex/sqlite-checkpoint-store
```

throughout `docs/development-workflow.md`.

- [ ] **Step 4: Inspect final scope without running tests**

Run:

```bash
git status --short
git diff --check
git diff --stat main...HEAD
rg -n "MemorySaver" app.py run.py src README.md docs
git check-ignore -v cache/langgraph/checkpoints.sqlite cache/langgraph/store.sqlite
```

Expected:

- No whitespace errors.
- No `MemorySaver` reference in application code.
- Both SQLite files match existing ignore rules.
- Only persistence-related source, configuration, and documentation changed.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/architecture.md docs/development-workflow.md
git commit -m "docs: explain SQLite job recovery"
```

- [ ] **Step 6: Review branch history**

Run:

```bash
git status --short --branch
git log --oneline --decorate main..HEAD
```

Expected: a clean feature branch containing the design commit and focused
configuration, persistence, Streamlit, CLI, and documentation commits.

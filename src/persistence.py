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

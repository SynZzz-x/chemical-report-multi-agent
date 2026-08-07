from src.nodes.artifact_commit import artifact_commit


class RecordingStore:
    def __init__(self):
        self.calls = []

    def put(self, namespace, key, value, *, index=None):
        self.calls.append((namespace, key, value, index))


def _state():
    return {
        "user_id": "u1",
        "job_id": "j1",
        "current_execution_id": "execution-2",
        "current_result": {
            "task_id": "T1",
            "text_output": "revision",
            "citations": [],
        },
        "task_records": {
            "T1": {
                "task_id": "T1",
                "attempt_count": 2,
                "status": "RUNNING",
            }
        },
        "artifacts": {
            "artifact-old": {
                "artifact_id": "artifact-old",
                "task_id": "T1",
            }
        },
        "active_artifact_ids": {"T1": "artifact-old"},
    }


def test_artifact_commit_versions_without_overwriting_history():
    state = _state()

    update = artifact_commit(state)
    artifact = update["current_result"]

    assert artifact["supersedes"] == "artifact-old"
    assert update["active_artifact_ids"]["T1"] == artifact["artifact_id"]
    assert set(update["artifacts"]) == {"artifact-old", artifact["artifact_id"]}
    assert update["task_records"]["T1"]["active_artifact_id"] == artifact["artifact_id"]


def test_replayed_commit_reuses_existing_artifact_record():
    first = artifact_commit(_state())
    replay_state = {**_state(), **first}

    replay = artifact_commit(replay_state)

    assert replay["current_result"] == first["current_result"]
    assert replay["current_result"]["supersedes"] == "artifact-old"
    assert len(replay["artifacts"]) == 2


def test_artifact_commit_mirrors_record_to_job_store():
    store = RecordingStore()

    update = artifact_commit(_state(), store=store)

    artifact = update["current_result"]
    assert store.calls == [
        (
            ("u1", "report_jobs", "j1", "artifacts"),
            artifact["artifact_id"],
            artifact,
            False,
        )
    ]


def test_artifact_commit_rejects_unscoped_worker_result():
    state = _state()
    state["current_result"] = {"text_output": "missing task"}

    try:
        artifact_commit(state)
    except ValueError as exc:
        assert "task-scoped" in str(exc)
    else:
        raise AssertionError("missing task_id must be rejected")

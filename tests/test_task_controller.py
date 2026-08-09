from src.nodes.task_controller import route_task_controller, task_controller


class RecordingStore:
    def __init__(self):
        self.puts = []

    def put(self, namespace, key, value, **kwargs):
        self.puts.append((namespace, key, value, kwargs))


def _state(count=9):
    return {
        "job_id": "job-1",
        "tasks": [
            {"task_id": f"T{i}", "task_name": f"Task {i}"}
            for i in range(1, count + 1)
        ],
        "results": [],
        "task_records": {},
    }


def test_controller_selects_first_unpassed_task_without_planner_messages():
    state = _state()

    first = task_controller(state)

    assert first["current_task"]["task_id"] == "T1"
    assert first["cursor"] == 0
    assert first["task_records"]["T1"]["attempt_count"] == 1
    assert first["task_records"]["T1"]["status"] == "RUNNING"
    assert route_task_controller({**state, **first}) == "Worker"


def test_controller_uses_ledger_to_select_next_task():
    state = _state(3)
    state["task_records"] = {
        "T1": {"task_id": "T1", "sequence": 0, "status": "PASSED", "attempt_count": 1},
        "T2": {"task_id": "T2", "sequence": 1, "status": "PENDING", "attempt_count": 0},
        "T3": {"task_id": "T3", "sequence": 2, "status": "PENDING", "attempt_count": 0},
    }

    update = task_controller(state)

    assert update["current_task"]["task_id"] == "T2"
    assert update["cursor"] == 1
    assert update["task_records"]["T2"]["attempt_count"] == 1


def test_controller_never_summarizes_with_a_blocked_task():
    state = _state(3)
    state["task_records"] = {
        "T1": {"task_id": "T1", "sequence": 0, "status": "PASSED", "attempt_count": 1},
        "T2": {"task_id": "T2", "sequence": 1, "status": "BLOCKED", "attempt_count": 2},
        "T3": {"task_id": "T3", "sequence": 2, "status": "PENDING", "attempt_count": 0},
    }

    update = task_controller(state)

    assert update["controller_action"] == "NEEDS_USER_INPUT"
    assert update["current_task"]["task_id"] == "T2"
    assert route_task_controller({**state, **update}) == "NeedsUserInput"


def test_controller_summarizes_only_when_every_task_passed():
    state = _state(2)
    state["task_records"] = {
        "T1": {"task_id": "T1", "sequence": 0, "status": "PASSED", "attempt_count": 1},
        "T2": {"task_id": "T2", "sequence": 1, "status": "PASSED", "attempt_count": 1},
    }

    update = task_controller(state)

    assert update["controller_action"] == "SUMMARIZE"
    assert route_task_controller({**state, **update}) == "Summarizer"


def test_replaying_controller_input_reuses_execution_identity():
    state = _state(1)

    first = task_controller(state)
    replay = task_controller(state)

    assert first["current_execution_id"] == replay["current_execution_id"]


def test_controller_mirrors_legacy_migration_records_to_store():
    state = _state(1)
    state.update(
        {
            "user_id": "user-1",
            "results": [
                {
                    "task_id": "T1",
                    "text_output": "accepted legacy output",
                    "citations": [],
                }
            ],
        }
    )
    store = RecordingStore()

    update = task_controller(state, store=store)

    assert update["controller_action"] == "SUMMARIZE"
    assert {namespace[-1] for namespace, *_ in store.puts} == {
        "artifacts",
        "reviews",
    }

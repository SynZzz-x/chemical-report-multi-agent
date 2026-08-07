from src.workflow_records import (
    all_tasks_passed,
    build_artifact,
    ensure_task_records,
    first_runnable_task,
    make_execution_id,
    set_task_status,
)


def test_legacy_results_bootstrap_passed_task_records():
    state = {
        "tasks": [{"task_id": "T1"}, {"task_id": "T2"}],
        "results": [{"task_id": "T1", "text_output": "accepted"}],
    }

    records = ensure_task_records(state)

    assert records["T1"]["status"] == "PASSED"
    assert records["T2"]["status"] == "PENDING"
    assert first_runnable_task(state["tasks"], records) == (1, state["tasks"][1])
    assert all_tasks_passed(state["tasks"], records) is False


def test_legacy_result_wins_over_stale_running_record():
    state = {
        "tasks": [{"task_id": "T1"}],
        "results": [{"task_id": "T1", "text_output": "accepted"}],
        "task_records": {
            "T1": {"status": "RUNNING", "attempt_count": 2},
        },
    }

    records = ensure_task_records(state)

    assert records["T1"]["status"] == "PASSED"
    assert records["T1"]["attempt_count"] == 2
    assert all_tasks_passed(state["tasks"], records) is True


def test_dependencies_must_pass_before_task_is_runnable():
    tasks = [
        {"task_id": "T1"},
        {"task_id": "T2", "dependencies": ["T1"]},
    ]
    records = ensure_task_records({"tasks": tasks, "results": []})
    records = set_task_status(records, "T1", "BLOCKED")

    assert first_runnable_task(tasks, records) == (0, tasks[0])


def test_artifact_identity_is_stable_for_a_replayed_attempt():
    state = {
        "job_id": "job-1",
        "current_execution_id": make_execution_id("job-1", "T2", 3),
        "task_records": {"T2": {"attempt_count": 3}},
        "active_artifact_ids": {"T2": "artifact-old"},
    }
    current = {
        "task_id": "T2",
        "text_output": "new",
        "citations": [{"evidence_id": "E1"}],
        "sources_used": ["process.docx"],
    }

    first = build_artifact(state, current)
    second = build_artifact(state, current)

    assert first["artifact_id"] == second["artifact_id"]
    assert first["attempt_no"] == 3
    assert first["supersedes"] == "artifact-old"
    assert first["content"] == "new"
    assert first["evidence_refs"] == [{"evidence_id": "E1"}]

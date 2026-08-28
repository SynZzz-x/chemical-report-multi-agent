from src.job_outcome import derive_job_status, project_job_outcome


def test_fatal_checkpoint_normal_return_projects_failed_with_safe_diagnostic():
    state = {
        "workflow_action": "FATAL_SYSTEM",
        "failure_decision": {
            "failure_class": "FATAL_SYSTEM",
            "action": "FAIL_JOB",
            "subtype": "VERIFIER_UNAVAILABLE",
        },
        "fatal_system_error": {
            "failure_id": "fatal-1",
            "failure_class": "FATAL_SYSTEM",
            "subtype": "VERIFIER_UNAVAILABLE",
            "origin": "graph",
            "component": "Verifier",
            "operation": "assessment",
            "task_id": "T1",
            "diagnostic_code": "VERIFIER_UNAVAILABLE",
            "retryable": False,
            "metadata": {"attempts": 2},
            "raw_exception": "secret token",
        },
    }

    projection = project_job_outcome(state, pending_interrupt=None)

    assert projection["status"] == "failed"
    assert projection["fatal_system_error"]["diagnostic_code"] == (
        "VERIFIER_UNAVAILABLE"
    )
    assert "raw_exception" not in projection["fatal_system_error"]
    assert "secret token" not in str(projection)


def test_normal_done_checkpoint_projects_completed():
    assert derive_job_status({"workflow_action": "DONE"}, None) == "completed"


def test_nonfatal_projection_clears_stale_fatal_diagnostic():
    projection = project_job_outcome({"workflow_action": "DONE"}, None)

    assert projection == {
        "status": "completed",
        "pending_interrupt": None,
        "fatal_system_error": None,
    }


def test_pending_interrupt_or_canonical_blocker_projects_waiting_before_fatal():
    fatal = {
        "workflow_action": "FATAL_SYSTEM",
        "failure_decision": {"failure_class": "FATAL_SYSTEM"},
    }
    assert derive_job_status(fatal, {"type": "needs_user_input"}) == "waiting"
    assert (
        derive_job_status(
            {
                "workflow_action": "NEEDS_USER_INPUT",
                "pending_user_blockers": [{"status": "pending"}],
            },
            None,
        )
        == "waiting"
    )


def test_fail_job_action_is_a_fatal_marker_without_python_exception():
    state = {
        "failure_decision": {
            "failure_class": "FATAL_SYSTEM",
            "action": "FAIL_JOB",
            "subtype": "VERIFIER_UNAVAILABLE",
            "task_id": "T1",
        }
    }

    assert derive_job_status(state, None) == "failed"
    projection = project_job_outcome(state, None)
    assert projection["fatal_system_error"]["diagnostic_code"] == (
        "VERIFIER_UNAVAILABLE"
    )
    assert projection["fatal_system_error"]["component"] == "Workflow"


def test_unreadable_or_incomplete_final_checkpoint_cannot_project_completed():
    assert derive_job_status({}, None, graph_incomplete=True) == "failed"

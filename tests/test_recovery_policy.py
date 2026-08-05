from src.recovery.policy import (
    IssueCategory,
    WorkflowAction,
    classify_assessment,
    commit_current_result,
    decide_recovery_action,
)


def recovery_state(
    *,
    task_id="T1",
    docs=None,
    task_resources=None,
    task_retry_count=None,
    evidence_recovery_count=None,
    task_patch_count=None,
    job_patch_count=0,
    results=None,
):
    return {
        "tasks": [{"task_id": task_id, "use_resources": task_resources or []}],
        "cursor": 0,
        "docs": list(docs or []),
        "current_result": {"task_id": task_id, "text_output": "current result"},
        "results": list(results or []),
        "task_retry_count": dict(task_retry_count or {}),
        "evidence_recovery_count": dict(evidence_recovery_count or {}),
        "task_patch_count": dict(task_patch_count or {}),
        "job_patch_count": job_patch_count,
        "verification_warnings": [],
    }


def assessment_with(code, category, **issue):
    return {
        "status": "FAILED",
        "issues": [{"code": code, "category": category, **issue}],
    }


def test_evidence_gap_recovers_once_then_requests_user_input():
    state = recovery_state(task_id="T2")
    assessment = assessment_with("EVIDENCE_GAP", "EVIDENCE_GAP")
    first = decide_recovery_action(state, assessment)
    assert first["workflow_action"] == "EVIDENCE_RECOVERY"
    assert first["evidence_recovery_count"] == {"T2": 1}

    second_state = {**state, **first}
    second = decide_recovery_action(second_state, assessment)
    assert second["workflow_action"] == "NEEDS_USER_INPUT"
    assert second["pending_user_action"]["category"] == "EVIDENCE_GAP"


def test_missing_resource_without_available_match_is_external_blocker():
    state = recovery_state(task_id="T2", docs=[])
    decision = decide_recovery_action(
        state,
        assessment_with("MISSING_RESOURCE", None, resource_name="missing.csv"),
    )
    assert decision["workflow_action"] == "NEEDS_USER_INPUT"


def test_available_but_unassigned_resource_is_local_plan_defect():
    state = recovery_state(
        task_id="T2",
        docs=[{"name": "data.csv", "path": "/job/data.csv"}],
        task_resources=[],
    )
    decision = decide_recovery_action(
        state,
        assessment_with("MISSING_RESOURCE", None, resource_name="data.csv"),
    )
    assert decision["workflow_action"] == "PLAN_PATCH"


def test_content_retry_limit_accepts_with_warning_and_commits_result():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})
    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )
    assert decision["workflow_action"] == "ACCEPT_WITH_WARNING"
    assert decision["results"][-1]["task_id"] == "T2"


def test_classification_uses_priority_and_never_treats_evidence_as_plan_defect():
    state = recovery_state(task_id="T2")
    assessment = {
        "status": "BLOCKED",
        "issues": [
            {"code": "EVIDENCE_GAP"},
            {"code": "UNEXECUTABLE_TASK"},
            {"code": "MISSING_RESOURCE", "resource_name": "missing.csv"},
        ],
    }

    assert classify_assessment(assessment, state) is IssueCategory.EXTERNAL_BLOCKER
    assert (
        classify_assessment(assessment_with("MISSING_EVIDENCE", None), state)
        is IssueCategory.EVIDENCE_GAP
    )


def test_legacy_cursor_counter_keys_are_read_and_written_with_task_ids():
    state = recovery_state(task_id="T2", task_retry_count={0: 1})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.REWORK
    assert decision["task_retry_count"] == {"T2": 2}


def test_pass_commits_current_result_once_and_uses_done_at_final_task():
    state = recovery_state(task_id="T2")

    first = decide_recovery_action(state, {"status": "PASS", "issues": []})
    second = decide_recovery_action({**state, **first}, {"status": "PASS", "issues": []})

    assert first["workflow_action"] == WorkflowAction.DONE
    assert [result["task_id"] for result in first["results"]] == ["T2"]
    assert [result["task_id"] for result in second["results"]] == ["T2"]
    assert [result["task_id"] for result in commit_current_result({**state, **first})] == ["T2"]

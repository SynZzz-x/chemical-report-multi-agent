import pytest

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
    verifier_retry_count=None,
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
        "verifier_retry_count": dict(verifier_retry_count or {}),
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
    assert second["pending_user_action"]["accepted_choices"] == [
        "UPLOAD_RESOURCES",
        "AUTHORIZE_WEB",
        "ADJUST_REQUIREMENT",
        "ACCEPT_EVIDENCE_GAP",
    ]
    assert "上传" in second["pending_user_action"]["guidance"]


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


def test_json_restored_numeric_string_counter_preserves_content_retry_cap():
    state = recovery_state(task_id="T2", task_retry_count={"0": 2})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.ACCEPT_WITH_WARNING
    assert decision["task_retry_count"] == {"T2": 2}


def test_numeric_string_that_is_a_real_task_id_is_not_treated_as_a_cursor():
    state = recovery_state(task_id="0", task_retry_count={"0": 2})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.ACCEPT_WITH_WARNING
    assert decision["task_retry_count"] == {"0": 2}


@pytest.mark.parametrize(
    "counter_items",
    [
        [("0", 2), ("T1", 1)],
        [("T1", 1), ("0", 2)],
    ],
)
@pytest.mark.parametrize(
    ("counter_field", "assessment", "expected_action"),
    [
        (
            "task_retry_count",
            assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
            WorkflowAction.ACCEPT_WITH_WARNING,
        ),
        (
            "evidence_recovery_count",
            assessment_with("MISSING_EVIDENCE", "EVIDENCE_GAP"),
            WorkflowAction.NEEDS_USER_INPUT,
        ),
        (
            "verifier_retry_count",
            assessment_with("ASSESSMENT_CONTRACT_ERROR", "VERIFIER_FAILURE"),
            WorkflowAction.NEEDS_USER_INPUT,
        ),
    ],
)
def test_json_counter_aliases_merge_by_max_without_reopening_caps(
    counter_items, counter_field, assessment, expected_action
):
    state = recovery_state(task_id="T1")
    state[counter_field] = dict(counter_items)

    decision = decide_recovery_action(state, assessment)

    assert decision["workflow_action"] == expected_action
    assert decision[counter_field] == {"T1": 2}


def test_pass_commits_current_result_once_and_uses_done_at_final_task():
    state = recovery_state(task_id="T2")

    first = decide_recovery_action(state, {"status": "PASS", "issues": []})
    second = decide_recovery_action({**state, **first}, {"status": "PASS", "issues": []})

    assert first["workflow_action"] == WorkflowAction.DONE
    assert [result["task_id"] for result in first["results"]] == ["T2"]
    assert [result["task_id"] for result in second["results"]] == ["T2"]
    assert [result["task_id"] for result in commit_current_result({**state, **first})] == ["T2"]


def test_non_final_pass_commits_current_result_and_returns_next():
    state = recovery_state(task_id="T2")
    state["tasks"].append({"task_id": "T3", "use_resources": []})

    decision = decide_recovery_action(state, {"status": "PASS", "issues": []})

    assert decision["workflow_action"] == WorkflowAction.NEXT
    assert [result["task_id"] for result in decision["results"]] == ["T2"]


def test_plan_patch_per_task_limit_requires_user_input_without_consuming_counts():
    state = recovery_state(task_id="T2", task_patch_count={"T2": 1})

    decision = decide_recovery_action(
        state,
        assessment_with("RESOURCE_NOT_ASSIGNED", None),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert decision["task_patch_count"] == {"T2": 1}
    assert decision["job_patch_count"] == 0


def test_plan_patch_per_job_limit_requires_user_input_without_consuming_counts():
    state = recovery_state(task_id="T2", job_patch_count=3)

    decision = decide_recovery_action(
        state,
        assessment_with("RESOURCE_NOT_ASSIGNED", None),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert decision["task_patch_count"] == {}
    assert decision["job_patch_count"] == 3


def test_plan_patch_decision_does_not_increment_applied_patch_counts():
    state = recovery_state(
        task_id="T2",
        docs=[{"name": "data.csv", "path": "/job/data.csv"}],
    )

    decision = decide_recovery_action(
        state,
        assessment_with("MISSING_RESOURCE", None, resource_name="data.csv"),
    )

    assert decision["workflow_action"] == WorkflowAction.PLAN_PATCH
    assert decision["task_patch_count"] == {}
    assert decision["job_patch_count"] == 0


def test_retry_limit_warning_has_structured_fields_and_is_idempotent():
    previous_warning = {"code": "EARLIER_WARNING", "task_id": "T1"}
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})
    state["verification_warnings"] = [previous_warning]
    assessment = assessment_with("TOO_SHORT", "CONTENT_DEFECT")

    first = decide_recovery_action(state, assessment)
    second = decide_recovery_action({**state, **first}, assessment)

    warning = first["verification_warning"]
    assert warning["code"] == "CONTENT_RETRY_LIMIT_REACHED"
    assert warning["category"] == "CONTENT_DEFECT"
    assert warning["task_id"] == "T2"
    assert warning["issues"] == assessment["issues"]
    assert first["verification_warnings"] == [previous_warning, warning]
    assert second["verification_warning"] == warning
    assert second["verification_warnings"] == [previous_warning, warning]


def test_only_explicit_auto_fixable_codes_are_local_plan_defects():
    state = recovery_state(task_id="T2")
    for code in (
        "CONTRADICTORY_REQUIREMENTS",
        "REQUIREMENTS_CONFLICT",
        "UNEXECUTABLE_TASK",
        "BAD_PLAN",
        "INVALID_PLAN",
    ):
        assert (
            classify_assessment(assessment_with(code, None), state)
            is IssueCategory.EXTERNAL_BLOCKER
        )

    for code in (
        "RESOURCE_NOT_ASSIGNED",
        "INVALID_TASK_ORDER",
        "MISSING_DEPENDENCY",
        "TASK_GRANULARITY",
    ):
        assert (
            classify_assessment(assessment_with(code, None), state)
            is IssueCategory.LOCAL_PLAN_DEFECT
        )


def test_unknown_code_cannot_use_llm_local_plan_defect_category():
    state = recovery_state(task_id="T2")
    assessment = assessment_with("UNRECOGNIZED_CODE", "LOCAL_PLAN_DEFECT")

    assert classify_assessment(assessment, state) is IssueCategory.EXTERNAL_BLOCKER
    assert (
        decide_recovery_action(state, assessment)["workflow_action"]
        == WorkflowAction.NEEDS_USER_INPUT
    )


def test_blank_code_cannot_use_llm_local_plan_defect_category():
    state = recovery_state(task_id="T2")
    assessment = assessment_with("", "LOCAL_PLAN_DEFECT")

    assert classify_assessment(assessment, state) is IssueCategory.EXTERNAL_BLOCKER
    assert (
        decide_recovery_action(state, assessment)["workflow_action"]
        == WorkflowAction.NEEDS_USER_INPUT
    )


def test_content_retry_overflow_continues_next_for_a_non_final_task():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})
    state["tasks"].append({"task_id": "T3", "use_resources": []})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.ACCEPT_WITH_WARNING
    assert decision["continuation_action"] == WorkflowAction.NEXT


def test_content_retry_overflow_continues_done_for_a_final_task():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.ACCEPT_WITH_WARNING
    assert decision["continuation_action"] == WorkflowAction.DONE

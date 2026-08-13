import pytest

from src.report_acceptance import (
    ACCEPT_WITH_WARNING,
    BLOCKED,
    DRAFT_WITH_GAPS,
    EXTERNAL_BLOCKER,
    READY_FOR_FINAL,
    USER_ACCEPTED_GAP,
    USER_ACCEPTED_WARNING,
    VERIFIED_PASS,
    derive_report_status,
    delivery_path_candidates,
    eligible_task_ids,
    record_section_status,
)


TASKS = [{"task_id": "T1"}, {"task_id": "T2"}, {"task_id": "T3"}]


def _status(value):
    return {
        "status": value,
        "accepted_by": "user" if value.startswith("USER_") else "verifier",
        "issues": [],
        "plan_revision": 1,
        "task_revision": 1,
    }


def test_all_verified_sections_are_ready_for_final_report():
    statuses = {task["task_id"]: _status(VERIFIED_PASS) for task in TASKS}

    assert derive_report_status(TASKS, statuses) == READY_FOR_FINAL
    assert eligible_task_ids(TASKS, statuses, READY_FOR_FINAL) == ["T1", "T2", "T3"]


def test_user_accepted_gap_produces_ordered_draft_sections():
    statuses = {
        "T1": _status(VERIFIED_PASS),
        "T2": _status(USER_ACCEPTED_GAP),
        "T3": _status(USER_ACCEPTED_WARNING),
    }

    assert derive_report_status(TASKS, statuses) == DRAFT_WITH_GAPS
    assert eligible_task_ids(TASKS, statuses, DRAFT_WITH_GAPS) == ["T1", "T2", "T3"]


def test_automatic_warning_and_blockers_prevent_report_generation():
    for blocking_status in (ACCEPT_WITH_WARNING, BLOCKED, EXTERNAL_BLOCKER):
        statuses = {
            "T1": _status(VERIFIED_PASS),
            "T2": _status(blocking_status),
            "T3": _status(VERIFIED_PASS),
        }

        assert derive_report_status(TASKS, statuses) == BLOCKED
        assert eligible_task_ids(TASKS, statuses, BLOCKED) == []


def test_missing_section_status_is_blocked_even_when_results_exist_elsewhere():
    statuses = {"T1": _status(VERIFIED_PASS), "T3": _status(VERIFIED_PASS)}

    assert derive_report_status(TASKS, statuses) == BLOCKED


def test_record_section_status_binds_acceptance_to_current_revisions():
    state = {
        "tasks": TASKS,
        "cursor": 1,
        "plan_revision": 4,
        "task_revisions": {"T2": 3},
        "section_status": {"T1": _status(VERIFIED_PASS)},
    }
    issues = [{"code": "EVIDENCE_GAP", "description": "缺少检测标准"}]

    statuses = record_section_status(
        state,
        USER_ACCEPTED_GAP,
        accepted_by="user",
        issues=issues,
    )

    assert statuses["T1"] == _status(VERIFIED_PASS)
    assert statuses["T2"] == {
        "status": USER_ACCEPTED_GAP,
        "accepted_by": "user",
        "issues": issues,
        "plan_revision": 4,
        "task_revision": 3,
    }


def test_authoritative_blocked_result_never_exposes_stale_report_paths():
    assert delivery_path_candidates(
        {"report_status": "BLOCKED", "attachments": [], "path": None},
        stored_paths=["/old/report.docx"],
        fallback_paths=["/cache/report.docx"],
    ) == []


def test_authoritative_renderer_failure_does_not_fall_back_to_old_artifacts():
    assert delivery_path_candidates(
        {"report_status": "READY_FOR_FINAL", "attachments": [], "path": None},
        stored_paths=["/old/report.docx"],
        fallback_paths=["/cache/report.docx"],
    ) == []


def test_legacy_state_can_still_use_recorded_report_paths():
    assert delivery_path_candidates(
        {},
        stored_paths=["/old/report.docx"],
        fallback_paths=["/cache/report.docx"],
    ) == ["/old/report.docx"]


@pytest.mark.parametrize(
    "final_result",
    [
        {"success": False, "status": "cancelled"},
        {"status": "failed"},
        {"delivery_status": "FAILED"},
    ],
)
def test_failed_or_cancelled_result_never_exposes_stale_paths(final_result):
    assert delivery_path_candidates(
        final_result,
        stored_paths=["/old/report.docx"],
        fallback_paths=["/cache/report.docx"],
    ) == []


def test_user_accepted_status_requires_explicit_user_actor():
    statuses = {
        "T1": _status(VERIFIED_PASS),
        "T2": {**_status(USER_ACCEPTED_GAP), "accepted_by": "system"},
        "T3": _status(VERIFIED_PASS),
    }

    assert derive_report_status(TASKS, statuses) == BLOCKED

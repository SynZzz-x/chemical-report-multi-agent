import logging
from pathlib import Path

from src.fatal_errors import build_fatal_system_error
from src.recovery.policy import decide_recovery_action


def test_runner_fatal_record_is_safe_and_schema_exact():
    secret = "secret database path and token"
    record = build_fatal_system_error(
        RuntimeError(secret),
        origin="runner",
        component="SQLitePersistence",
        operation="checkpoint_write",
        task_id=None,
        metadata={"job_id": "job-1"},
    )

    assert set(record) == {
        "failure_id",
        "failure_class",
        "subtype",
        "origin",
        "component",
        "operation",
        "task_id",
        "diagnostic_code",
        "retryable",
        "metadata",
    }
    assert record["origin"] == "runner"
    assert record["subtype"] == "RuntimeError"
    assert record["retryable"] is False
    assert secret not in str(record)


def test_policy_log_contains_attribution_without_issue_body(caplog):
    secret = "sensitive issue prose"
    state = {
        "tasks": [{"task_id": "T1", "use_rag": True, "requirement_ids": [], "depends_on_task_ids": []}],
        "cursor": 0,
        "current_result": {"task_id": "T1", "text_output": "partial"},
        "results": [],
        "task_retry_count": {},
        "asset_retry_count": {},
        "evidence_recovery_count": {"T1": 1},
        "task_patch_count": {},
        "verifier_retry_count": {},
        "verification_warnings": [],
    }
    assessment = {
        "status": "FAILED",
        "issues": [{"code": "EVIDENCE_GAP", "description": secret}],
    }

    with caplog.at_level(logging.INFO, logger="src.recovery.policy"):
        decide_recovery_action(state, assessment)

    message = next(
        record.getMessage()
        for record in caplog.records
        if "FAILURE_POLICY_DECISION" in record.getMessage()
    )
    assert "task_id=T1" in message
    assert "failure_class=DEGRADABLE_QUALITY" in message
    assert "policy_action=COMMIT_WITH_WARNING" in message
    assert secret not in message


def test_legacy_policy_transition_persists_safe_failure_state_defaults():
    state = {
        "tasks": [{"task_id": "T1", "use_rag": True}],
        "cursor": 0,
        "current_result": {"task_id": "T1", "text_output": "ok"},
        "results": [],
        "messages": [],
    }

    update = decide_recovery_action(state, {"status": "PASS", "issues": []})

    assert update["requirement_registry"] == []
    assert update["degraded_issue_registry"] == []
    assert update["pending_user_blockers"] == []
    assert update["blocker_resolution_registry"] == []
    assert update["resolved_user_blocker_ids"] == []
    assert update["fatal_system_error"] == {}


def test_runner_fatal_handlers_do_not_restore_blockers_or_render_exception_text():
    root = Path(__file__).parents[1]
    app_source = (root / "app.py").read_text(encoding="utf-8")
    run_source = (root / "run.py").read_text(encoding="utf-8")

    recover_body = app_source.split("def _recover_stream_failure", 1)[1].split(
        "def _safe_stream_updates", 1
    )[0]
    assert 'st.session_state["pending_interrupt"] = None' in recover_body
    assert "interrupt_from_snapshot" not in recover_body
    assert "st.exception(exc)" not in app_source
    assert "fatal_system_error=fatal" in app_source
    assert "fatal_system_error=fatal" in run_source

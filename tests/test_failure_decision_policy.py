from src.failure_semantics import FailureAction, FailureClass
from src.recovery.policy import MAX_ASSET_RETRIES, decide_recovery_action
import pytest


def _state(
    *,
    code: str,
    requirement_severity: str = "soft",
    requirement_kind: str = "quality",
):
    return {
        "tasks": [
            {
                "task_id": "T1",
                "task_description": "正文不超过10字。",
                "task_type": "analysis",
                "use_rag": True,
                "use_web": False,
                "use_resources": [],
                "requirement_ids": ["REQ-001"],
                "depends_on_task_ids": [],
            }
        ],
        "cursor": 0,
        "current_result": {
            "task_id": "T1",
            "text_output": "不可安全截断的完整事实陈述[E1]。",
        },
        "results": [],
        "section_status": {},
        "task_retry_count": {},
        "asset_retry_count": {},
        "evidence_recovery_count": {},
        "task_patch_count": {},
        "verifier_retry_count": {},
        "job_patch_count": 0,
        "length_rewrite_attempts": {},
        "task_revisions": {"T1": 1},
        "verification_warnings": [],
        "requirement_registry": [
            {
                "requirement_id": "REQ-001",
                "text": f"requirement for {code}",
                "severity": requirement_severity,
                "kind": requirement_kind,
                "status": "active",
                "contract_revision": 1,
                "provenance": {
                    "origin": "explicit_user",
                    "source_message_id": "msg-1",
                    "source_field": "constraints",
                    "source_index": 0,
                    "derivation": "test",
                },
            }
        ],
    }


def _assessment(code: str):
    return {
        "status": "FAILED",
        "issues": [
            {
                "code": code,
                "category": (
                    "EVIDENCE_GAP"
                    if code == "EVIDENCE_GAP"
                    else "CONTENT_DEFECT"
                ),
                "requirement_ids": ["REQ-001"],
            }
        ],
    }


def test_evidence_recovery_publishes_canonical_retry_decision():
    update = decide_recovery_action(_state(code="EVIDENCE_GAP"), _assessment("EVIDENCE_GAP"))

    assert update["failure_decision"]["failure_class"] == FailureClass.RETRYABLE_EXECUTION
    assert update["failure_decision"]["action"] == FailureAction.RECOVER_EVIDENCE
    assert update["failure_decision"]["repair_attempt"] == 1
    assert update["failure_decision"]["repair_budget"] == 1


def test_exhausted_soft_evidence_is_terminal_degradation_without_human():
    state = _state(code="EVIDENCE_GAP")
    state["evidence_recovery_count"] = {"T1": 1}

    update = decide_recovery_action(state, _assessment("EVIDENCE_GAP"))

    assert update["failure_decision"]["failure_class"] == FailureClass.DEGRADABLE_QUALITY
    assert update["failure_decision"]["action"] == FailureAction.COMMIT_WITH_WARNING
    assert update["workflow_action"] == "DONE"
    assert update["pending_user_action"] == {}
    assert update["results"][0]["task_id"] == "T1"


def test_exhausted_hard_evidence_requires_contract_decision():
    state = _state(code="EVIDENCE_GAP", requirement_severity="hard")
    state["evidence_recovery_count"] = {"T1": 1}

    update = decide_recovery_action(state, _assessment("EVIDENCE_GAP"))

    assert update["failure_decision"]["failure_class"] == FailureClass.USER_DECISION_REQUIRED
    assert update["failure_decision"]["action"] == FailureAction.REGISTER_BLOCKER
    assert update["failure_decision"]["hard_requirement_ids"] == ["REQ-001"]
    assert update["workflow_action"] == "NEEDS_USER_INPUT"


def test_exhausted_optional_asset_degrades_and_commits():
    state = _state(code="MISSING_FIGURE", requirement_kind="asset")
    state["asset_retry_count"] = {"T1": MAX_ASSET_RETRIES}

    update = decide_recovery_action(state, _assessment("MISSING_FIGURE"))

    assert update["failure_decision"]["failure_class"] == FailureClass.DEGRADABLE_QUALITY
    assert update["workflow_action"] == "DONE"
    assert update["pending_user_action"] == {}


def test_exhausted_soft_length_degrades_after_bounded_repair():
    state = _state(code="TOO_LONG", requirement_kind="length")
    state["length_rewrite_attempts"] = {"T1:length_rewrite:t1": 1}

    update = decide_recovery_action(state, _assessment("TOO_LONG"))

    assert update["failure_decision"]["failure_class"] == FailureClass.DEGRADABLE_QUALITY
    assert update["workflow_action"] == "DONE"
    assert update["pending_user_action"] == {}


def test_verifier_exhaustion_is_graph_fatal_not_user_blocker():
    state = _state(code="VERIFIER_UNAVAILABLE")
    state["verifier_retry_count"] = {"T1": 1}
    state["verifier_failure"] = {
        "code": "VERIFIER_UNAVAILABLE",
        "retryable": False,
    }

    update = decide_recovery_action(state, _assessment("TOO_LONG"))

    assert update["failure_decision"]["failure_class"] == FailureClass.FATAL_SYSTEM
    assert update["failure_decision"]["subtype"] == "VERIFIER_UNAVAILABLE"
    assert update["failure_decision"]["action"] == FailureAction.FAIL_JOB
    assert update["pending_user_action"] == {}
    assert update["fatal_system_error"]["origin"] == "graph"
    assert update["fatal_system_error"]["retryable"] is False


def test_issue_text_cannot_promote_unlinked_soft_requirement_to_hard():
    state = _state(code="EVIDENCE_GAP", requirement_severity="hard")
    state["tasks"][0]["requirement_ids"] = []
    state["evidence_recovery_count"] = {"T1": 1}
    assessment = _assessment("EVIDENCE_GAP")
    assessment["issues"][0]["requirement_ids"] = []
    assessment["issues"][0]["description"] = "必须提供证据，否则不可接受"

    update = decide_recovery_action(state, assessment)

    assert update["failure_decision"]["hard_requirement_ids"] == []
    assert update["failure_decision"]["failure_class"] == FailureClass.DEGRADABLE_QUALITY


@pytest.mark.parametrize(
    "code",
    [
        "MISSING_TASK",
        "MISSING_DEPENDENCY",
        "INVALID_TASK_ORDER",
        "UNEXECUTABLE_TASK",
        "TASK_CONFLICT",
        "INVALID_TASK_DECOMPOSITION",
    ],
)
def test_stable_plan_defects_alone_route_to_plan_patcher(code):
    state = _state(code=code)
    assessment = {
        "status": "FAILED",
        "issues": [{"code": code, "category": "LOCAL_PLAN_DEFECT"}],
    }

    update = decide_recovery_action(state, assessment)

    assert update["workflow_action"] == "PLAN_PATCH"
    assert update["failure_decision"]["action"] == FailureAction.PATCH_PLAN


@pytest.mark.parametrize(
    "code,category",
    [
        ("EVIDENCE_GAP", "EVIDENCE_GAP"),
        ("MISSING_FIGURE", "CONTENT_DEFECT"),
        ("TOO_LONG", "CONTENT_DEFECT"),
        ("INVALID_CITATION_ID", "EVIDENCE_GAP"),
        ("FORMAT_ERROR", "CONTENT_DEFECT"),
        ("LLM_ERROR", "VERIFIER_FAILURE"),
    ],
)
def test_non_plan_failures_never_route_to_plan_patcher(code, category):
    state = _state(code=code)
    assessment = {
        "status": "FAILED",
        "issues": [{"code": code, "category": category}],
    }

    update = decide_recovery_action(state, assessment)

    assert update["workflow_action"] != "PLAN_PATCH"

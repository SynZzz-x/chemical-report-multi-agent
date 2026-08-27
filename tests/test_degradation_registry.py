import ast
from copy import deepcopy
import inspect

from src.nodes import asset_recovery as asset_recovery_module
from src.nodes import recovery as recovery_module
from src.recovery.policy import decide_recovery_action


def _state(*, task_revision: int = 2):
    return {
        "tasks": [
            {
                "task_id": "T1",
                "task_description": "撰写分析。",
                "task_type": "analysis",
                "use_rag": True,
                "use_web": False,
                "use_resources": [],
                "requirement_ids": ["REQ-002", "REQ-001"],
                "depends_on_task_ids": [],
            }
        ],
        "cursor": 0,
        "current_result": {"task_id": "T1", "text_output": "有依据的正文[E1]。"},
        "results": [],
        "section_status": {},
        "task_retry_count": {},
        "asset_retry_count": {},
        "evidence_recovery_count": {"T1": 1},
        "task_patch_count": {},
        "verifier_retry_count": {},
        "job_patch_count": 0,
        "task_revisions": {"T1": task_revision},
        "verification_warnings": [],
        "degraded_issue_registry": [],
        "requirement_registry": [
            {
                "requirement_id": requirement_id,
                "text": requirement_id,
                "severity": "soft",
                "kind": "evidence",
                "status": "active",
                "contract_revision": 1,
                "provenance": {
                    "origin": "intake_inferred",
                    "source_message_id": None,
                    "source_field": "constraints",
                    "source_index": index,
                    "derivation": "test",
                },
            }
            for index, requirement_id in enumerate(("REQ-001", "REQ-002"))
        ],
    }


def _assessment():
    return {
        "status": "FAILED",
        "issues": [
            {
                "code": "EVIDENCE_GAP",
                "category": "EVIDENCE_GAP",
                "requirement_ids": ["REQ-001", "REQ-002"],
                "affected_claims": ["claim-b", "claim-a"],
            }
        ],
    }


def test_terminal_degradation_writes_exact_stable_record():
    update = decide_recovery_action(_state(), _assessment())

    assert len(update["degraded_issue_registry"]) == 1
    record = update["degraded_issue_registry"][0]
    assert set(record) == {
        "issue_id",
        "task_id",
        "task_revision",
        "failure_class",
        "subtype",
        "reason",
        "affected_claims",
        "affected_requirement_ids",
        "attempted_repairs",
        "final_fallback",
        "status",
        "metadata",
    }
    assert record["issue_id"].startswith("degraded-")
    assert record["task_id"] == "T1"
    assert record["task_revision"] == 2
    assert record["failure_class"] == "DEGRADABLE_QUALITY"
    assert record["subtype"] == "EVIDENCE_GAP"
    assert record["affected_requirement_ids"] == ["REQ-001", "REQ-002"]
    assert record["attempted_repairs"] == [
        {
            "repair_type": "RECOVER_EVIDENCE",
            "attempt": 1,
            "budget": 1,
            "outcome": "exhausted",
            "diagnostic_code": "EVIDENCE_GAP",
        }
    ]
    assert record["status"] == "active"


def test_same_degradation_upserts_across_retry_and_resume():
    state = _state()
    first = decide_recovery_action(state, _assessment())
    resumed = {**state, **first}
    second = decide_recovery_action(resumed, _assessment())

    assert len(second["degraded_issue_registry"]) == 1
    assert second["degraded_issue_registry"] == first["degraded_issue_registry"]


def test_new_task_revision_supersedes_prior_active_degradation():
    state = _state(task_revision=2)
    first = decide_recovery_action(state, _assessment())
    revised = deepcopy({**state, **first})
    revised["task_revisions"] = {"T1": 3}

    second = decide_recovery_action(revised, _assessment())

    assert len(second["degraded_issue_registry"]) == 2
    assert second["degraded_issue_registry"][0]["status"] == "superseded"
    assert second["degraded_issue_registry"][1]["status"] == "active"
    assert second["degraded_issue_registry"][0]["issue_id"] != second[
        "degraded_issue_registry"
    ][1]["issue_id"]


def test_degradation_identity_is_independent_of_claim_order():
    state = _state()
    first = decide_recovery_action(state, _assessment())
    reordered = _assessment()
    reordered["issues"][0]["affected_claims"] = ["claim-a", "claim-b"]
    second = decide_recovery_action(state, reordered)

    assert first["degraded_issue_registry"][0]["issue_id"] == second[
        "degraded_issue_registry"
    ][0]["issue_id"]


def test_local_evidence_and_asset_recovery_cannot_interrupt_or_create_blockers():
    for function in (
        recovery_module.evidence_recovery,
        asset_recovery_module.asset_recovery,
    ):
        tree = ast.parse(inspect.getsource(function))
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "interrupt" not in called_names
        assert "pending_user_action" not in string_literals
        assert "NEEDS_USER_INPUT" not in string_literals

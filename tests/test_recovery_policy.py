import pytest

from src.recovery.policy import (
    IssueCategory,
    MAX_ASSET_RETRIES,
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
    asset_retry_count=None,
    job_patch_count=0,
    results=None,
    use_rag=True,
):
    return {
        "tasks": [
            {
                "task_id": task_id,
                "use_rag": use_rag,
                "use_web": False,
                "use_resources": task_resources or [],
                "requirement_ids": ["REQ-001"],
                "depends_on_task_ids": [],
            }
        ],
        "cursor": 0,
        "docs": list(docs or []),
        "current_result": {"task_id": task_id, "text_output": "current result"},
        "results": list(results or []),
        "section_status": {},
        "task_retry_count": dict(task_retry_count or {}),
        "evidence_recovery_count": dict(evidence_recovery_count or {}),
        "task_patch_count": dict(task_patch_count or {}),
        "verifier_retry_count": dict(verifier_retry_count or {}),
        "asset_retry_count": dict(asset_retry_count or {}),
        "job_patch_count": job_patch_count,
        "verification_warnings": [],
        "requirement_registry": [
            {
                "requirement_id": "REQ-001",
                "text": "测试中的显式硬约束",
                "severity": "hard",
                "kind": "test_contract",
                "status": "active",
                "contract_revision": 1,
                "provenance": {
                    "origin": "explicit_user",
                    "source_message_id": "test-message",
                    "source_field": "constraints",
                    "source_index": 0,
                    "derivation": "test_fixture",
                },
            }
        ],
    }


@pytest.mark.parametrize("code", ["MISSING_FIGURE", "MISSING_TABLE"])
def test_pure_asset_defect_uses_bounded_asset_recovery(code):
    state = recovery_state(task_id="T2")

    decision = decide_recovery_action(
        state,
        assessment_with(code, "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.ASSET_RECOVERY
    assert decision["asset_retry_count"] == {"T2": 1}
    assert decision["task_retry_count"] == {}


def test_asset_recovery_does_not_override_mixed_content_defects():
    state = recovery_state(task_id="T2")
    assessment = assessment_with_issues(
        {"code": "MISSING_FIGURE", "category": "CONTENT_DEFECT"},
        {"code": "TOO_LONG", "category": "CONTENT_DEFECT"},
    )

    decision = decide_recovery_action(state, assessment)

    assert decision["workflow_action"] == WorkflowAction.REWORK
    assert decision["asset_retry_count"] == {}
    assert decision["task_retry_count"] == {"T2": 1}


def test_asset_recovery_does_not_override_evidence_recovery():
    state = recovery_state(task_id="T2")
    assessment = assessment_with_issues(
        {"code": "MISSING_FIGURE", "category": "CONTENT_DEFECT"},
        {"code": "EVIDENCE_GAP", "category": "EVIDENCE_GAP"},
    )

    decision = decide_recovery_action(state, assessment)

    assert decision["workflow_action"] == WorkflowAction.EVIDENCE_RECOVERY
    assert decision["asset_retry_count"] == {}


def test_length_rewrite_is_semantic_once_then_uses_deterministic_convergence():
    duplicate = "这是一段可安全去重的过渡说明，不包含新的事实。"
    state = recovery_state(task_id="T2")
    state["tasks"][0]["task_description"] = "正文不超过35字。"
    state["task_revisions"] = {"T2": 3}
    state["current_result"]["text_output"] = (
        f"核心结论[E1]。\n\n{duplicate}\n\n{duplicate}"
    )
    assessment = assessment_with("TOO_LONG", "CONTENT_DEFECT")

    first = decide_recovery_action(state, assessment)
    second = decide_recovery_action({**state, **first}, assessment)

    assert first["workflow_action"] == "LENGTH_REWRITE"
    assert first["length_rewrite_attempts"] == {"T2:length_rewrite:t3": 1}
    assert second["workflow_action"] == "RETRY_VERIFIER"
    assert second["length_rewrite_attempts"] == first["length_rewrite_attempts"]
    assert second["current_result"]["text_output"].count(duplicate) == 1


def test_failed_length_rewrite_never_schedules_a_second_semantic_call():
    state = recovery_state(task_id="T2")
    state["tasks"][0]["task_description"] = "正文不超过10字。"
    state["task_revisions"] = {"T2": 1}
    state["length_rewrite_attempts"] = {"T2:length_rewrite:t1": 1}
    state["current_result"]["text_output"] = "不可安全截断的完整事实陈述[E1]。"

    decision = decide_recovery_action(
        state, assessment_with("TOO_LONG", "CONTENT_DEFECT")
    )

    assert decision["workflow_action"] == "NEEDS_USER_INPUT"
    assert decision["length_rewrite_attempts"] == state["length_rewrite_attempts"]


def test_prior_non_length_rework_does_not_consume_scoped_length_rewrite():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 1})
    state["length_rewrite_attempts"] = {}
    state["task_revisions"] = {"T2": 2}

    decision = decide_recovery_action(
        state, assessment_with("TOO_LONG", "CONTENT_DEFECT")
    )

    assert decision["workflow_action"] == "LENGTH_REWRITE"
    assert decision["length_rewrite_attempts"] == {"T2:length_rewrite:t2": 1}


def test_report_source_inconsistency_is_repaired_without_worker_rework():
    state = recovery_state(task_id="T2")
    state["current_result"]["report_sources"] = [
        "polyethylene.docx",
        "GB 18218-2018",
    ]
    assessment = assessment_with(
        "REPORT_SOURCE_INCONSISTENT",
        "CONTENT_DEFECT",
        expected_sources=["polyethylene.docx"],
        actual_sources=["polyethylene.docx", "GB 18218-2018"],
    )

    decision = decide_recovery_action(state, assessment)

    assert decision["workflow_action"] == "RETRY_VERIFIER"
    assert decision["current_result"]["report_sources"] == ["polyethylene.docx"]
    assert decision["task_retry_count"] == {}


def test_asset_retry_limit_uses_asset_specific_user_blocker():
    state = recovery_state(
        task_id="T2",
        asset_retry_count={"T2": MAX_ASSET_RETRIES},
    )

    decision = decide_recovery_action(
        state,
        assessment_with("MISSING_FIGURE", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert decision["pending_user_action"]["accepted_choices"] == [
        "RETRY_ASSET",
        "ADJUST_REQUIREMENT",
        "ACCEPT_AS_DRAFT",
        "DONE",
    ]
    assert "正文不会重新生成" in decision["pending_user_action"]["guidance"]


def assessment_with(code, category, **issue):
    return {
        "status": "FAILED",
        "issues": [{"code": code, "category": category, **issue}],
    }


def assessment_with_issues(*issues):
    return {
        "status": "FAILED",
        "issues": list(issues),
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
    assert second["pending_user_action"]["blocker_status"] == "ACTIVE"
    assert second["pending_user_action"]["blocker_id"].startswith("T2:p")
    assert second["pending_user_action"]["accepted_choices"] == [
        "UPLOAD_RESOURCES",
        "AUTHORIZE_WEB",
        "ADJUST_REQUIREMENT",
        "ACCEPT_EVIDENCE_GAP",
        "ACCEPT_AS_DRAFT",
    ]
    assert "上传" in second["pending_user_action"]["guidance"]
    assert "页面" in second["pending_user_action"]["guidance"]
    assert "直接回复" not in second["pending_user_action"]["guidance"]


def test_evidence_gap_without_authorized_retrieval_requires_user_input_immediately():
    state = recovery_state(task_id="T1", use_rag=False)

    decision = decide_recovery_action(
        state,
        assessment_with(
            "EVIDENCE_GAP",
            "EVIDENCE_GAP",
            retrieval_query="聚乙烯 引言 知识库依据",
        ),
    )

    assert decision["workflow_action"] == "NEEDS_USER_INPUT"
    assert decision["evidence_recovery_count"] == {}
    assert decision["pending_user_action"]["category"] == "EVIDENCE_GAP"
    assert "未授权可执行的证据检索能力" in decision["pending_user_action"]["guidance"]


def test_verifier_failure_exhaustion_is_fatal_without_blocker_actions():
    state = recovery_state(
        task_id="T1",
        verifier_retry_count={"T1": 1},
    )

    decision = decide_recovery_action(
        state,
        assessment_with("LLM_ERROR", "VERIFIER_FAILURE"),
    )

    assert decision["workflow_action"] == "FATAL_SYSTEM"
    assert decision["pending_user_action"] == {}
    assert decision["fatal_system_error"]["subtype"] == "VERIFIER_UNAVAILABLE"


def test_exhausted_verifier_execution_failure_precedes_semantic_assessment():
    state = recovery_state(
        task_id="T1",
        task_retry_count={"T1": 1},
        evidence_recovery_count={"T1": 1},
        task_patch_count={"T1": 1},
        verifier_retry_count={"T1": 2},
        asset_retry_count={"T1": 1},
        job_patch_count=2,
    )
    state["verifier_failure"] = {
        "code": "VERIFIER_UNAVAILABLE",
        "category": "VERIFIER_FAILURE",
        "message": "自动校验器本身未能产生合法校验结果。",
        "retryable": False,
        "contract_attempts": 3,
    }

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == "FATAL_SYSTEM"
    assert decision["task_retry_count"] == {"T1": 1}
    assert decision["asset_retry_count"] == {"T1": 1}
    assert decision["evidence_recovery_count"] == {"T1": 1}
    assert decision["task_patch_count"] == {"T1": 1}
    assert decision["job_patch_count"] == 2
    assert decision["pending_user_action"] == {}
    assert decision["fatal_system_error"]["diagnostic_code"] == (
        "VERIFIER_UNAVAILABLE"
    )


def test_content_and_waivable_evidence_gap_can_be_explicitly_accepted_as_draft():
    state = recovery_state(
        task_id="T1",
        evidence_recovery_count={"T1": 1},
    )
    assessment = assessment_with_issues(
        {"code": "TOO_LONG", "category": "CONTENT_DEFECT"},
        {"code": "EVIDENCE_GAP", "category": "EVIDENCE_GAP"},
    )

    decision = decide_recovery_action(state, assessment)

    assert decision["pending_user_action"]["category"] == "EVIDENCE_GAP"
    assert "ACCEPT_EVIDENCE_GAP" in decision["pending_user_action"][
        "accepted_choices"
    ]
    assert "ACCEPT_AS_DRAFT" in decision["pending_user_action"][
        "accepted_choices"
    ]


@pytest.mark.parametrize(
    "integrity_code",
    ["INVALID_CITATION_ID", "MISSING_INLINE_CITATION", "SOURCE_UNSUPPORTED"],
)
def test_content_and_integrity_issue_cannot_be_accepted_as_draft(integrity_code):
    state = recovery_state(
        task_id="T1",
        evidence_recovery_count={"T1": 1},
    )
    assessment = assessment_with_issues(
        {"code": "TOO_LONG", "category": "CONTENT_DEFECT"},
        {"code": integrity_code, "category": "EVIDENCE_GAP"},
    )

    decision = decide_recovery_action(state, assessment)

    assert "ACCEPT_EVIDENCE_GAP" not in decision["pending_user_action"][
        "accepted_choices"
    ]
    assert "ACCEPT_AS_DRAFT" not in decision["pending_user_action"][
        "accepted_choices"
    ]
    assert "带风险草稿" not in decision["pending_user_action"]["guidance"]


@pytest.mark.parametrize(
    "integrity_code",
    ["INVALID_CITATION_ID", "MISSING_INLINE_CITATION", "SOURCE_UNSUPPORTED"],
)
def test_verifier_failure_and_integrity_issue_is_fatal_without_draft_action(
    integrity_code,
):
    state = recovery_state(
        task_id="T1",
        verifier_retry_count={"T1": 1},
    )
    assessment = assessment_with_issues(
        {"code": "LLM_ERROR", "category": "VERIFIER_FAILURE"},
        {"code": integrity_code, "category": "EVIDENCE_GAP"},
    )

    decision = decide_recovery_action(state, assessment)

    assert decision["workflow_action"] == "FATAL_SYSTEM"
    assert decision["pending_user_action"] == {}
    assert decision["failure_decision"]["subtype"] == "VERIFIER_UNAVAILABLE"


def test_evidence_blocker_does_not_offer_web_when_runtime_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "src.recovery.policy.public_web_runtime_available",
        lambda: False,
    )
    state = recovery_state(task_id="T2", evidence_recovery_count={"T2": 1})

    decision = decide_recovery_action(
        state, assessment_with("EVIDENCE_GAP", "EVIDENCE_GAP")
    )

    assert "AUTHORIZE_WEB" not in decision["pending_user_action"]["accepted_choices"]
    assert "ACCEPT_EVIDENCE_GAP" in decision["pending_user_action"][
        "accepted_choices"
    ]
    assert "当前服务器未提供可用的公开网络检索工具" in decision["pending_user_action"]["guidance"]


def test_evidence_blocker_offers_web_when_runtime_is_available(monkeypatch):
    monkeypatch.setattr(
        "src.recovery.policy.public_web_runtime_available",
        lambda: True,
    )
    state = recovery_state(task_id="T2", evidence_recovery_count={"T2": 1})

    decision = decide_recovery_action(
        state, assessment_with("EVIDENCE_GAP", "EVIDENCE_GAP")
    )

    assert "AUTHORIZE_WEB" in decision["pending_user_action"]["accepted_choices"]
    assert "ACCEPT_EVIDENCE_GAP" in decision["pending_user_action"][
        "accepted_choices"
    ]


def test_nonwaivable_source_failure_does_not_offer_gap_acceptance():
    state = recovery_state(
        task_id="T2", evidence_recovery_count={"T2": 1}
    )

    decision = decide_recovery_action(
        state,
        assessment_with("SOURCE_UNSUPPORTED", "EVIDENCE_GAP"),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert "ACCEPT_EVIDENCE_GAP" not in decision["pending_user_action"][
        "accepted_choices"
    ]


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


def test_content_retry_limit_requires_explicit_user_acceptance_without_committing():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})
    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )
    assert decision["workflow_action"] == "NEEDS_USER_INPUT"
    assert "results" not in decision
    assert decision["section_status"]["T2"]["status"] == "ACCEPT_WITH_WARNING"
    assert decision["pending_user_action"]["accepted_choices"] == [
        "REWORK",
        "ADJUST_REQUIREMENT",
        "ACCEPT_AS_DRAFT",
        "DONE",
    ]


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


def test_legacy_mixed_contract_error_invalidates_semantic_issues():
    state = recovery_state(task_id="T1")
    assessment = {
        "status": "FAILED",
        "issues": [
            {
                "code": "ASSESSMENT_CONTRACT_ERROR",
                "category": "VERIFIER_FAILURE",
            },
            {"code": "TOO_LONG", "category": "CONTENT_DEFECT"},
        ],
    }

    assert classify_assessment(assessment, state) is IssueCategory.VERIFIER_FAILURE

    decision = decide_recovery_action(state, assessment)

    assert decision["workflow_action"] == WorkflowAction.RETRY_VERIFIER
    assert decision["task_retry_count"] == {}
    assert decision["verifier_retry_count"] == {"T1": 1}


def test_synthesis_content_or_evidence_failure_uses_synthesis_rewrite():
    state = recovery_state(task_id="T2")
    state["tasks"][0]["task_type"] = "synthesis"

    for code, category in (
        ("CONTENT_DEFECT", "CONTENT_DEFECT"),
        ("EVIDENCE_GAP", "EVIDENCE_GAP"),
    ):
        decision = decide_recovery_action(
            state, assessment_with(code, category)
        )
        assert decision["workflow_action"] == WorkflowAction.SYNTHESIS_REWRITE


def test_legacy_cursor_counter_keys_are_read_and_written_with_task_ids():
    state = recovery_state(task_id="T2", task_retry_count={0: 1})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert decision["task_retry_count"] == {"T2": 1}
    assert decision["length_rewrite_attempts"] == {"T2:length_rewrite:t1": 1}


@pytest.mark.parametrize(
    ("assessment", "expected_action"),
    [
        (
            assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
            WorkflowAction.LENGTH_REWRITE,
        ),
        (
            assessment_with("EVIDENCE_GAP", "EVIDENCE_GAP"),
            WorkflowAction.EVIDENCE_RECOVERY,
        ),
    ],
)
def test_failed_assessment_revokes_stale_pass_while_recovery_runs(
    assessment, expected_action
):
    state = recovery_state(task_id="T2")
    state["section_status"] = {"T2": {"status": "VERIFIED_PASS"}}

    decision = decide_recovery_action(state, assessment)

    assert decision["workflow_action"] == expected_action
    assert decision["section_status"]["T2"]["status"] == "BLOCKED"
    assert decision["report_status"] == "BLOCKED"


def test_json_restored_numeric_string_counter_preserves_content_retry_cap():
    state = recovery_state(task_id="T2", task_retry_count={"0": 2})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert decision["task_retry_count"] == {"T2": 2}


def test_numeric_string_that_is_a_real_task_id_is_not_treated_as_a_cursor():
    state = recovery_state(task_id="0", task_retry_count={"0": 2})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
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
            WorkflowAction.NEEDS_USER_INPUT,
        ),
        (
            "evidence_recovery_count",
            assessment_with("MISSING_EVIDENCE", "EVIDENCE_GAP"),
            WorkflowAction.NEEDS_USER_INPUT,
        ),
        (
            "verifier_retry_count",
            assessment_with("ASSESSMENT_CONTRACT_ERROR", "VERIFIER_FAILURE"),
            WorkflowAction.FATAL_SYSTEM,
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
    assert first["section_status"]["T2"]["status"] == "VERIFIED_PASS"
    assert first["report_status"] == "READY_FOR_FINAL"


def test_pass_preserves_matching_user_accepted_evidence_gap_status():
    state = recovery_state(task_id="T2")
    state.update(
        {
            "plan_revision": 3,
            "task_revisions": {"T2": 2},
            "accepted_evidence_gaps": {
                "T2": {
                    "plan_revision": 3,
                    "task_revision": 2,
                    "issues": [
                        {
                            "code": "EVIDENCE_GAP",
                            "category": "EVIDENCE_GAP",
                            "description": "知识库没有粒径取样规程",
                        }
                    ],
                }
            },
        }
    )

    decision = decide_recovery_action(state, {"status": "PASS", "issues": []})

    assert decision["workflow_action"] == WorkflowAction.DONE
    assert decision["section_status"]["T2"]["status"] == "USER_ACCEPTED_GAP"
    assert decision["section_status"]["T2"]["accepted_by"] == "user"
    assert decision["report_status"] == "DRAFT_WITH_GAPS"


def test_newly_verified_revision_replaces_stale_result_for_same_task():
    state = recovery_state(
        task_id="T2",
        results=[
            {
                "task_id": "T2",
                "text_output": "old",
                "plan_revision": 1,
                "task_revision": 1,
            }
        ],
    )
    state["plan_revision"] = 1
    state["task_revisions"] = {"T2": 2}
    state["current_result"] = {
        "task_id": "T2",
        "text_output": "new",
        "plan_revision": 0,
        "task_revision": 1,
    }

    decision = decide_recovery_action(state, {"status": "PASS", "issues": []})

    assert len(decision["results"]) == 1
    assert decision["results"][0]["text_output"] == "new"
    assert decision["results"][0]["task_revision"] == 2


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
        "UNEXECUTABLE_TASK",
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


def test_content_retry_overflow_blocks_non_final_task_for_user_input():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})
    state["tasks"].append({"task_id": "T3", "use_resources": []})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert "continuation_action" not in decision
    assert decision["report_status"] == "BLOCKED"


def test_content_retry_overflow_blocks_final_task_for_user_input():
    state = recovery_state(task_id="T2", task_retry_count={"T2": 2})

    decision = decide_recovery_action(
        state,
        assessment_with("TOO_SHORT", "CONTENT_DEFECT"),
    )

    assert decision["workflow_action"] == WorkflowAction.NEEDS_USER_INPUT
    assert "continuation_action" not in decision
    assert decision["report_status"] == "BLOCKED"

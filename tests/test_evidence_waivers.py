from src.evidence_waivers import apply_evidence_gap_acceptance


def _state(*, plan_revision=2, task_revision=3):
    return {
        "tasks": [{"task_id": "T1"}],
        "cursor": 0,
        "plan_revision": plan_revision,
        "task_revisions": {"T1": task_revision},
        "accepted_evidence_gaps": {
            "T1": {
                "plan_revision": 2,
                "task_revision": 3,
                "issues": [{"code": "EVIDENCE_GAP", "description": "缺少定义"}],
            }
        },
    }


def test_matching_acceptance_waives_availability_gap_but_not_citation_integrity():
    assessment = {
        "status": "FAILED",
        "issues": [
            {"code": "EVIDENCE_GAP", "category": "EVIDENCE_GAP"},
            {"code": "INVALID_CITATION_ID", "category": "EVIDENCE_GAP"},
        ],
        "requirements_missing": ["定义", "有效引用"],
    }

    filtered = apply_evidence_gap_acceptance(assessment, _state())

    assert filtered["status"] == "FAILED"
    assert [issue["code"] for issue in filtered["issues"]] == [
        "INVALID_CITATION_ID"
    ]
    assert [issue["code"] for issue in filtered["waived_evidence_issues"]] == [
        "EVIDENCE_GAP"
    ]


def test_acceptance_is_invalid_after_task_revision_changes():
    assessment = {
        "status": "FAILED",
        "issues": [{"code": "EVIDENCE_GAP", "category": "EVIDENCE_GAP"}],
        "requirements_missing": ["定义"],
    }

    filtered = apply_evidence_gap_acceptance(
        assessment, _state(task_revision=4)
    )

    assert filtered == assessment

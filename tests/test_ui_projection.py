from pathlib import Path


def test_verifier_summary_renders_assessment_status_and_issue_category():
    from src.ui_projection import summarize_step

    summary = summarize_step(
        "QualityReview",
        {
            "assessment": {
                "status": "FAILED",
                "issues": [{"category": "EVIDENCE_GAP"}],
            }
        },
    )

    assert summary == "审核状态：FAILED，问题类型：EVIDENCE_GAP"
    assert "审核决策：-" not in summary


def test_decision_policy_summary_renders_recovery_action():
    from src.ui_projection import summarize_step

    assert (
        summarize_step("DecisionPolicy", {"workflow_action": "EVIDENCE_RECOVERY"})
        == "恢复动作：EVIDENCE_RECOVERY"
    )


def test_streamlit_consumer_uses_shared_summary_projection():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")

    assert "summarize_step as _summarize_step" in source


def test_task_progress_view_uses_ledger_not_cursor():
    from src.ui_projection import task_progress_view

    state = {
        "tasks": [
            {"task_id": "T1", "task_name": "指标"},
            {"task_id": "T2", "task_name": "参数"},
        ],
        "cursor": 0,
        "task_records": {
            "T1": {"status": "PASSED", "attempt_count": 1},
            "T2": {
                "status": "RUNNING",
                "attempt_count": 2,
                "active_artifact_id": "A2",
            },
        },
        "review_records": [
            {
                "review_id": "R2",
                "task_id": "T2",
                "artifact_id": "A2",
                "status": "REVISE",
                "issues": [{"category": "CONTENT_DEFECT"}],
            }
        ],
    }

    view = task_progress_view(state)

    assert [(item["task_id"], item["status"]) for item in view] == [
        ("T1", "PASSED"),
        ("T2", "RUNNING"),
    ]
    assert view[1]["attempt_count"] == 2
    assert view[1]["active_artifact_id"] == "A2"
    assert view[1]["latest_review_status"] == "REVISE"


def test_report_status_view_exposes_not_attempted_and_failures():
    from src.ui_projection import report_status_view

    view = report_status_view(
        {
            "report_manifest": {
                "outputs": {
                    "md": {"status": "SUCCEEDED", "path": "/job/report.md", "error": None},
                    "pdf": {"status": "FAILED", "path": None, "error": "wide table"},
                }
            }
        }
    )

    assert view["md"]["status"] == "SUCCEEDED"
    assert view["pdf"] == {"status": "FAILED", "path": None, "error": "wide table"}
    assert view["docx"]["status"] == "NOT_ATTEMPTED"


def test_streamlit_renders_auditable_views_and_supported_uploads():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")

    assert "task_progress_view" in source
    assert "report_status_view" in source
    assert "允许本报告任务检索可信公开网络资料" in source
    assert 'file_type=["pdf", "docx", "csv", "xlsx", "xls"]' in source

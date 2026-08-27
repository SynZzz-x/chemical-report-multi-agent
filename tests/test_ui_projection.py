from pathlib import Path


def test_verifier_summary_renders_assessment_status_and_issue_category():
    from src.ui_projection import summarize_step

    summary = summarize_step(
        "Verifier",
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


def test_decision_policy_summary_projects_degradation_without_blocker_wording():
    from src.ui_projection import summarize_step

    summary = summarize_step(
        "DecisionPolicy",
        {
            "workflow_action": "NEXT",
            "failure_decision": {
                "failure_class": "DEGRADABLE_QUALITY",
                "subtype": "MISSING_FIGURE",
            },
        },
    )

    assert summary == "已记录非阻塞交付限制：MISSING_FIGURE"


def test_streamlit_consumer_uses_shared_summary_projection():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")

    assert "from src.ui_projection import summarize_step as _summarize_step" in source

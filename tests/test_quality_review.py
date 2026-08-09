import json
from types import SimpleNamespace

from src.quality.validators import validate_artifact
from src.quality.models import QualityDimensions, ReviewAssessment
from src.nodes import quality_review as review_module


def _state(*, task_id="T1", artifact_id="A1"):
    task = {
        "task_id": task_id,
        "task_name": "质量指标",
        "task_description": "分析质量指标",
        "generate_table": False,
        "generate_figure": False,
    }
    artifact = {
        "task_id": task_id,
        "artifact_id": artifact_id,
        "status": "COMPLETED",
        "text_output": "有证据支持的质量分析正文。",
        "content": "有证据支持的质量分析正文。",
        "tables": [],
        "figures": [],
        "citations": [{"evidence_id": "E1", "source_type": "rag"}],
    }
    return {
        "user_id": "u1",
        "job_id": "j1",
        "tasks": [task],
        "cursor": 0,
        "current_task": task,
        "current_result": artifact,
        "active_artifact_ids": {task_id: artifact_id},
        "review_records": [],
    }


def _passing_payload():
    return {
        "status": "PASS",
        "issues": [],
        "quality_dimensions": {
            "completeness": 5,
            "evidence": 4,
            "logic": 4,
            "actionability": 4,
            "safety": 5,
        },
    }


def test_deterministic_review_rejects_missing_required_table_and_bad_binding():
    task = {
        "task_id": "T2",
        "generate_table": True,
        "task_description": "生成指标表",
    }
    artifact = {
        "task_id": "T1",
        "artifact_id": "A1",
        "status": "COMPLETED",
        "text_output": "正文",
        "tables": [],
        "figures": [],
        "citations": [],
    }

    issues = validate_artifact(task, artifact, active_artifact_id="A2")

    assert {issue.code for issue in issues} == {
        "ARTIFACT_TASK_MISMATCH",
        "STALE_ARTIFACT",
        "MISSING_TABLE",
    }
    assert all(issue.responsible_handler for issue in issues)


def test_quality_review_is_assessment_only_and_binds_artifact(monkeypatch):
    class Model:
        def invoke(self, payload, **kwargs):
            return SimpleNamespace(content=json.dumps(_passing_payload()))

    modes = []
    monkeypatch.setattr(
        review_module,
        "get_llm",
        lambda config, json_mode=True: modes.append(json_mode) or Model(),
    )

    update = review_module.quality_review(
        _state(), {"configurable": {"use_llm": True}}
    )

    assert set(update) == {"review_record", "review_records", "assessment"}
    assert modes == [True]
    assert update["review_record"]["status"] == "PASS"
    assert update["review_record"]["task_id"] == "T1"
    assert update["review_record"]["artifact_id"] == "A1"
    assert update["assessment"]["artifact_id"] == "A1"
    assert "decision" not in update


def test_quality_review_service_failure_does_not_become_content_failure(monkeypatch):
    class FailingModel:
        def invoke(self, payload, **kwargs):
            raise RuntimeError("review API down")

    monkeypatch.setattr(
        review_module,
        "get_llm",
        lambda *args, **kwargs: FailingModel(),
    )

    update = review_module.quality_review(
        _state(), {"configurable": {"use_llm": True}}
    )

    issue = update["review_record"]["issues"][0]
    assert issue["category"] == "REVIEW_FAILURE"
    assert issue["code"] == "REVIEW_SERVICE_ERROR"
    assert update["review_record"]["artifact_id"] == "A1"


def test_replayed_quality_review_does_not_duplicate_review_record(monkeypatch):
    class Model:
        def invoke(self, payload, **kwargs):
            return SimpleNamespace(content=json.dumps(_passing_payload()))

    monkeypatch.setattr(review_module, "get_llm", lambda *args, **kwargs: Model())
    state = _state()
    first = review_module.quality_review(
        state, {"configurable": {"use_llm": True}}
    )
    replay = review_module.quality_review(
        {**state, **first}, {"configurable": {"use_llm": True}}
    )

    assert replay["review_record"]["review_id"] == first["review_record"]["review_id"]
    assert len(replay["review_records"]) == 1


def test_non_pass_semantic_status_without_issue_becomes_review_failure(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        review_module,
        "_semantic_assessment",
        lambda *args, **kwargs: ReviewAssessment(
            status="REVISE",
            issues=[],
            quality_dimensions=QualityDimensions(
                completeness=2,
                evidence=2,
                logic=2,
                actionability=2,
                safety=5,
            ),
        ),
    )

    update = review_module.quality_review(
        state, {"configurable": {"use_llm": True}}
    )

    assert update["review_record"]["status"] == "BLOCKED"
    assert update["review_record"]["issues"][0]["category"] == "REVIEW_FAILURE"


def test_review_id_is_stable_for_same_artifact_and_review_attempt(monkeypatch):
    state = _state()
    changed_payload = _passing_payload()
    changed_payload["quality_dimensions"]["logic"] = 3
    assessments = iter(
        (
            ReviewAssessment.model_validate(_passing_payload()),
            ReviewAssessment.model_validate(changed_payload),
        )
    )
    monkeypatch.setattr(
        review_module,
        "_semantic_assessment",
        lambda *args, **kwargs: next(assessments),
    )

    first = review_module.quality_review(
        state, {"configurable": {"use_llm": True}}
    )
    replay_state = {**state, "review_records": []}
    second = review_module.quality_review(
        replay_state, {"configurable": {"use_llm": True}}
    )

    assert first["review_record"]["review_id"] == second["review_record"]["review_id"]

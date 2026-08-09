from src.nodes import legacy_verifier as module


class RecordingStore:
    def __init__(self):
        self.puts = []

    def put(self, namespace, key, value, **kwargs):
        self.puts.append((namespace, key, value, kwargs))


def _legacy_state():
    return {
        "user_id": "user-1",
        "job_id": "job-1",
        "tasks": [{"task_id": "T1", "task_name": "Legacy task"}],
        "cursor": 0,
        "results": [],
        "current_result": {
            "task_id": "T1",
            "text_output": "legacy worker output",
            "citations": [],
        },
    }


def test_legacy_auto_verifier_commits_artifact_before_quality_review(monkeypatch):
    observed = {}

    def fake_quality_review(state, config, store=None, **kwargs):
        observed.update(state)
        return {"review_record": {"status": "PASS"}}

    monkeypatch.setattr(module, "quality_review", fake_quality_review)
    store = RecordingStore()

    update = module.legacy_auto_verifier(_legacy_state(), {}, store=store)

    artifact = observed["current_result"]
    assert artifact["artifact_id"].startswith("artifact_")
    assert observed["active_artifact_ids"]["T1"] == artifact["artifact_id"]
    assert observed["task_records"]["T1"]["status"] == "RUNNING"
    assert update["current_result"] == artifact
    assert any(namespace[-1] == "artifacts" for namespace, *_ in store.puts)


def test_legacy_manual_verifier_receives_committed_artifact(monkeypatch):
    observed = {}

    def fake_manual_verifier(state, config, store=None, **kwargs):
        observed.update(state)
        return {"decision": "NEXT"}

    monkeypatch.setattr(module, "verifier_manual", fake_manual_verifier)

    update = module.legacy_manual_verifier(_legacy_state(), {})

    assert observed["current_result"]["artifact_id"]
    assert update["current_result"] == observed["current_result"]
    assert update["decision"] == "NEXT"


def test_legacy_verifier_keeps_existing_artifact_unchanged(monkeypatch):
    state = _legacy_state()
    state["current_result"]["artifact_id"] = "artifact-existing"
    state["artifacts"] = {"artifact-existing": dict(state["current_result"])}
    state["active_artifact_ids"] = {"T1": "artifact-existing"}

    monkeypatch.setattr(
        module,
        "quality_review",
        lambda upgraded, config, store=None, **kwargs: {
            "observed_artifact_id": upgraded["current_result"]["artifact_id"]
        },
    )

    update = module.legacy_auto_verifier(state, {})

    assert update["observed_artifact_id"] == "artifact-existing"
    assert "artifacts" not in update

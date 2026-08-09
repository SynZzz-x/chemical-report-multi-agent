import pytest

from src.job_store import JobStore, resumable_checkpoint
from src.nodes.planner import _apply_job_web_authorization
from src.task_contract import task_allows_web


def test_task_web_flag_cannot_bypass_job_authorization():
    assert task_allows_web({"use_web": True}, job_authorized=False) is False
    assert task_allows_web({"use_web": True}, job_authorized=True) is True


def test_planner_forces_all_web_flags_off_without_job_authorization():
    tasks = [
        {
            "task_id": "T1",
            "use_web": True,
            "allow_web_fallback": True,
            "tool_requirements": ["chemical_knowledge_base_tool", "spider_tool"],
            "visualization": {"allow_web_fallback": True},
        }
    ]

    normalized = _apply_job_web_authorization(tasks, job_authorized=False)

    assert normalized[0]["use_web"] is False
    assert normalized[0]["allow_web_fallback"] is False
    assert normalized[0]["visualization"]["allow_web_fallback"] is False
    assert normalized[0]["tool_requirements"] == ["chemical_knowledge_base_tool"]


class MemoryStore:
    def __init__(self):
        self.items = {}

    def put(self, namespace, key, value, index=False):
        self.items[(namespace, key)] = dict(value)

    def get(self, namespace, key):
        value = self.items.get((namespace, key))
        return None if value is None else type("Item", (), {"value": value})()

    def search(self, namespace, limit=100):
        return []


def test_job_web_authorization_is_persisted_and_immutable():
    jobs = JobStore(MemoryStore())
    record = jobs.create_job(
        user_id="u1",
        conversation_id="c1",
        job_id="j1",
        title="report",
        verifier_mode="auto",
        web_authorized=True,
    )

    assert record["web_authorized"] is True
    with pytest.raises(ValueError, match="web_authorized"):
        jobs.update_job("u1", "j1", web_authorized=False)


def test_non_interrupt_checkpoint_with_next_node_is_resumable():
    snapshot = type("Snapshot", (), {"next": ("QualityReview",), "tasks": ()})()

    assert resumable_checkpoint(snapshot) is True

from pathlib import Path

from src.recovery.policy import (
    MAX_CONTENT_RETRIES,
    MAX_EVIDENCE_RECOVERIES,
    MAX_JOB_PATCHES,
)


def test_nine_task_recursion_limit_covers_worst_bounded_recovery_with_margin():
    from src.runtime_config import recursion_limit_for_tasks

    task_count = 9
    estimated_supersteps = (
        8
        + task_count
        * (
            3
            + 3 * MAX_CONTENT_RETRIES
            + 4 * MAX_EVIDENCE_RECOVERIES
            + 2  # one verifier-only retry
        )
        + 4 * MAX_JOB_PATCHES
    )

    assert recursion_limit_for_tasks(task_count) >= estimated_supersteps + 18
    assert recursion_limit_for_tasks(task_count) < 1000


def test_execution_config_scales_with_checkpoint_task_count_without_mutating_input():
    from src.runtime_config import execution_config, recursion_limit_for_tasks

    base = {"configurable": {"thread_id": "job-1"}, "recursion_limit": 10}
    state = {"tasks": [{"task_id": f"T{i}"} for i in range(12)]}

    configured = execution_config(base, state)

    assert configured["recursion_limit"] == recursion_limit_for_tasks(12)
    assert configured["configurable"] == {"thread_id": "job-1"}
    assert base["recursion_limit"] == 10


def test_app_and_cli_use_shared_execution_config_consumer():
    root = Path(__file__).parents[1]

    for relative_path in ("app.py", "run.py"):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "from src.runtime_config import execution_config" in source
        assert "execution_config(" in source

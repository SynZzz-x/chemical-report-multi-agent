from pathlib import Path

import pytest

from src.limits import MAX_PLAN_TASKS
from src.recovery.policy import (
    MAX_ASSET_RETRIES,
    MAX_CONTENT_RETRIES,
    MAX_EVIDENCE_RECOVERIES,
    MAX_JOB_PATCHES,
    MAX_VERIFIER_RETRIES,
)


def test_runtime_budget_covers_maximum_plan_bounded_recovery_with_margin():
    from src.runtime_config import recursion_limit_for_tasks

    task_count = MAX_PLAN_TASKS
    estimated_supersteps = (
        8
        + task_count
        * (
            3
            + 3 * MAX_CONTENT_RETRIES
            + 4 * MAX_EVIDENCE_RECOVERIES
            + 2 * MAX_VERIFIER_RETRIES
            + 3 * MAX_ASSET_RETRIES
        )
        + 4 * MAX_JOB_PATCHES
    )

    assert recursion_limit_for_tasks(9) == recursion_limit_for_tasks(task_count)
    assert recursion_limit_for_tasks(task_count) >= estimated_supersteps + 2 * task_count


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


def test_length_rewrite_safety_ratio_is_loaded_and_bounded(monkeypatch):
    from src import config as config_module

    monkeypatch.setenv("LENGTH_REWRITE_SAFETY_RATIO", "0.9")
    config_module.get_app_config.cache_clear()
    assert config_module.get_app_config().length_rewrite_safety_ratio == 0.9

    monkeypatch.setenv("LENGTH_REWRITE_SAFETY_RATIO", "0.5")
    config_module.get_app_config.cache_clear()
    with pytest.raises(ValueError, match="between 0.8 and 1.0"):
        config_module.get_app_config()
    config_module.get_app_config.cache_clear()

from copy import deepcopy

import pytest

from src.limits import MAX_PLAN_TASKS
from src.recovery.plan_patch import PatchValidationError, apply_plan_patch, validate_plan_patch


def task(task_id):
    return {
        "task_id": task_id,
        "task_name": f"Task {task_id}",
        "task_description": f"Description for {task_id}",
        "task_type": "analysis",
        "use_rag": True,
        "use_web": False,
        "generate_table": False,
        "generate_figure": False,
        "query": f"query {task_id}",
        "use_resources": [],
    }


def patch_state(
    *,
    cursor=0,
    accepted_ids=(),
    docs=None,
    plan_revision=1,
    task_patch_count=None,
    job_patch_count=0,
):
    tasks = [task(task_id) for task_id in ("T1", "T2", "T3", "T4")]
    return {
        "tasks": tasks,
        "cursor": cursor,
        "docs": list(docs if docs is not None else [{"name": "evidence.csv"}]),
        "results": [
            {"task_id": task_id, "text_output": f"accepted {task_id}"}
            for task_id in accepted_ids
        ],
        "plan_revision": plan_revision,
        "task_revisions": {item["task_id"]: 1 for item in tasks},
        "task_patch_count": dict(task_patch_count or {}),
        "job_patch_count": job_patch_count,
        "pending_user_action": {"category": "LOCAL_PLAN_DEFECT"},
        "plan_patch_history": [],
    }


def update_patch(
    *,
    task_id="T3",
    changes=None,
    affected_task_ids=None,
    resume_task_id=None,
    base_plan_revision=1,
    **overrides,
):
    return {
        "base_plan_revision": base_plan_revision,
        "reason_code": "RESOURCE_NOT_ASSIGNED",
        "reason": "The task needs the existing evidence file.",
        "affected_task_ids": affected_task_ids or [task_id],
        "operations": [
            {
                "op": "update_task",
                "task_id": task_id,
                "changes": (
                    {"use_resources": ["evidence.csv"]}
                    if changes is None
                    else changes
                ),
            }
        ],
        "resume_task_id": resume_task_id or task_id,
        "expected_resolution": "The task can use the required evidence.",
        **overrides,
    }


def insert_before_patch(*, before_task_id="T3", inserted_task_id="T2A", **overrides):
    return {
        "base_plan_revision": 1,
        "reason_code": "MISSING_DEPENDENCY",
        "reason": "A prerequisite evidence task is required.",
        "affected_task_ids": [inserted_task_id, before_task_id],
        "operations": [
            {
                "op": "insert_before",
                "before_task_id": before_task_id,
                "task": {**task(inserted_task_id), "task_name": "Evidence prerequisite"},
            }
        ],
        "resume_task_id": inserted_task_id,
        "expected_resolution": "The prerequisite completes before the blocked task.",
        **overrides,
    }


def move_before_patch(*, task_id="T4", before_task_id="T3", **overrides):
    return {
        "base_plan_revision": 1,
        "reason_code": "INVALID_TASK_ORDER",
        "reason": "The dependency needs to run first.",
        "affected_task_ids": [task_id, before_task_id],
        "operations": [
            {"op": "move_before", "task_id": task_id, "before_task_id": before_task_id}
        ],
        "resume_task_id": task_id,
        "expected_resolution": "The prerequisite precedes its dependent task.",
        **overrides,
    }


def test_update_patch_preserves_unaffected_results_and_cursor_progress():
    state = patch_state(cursor=2, accepted_ids=["T1", "T2"])
    patch = update_patch(task_id="T3", resume_task_id="T3")

    update = apply_plan_patch(state, patch)

    assert [item["task_id"] for item in update["results"]] == ["T1", "T2"]
    assert update["cursor"] == 2
    assert update["plan_revision"] == 2


@pytest.mark.parametrize(
    "reason_code",
    [
        "EVIDENCE_GAP",
        "MISSING_FIGURE",
        "TOO_LONG",
        "INVALID_CITATION_ID",
        "FORMAT_ERROR",
        "LLM_ERROR",
    ],
)
def test_plan_patch_rejects_non_plan_failure_reason_codes(reason_code):
    with pytest.raises(PatchValidationError, match="plan-defect reason_code"):
        validate_plan_patch(patch_state(), update_patch(reason_code=reason_code))


def test_update_patch_rejects_accepted_task_without_any_partial_mutation():
    state = patch_state(cursor=1, accepted_ids=["T1"])
    before = deepcopy(state)
    patch = update_patch(
        task_id="T1",
        affected_task_ids=["T1"],
        resume_task_id="T1",
        changes={"task_description": "mutated accepted task"},
    )

    with pytest.raises(PatchValidationError, match="completed|accepted"):
        apply_plan_patch(state, patch)

    assert state == before


def test_update_patch_rejects_cursor_completed_task_without_results():
    state = patch_state(cursor=2, accepted_ids=[])
    before = deepcopy(state)
    patch = update_patch(
        task_id="T1",
        affected_task_ids=["T1"],
        resume_task_id="T1",
    )

    with pytest.raises(PatchValidationError, match="completed|accepted"):
        apply_plan_patch(state, patch)

    assert state == before


def test_move_patch_rejects_task_marked_completed_without_results():
    state = patch_state(cursor=0, accepted_ids=[])
    state["tasks"][3]["status"] = "COMPLETED"
    before = deepcopy(state)

    with pytest.raises(PatchValidationError, match="completed|accepted"):
        apply_plan_patch(state, move_before_patch(task_id="T4", before_task_id="T3"))

    assert state == before


def test_move_patch_cannot_reorder_report_section_coverage():
    state = patch_state(cursor=0, accepted_ids=[])
    for index, planned_task in enumerate(state["tasks"], start=1):
        planned_task["covers_sections"] = [f"{index}. Section {index}"]

    with pytest.raises(PatchValidationError, match="section coverage order"):
        validate_plan_patch(
            state,
            move_before_patch(task_id="T4", before_task_id="T3"),
        )


def test_patch_rejects_stale_base_revision_without_partial_write():
    state = patch_state(plan_revision=2)
    before = deepcopy(state)

    with pytest.raises(PatchValidationError, match="base_plan_revision"):
        apply_plan_patch(state, update_patch(base_plan_revision=1))

    assert state == before


def test_patch_rejects_unknown_resource():
    state = patch_state(docs=[])
    patch = update_patch(changes={"use_resources": ["invented.csv"]})

    with pytest.raises(PatchValidationError, match="resource"):
        validate_plan_patch(state, patch)


def test_inserted_task_gets_unique_stable_id_and_resume_position():
    state = patch_state(cursor=2)
    patch = insert_before_patch(before_task_id="T3", inserted_task_id="T2A")

    update = apply_plan_patch(state, patch)

    assert [task["task_id"] for task in update["tasks"]] == ["T1", "T2", "T2A", "T3", "T4"]
    assert update["cursor"] == 2
    assert update["task_revisions"] == {"T1": 1, "T2": 1, "T3": 2, "T4": 1, "T2A": 1}


def test_insert_patch_normalizes_legacy_revision_counter_before_reordering():
    state = patch_state(cursor=2)
    state["task_revisions"] = {2: 5}

    update = apply_plan_patch(state, insert_before_patch())

    assert update["task_revisions"]["T3"] == 6
    assert update["task_revisions"]["T2A"] == 1


def test_insert_patch_normalizes_numeric_string_legacy_revision_counter_before_reordering():
    state = patch_state(cursor=2)
    state["task_revisions"] = {"2": 5}

    update = apply_plan_patch(state, insert_before_patch())

    assert update["task_revisions"]["T3"] == 6
    assert update["task_revisions"]["T2A"] == 1


def test_apply_is_atomic_and_counts_only_after_a_valid_patch():
    state = patch_state(task_patch_count={"T3": 0})
    before = deepcopy(state)

    with pytest.raises(PatchValidationError, match="unsupported"):
        apply_plan_patch(state, update_patch(operations=[{"op": "delete_task"}]))

    assert state == before
    update = apply_plan_patch(state, update_patch())
    assert update["task_patch_count"] == {"T3": 1}
    assert update["job_patch_count"] == 1


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        (update_patch(reason=" "), "reason"),
        (update_patch(reason_code=" "), "reason_code"),
        (update_patch(expected_resolution=" "), "expected_resolution"),
        (update_patch(resume_task_id="missing"), "resume_task_id"),
        (move_before_patch(resume_task_id="T3"), "earliest"),
        (update_patch(affected_task_ids=["T3", "T3"]), "duplicate"),
        (update_patch(affected_task_ids=["missing"]), "unknown task"),
        (update_patch(affected_task_ids=["T2"]), "affected_task_ids"),
        (update_patch(changes={"task_id": "T2"}), "not allowed"),
        (update_patch(changes={}), "changes"),
        (update_patch(operations=[{"op": "delete_task", "task_id": "T3"}]), "unsupported"),
    ],
)
def test_validate_rejects_invalid_patch_contract(patch, match):
    with pytest.raises(PatchValidationError, match=match):
        validate_plan_patch(patch_state(), patch)


def test_validate_rejects_duplicate_or_unknown_task_ids():
    with pytest.raises(PatchValidationError, match="duplicate task_id"):
        validate_plan_patch(patch_state(), insert_before_patch(inserted_task_id="T3"))

    with pytest.raises(PatchValidationError, match="unknown task"):
        validate_plan_patch(patch_state(), update_patch(task_id="missing"))


def test_validate_rejects_moving_an_accepted_task():
    with pytest.raises(PatchValidationError, match="accepted"):
        validate_plan_patch(patch_state(accepted_ids=["T4"]), move_before_patch())


def test_validate_rejects_moving_before_an_accepted_anchor():
    with pytest.raises(PatchValidationError, match="accepted anchor"):
        validate_plan_patch(patch_state(accepted_ids=["T3"]), move_before_patch())


def test_validate_rejects_moving_across_an_accepted_task():
    patch = move_before_patch(
        before_task_id="T2", affected_task_ids=["T4", "T2", "T3"]
    )

    with pytest.raises(PatchValidationError, match="accepted crossed"):
        validate_plan_patch(patch_state(accepted_ids=["T3"]), patch)


def test_move_requires_all_crossed_tasks_in_affected_task_ids():
    patch = move_before_patch(before_task_id="T2")

    with pytest.raises(PatchValidationError, match="affected_task_ids"):
        validate_plan_patch(patch_state(), patch)


def test_insert_rejects_an_accepted_anchor():
    with pytest.raises(PatchValidationError, match="accepted anchor"):
        validate_plan_patch(patch_state(accepted_ids=["T3"]), insert_before_patch())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("task_id", " ", "task_id"),
        ("task_name", " ", "task_name"),
        ("task_description", " ", "task_description"),
        ("task_type", "unsupported", "task_type"),
        ("use_rag", 1, "use_rag"),
        ("use_web", "true", "use_web"),
        ("allow_web_fallback", "true", "allow_web_fallback"),
        ("generate_table", None, "generate_table"),
        ("generate_figure", 0, "generate_figure"),
        ("query", 1, "query"),
        ("use_resources", "evidence.csv", "use_resources"),
        ("tool_requirements", ["tool", 1], "tool_requirements"),
        ("visualization", [], "visualization"),
    ],
)
def test_insert_rejects_invalid_task_schema(field, value, match):
    patch = insert_before_patch()
    patch["operations"][0]["task"][field] = value

    with pytest.raises(PatchValidationError, match=match):
        validate_plan_patch(patch_state(), patch)


def test_insert_rejects_unknown_and_internal_task_fields():
    patch = insert_before_patch()
    patch["operations"][0]["task"]["_recovery_allow_web"] = True

    with pytest.raises(PatchValidationError, match="unknown.*_recovery_allow_web"):
        validate_plan_patch(patch_state(), patch)


def test_insert_rejects_conclusion_that_would_enter_ordinary_worker():
    patch = insert_before_patch()
    inserted = patch["operations"][0]["task"]
    inserted.update(
        {
            "task_name": "结论与建议",
            "task_type": "analysis",
            "covers_sections": ["结论与建议"],
        }
    )

    with pytest.raises(PatchValidationError, match="must use task_type=synthesis"):
        validate_plan_patch(patch_state(), patch)


def test_insert_rejects_synthesis_for_non_aggregate_section():
    patch = insert_before_patch()
    inserted = patch["operations"][0]["task"]
    inserted.update(
        {
            "task_name": "异常案例总结与原因检索",
            "task_type": "synthesis",
            "use_rag": False,
            "query": "",
            "covers_sections": ["异常案例总结与原因检索"],
        }
    )

    with pytest.raises(PatchValidationError, match="reserved for conclusion"):
        validate_plan_patch(patch_state(), patch)


def test_insert_rejects_spider_tool_without_explicit_web_permission():
    patch = insert_before_patch()
    patch["operations"][0]["task"]["tool_requirements"] = ["SpiderTool"]

    with pytest.raises(PatchValidationError, match="spider_tool.*web"):
        validate_plan_patch(patch_state(), patch)


def test_update_rejects_spider_tool_after_simulating_all_task_changes():
    patch = update_patch(
        changes={"tool_requirements": ["spider_tool"], "use_web": False}
    )

    with pytest.raises(PatchValidationError, match="spider_tool.*web"):
        validate_plan_patch(patch_state(), patch)


@pytest.mark.parametrize(
    "changes",
    [
        {"tool_requirements": ["spider_tool"], "use_web": True},
        {"tool_requirements": ["SpiderTool"], "allow_web_fallback": True},
        {
            "tool_requirements": ["spider_tool"],
            "visualization": {"allow_web_fallback": True},
        },
    ],
)
def test_update_allows_spider_tool_with_an_explicit_web_permission(changes):
    validate_plan_patch(patch_state(), update_patch(changes=changes))


@pytest.mark.parametrize("requirement", ["", "   "])
def test_plan_patch_rejects_blank_tool_requirement_names(requirement):
    patch = update_patch(changes={"tool_requirements": [requirement]})

    with pytest.raises(PatchValidationError, match="invalid tool requirement"):
        validate_plan_patch(patch_state(), patch)


def test_plan_patch_canonicalizes_supported_tool_aliases_on_apply():
    update = apply_plan_patch(
        patch_state(),
        update_patch(
            changes={"tool_requirements": ["SpiderTool"], "use_web": True}
        ),
    )

    assert update["tasks"][2]["tool_requirements"] == ["spider_tool"]
    assert update["plan_patch_history"][0]["operations"][0]["changes"][
        "tool_requirements"
    ] == ["spider_tool"]


def test_plan_patch_preserves_spider_as_an_exact_custom_tool_name():
    update = apply_plan_patch(
        patch_state(),
        update_patch(
            changes={"tool_requirements": ["spider"], "use_web": False}
        ),
    )

    assert update["tasks"][2]["tool_requirements"] == ["spider"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("task_name", " ", "task_name"),
        ("task_description", 1, "task_description"),
        ("query", None, "query"),
        ("use_rag", "false", "use_rag"),
        ("use_web", 0, "use_web"),
        ("allow_web_fallback", 1, "allow_web_fallback"),
        ("generate_table", "no", "generate_table"),
        ("generate_figure", 1, "generate_figure"),
        ("use_resources", ["evidence.csv", 1], "use_resources"),
        ("tool_requirements", "worker", "tool_requirements"),
        ("visualization", "table", "visualization"),
    ],
)
def test_update_rejects_invalid_allowed_field_types(field, value, match):
    with pytest.raises(PatchValidationError, match=match):
        validate_plan_patch(patch_state(), update_patch(changes={field: value}))


def test_resource_aliases_normalize_to_canonical_paths_in_applied_tasks():
    state = patch_state(
        docs=[
            {
                "name": "evidence.csv",
                "path": "/job/evidence.csv",
                "file_id": "file-evidence",
            }
        ]
    )
    update = apply_plan_patch(
        state,
        update_patch(changes={"use_resources": [" file-evidence "]}),
    )

    assert update["tasks"][2]["use_resources"] == ["/job/evidence.csv"]
    assert update["plan_patch_history"][0]["operations"][0]["changes"]["use_resources"] == [
        "/job/evidence.csv"
    ]


def test_insert_normalizes_stripped_task_id_and_resource_aliases():
    state = patch_state(docs=[{"name": "evidence.csv", "path": "/job/evidence.csv"}])
    patch = insert_before_patch(inserted_task_id=" T2A ")
    patch["operations"][0]["task"]["use_resources"] = ["evidence.csv"]
    patch["affected_task_ids"] = ["T2A", "T3"]
    patch["resume_task_id"] = "T2A"

    update = apply_plan_patch(state, patch)

    assert update["tasks"][2]["task_id"] == "T2A"
    assert update["tasks"][2]["use_resources"] == ["/job/evidence.csv"]


def test_resource_aliases_shared_by_multiple_docs_are_rejected_as_ambiguous():
    state = patch_state(
        docs=[
            {"name": "evidence.csv", "path": "/job/first/evidence.csv"},
            {"name": "evidence.csv", "path": "/job/second/evidence.csv"},
        ]
    )

    with pytest.raises(PatchValidationError, match="ambiguous resource"):
        validate_plan_patch(state, update_patch(changes={"use_resources": ["evidence.csv"]}))


def test_validate_rejects_undeclared_accepted_result_invalidation():
    patch = update_patch(affected_task_ids=["T2", "T3"])

    with pytest.raises(PatchValidationError, match="accepted result"):
        validate_plan_patch(patch_state(accepted_ids=["T2"]), patch)


def test_move_and_update_apply_in_order_and_record_history():
    state = patch_state(cursor=2)
    patch = move_before_patch()

    update = apply_plan_patch(state, patch)

    assert [item["task_id"] for item in update["tasks"]] == ["T1", "T2", "T4", "T3"]
    assert update["cursor"] == 2
    assert update["task_revisions"] == {"T1": 1, "T2": 1, "T3": 2, "T4": 2}
    assert update["pending_user_action"] == {}
    assert update["plan_patch_history"] == [
        {
            "base_plan_revision": 1,
            "plan_revision": 2,
            "reason_code": "INVALID_TASK_ORDER",
            "reason": "The dependency needs to run first.",
            "affected_task_ids": ["T4", "T3"],
            "operations": patch["operations"],
            "resume_task_id": "T4",
            "expected_resolution": "The prerequisite precedes its dependent task.",
        }
    ]


def test_patch_rejects_one_hundred_inserts_atomically():
    state = patch_state()
    before = deepcopy(state)
    inserted_ids = [f"TX{index}" for index in range(100)]
    patch = insert_before_patch()
    patch["operations"] = [
        {
            "op": "insert_before",
            "before_task_id": "T3",
            "task": task(task_id),
        }
        for task_id in inserted_ids
    ]
    patch["affected_task_ids"] = [*inserted_ids, "T3"]
    patch["resume_task_id"] = inserted_ids[0]

    with pytest.raises(PatchValidationError, match="maximum|limit"):
        apply_plan_patch(state, patch)

    assert state == before


def test_patch_accepts_exactly_max_tasks_when_otherwise_valid():
    state = patch_state()
    state["tasks"] = [task(f"T{index}") for index in range(1, MAX_PLAN_TASKS + 1)]
    state["task_revisions"] = {
        item["task_id"]: 1 for item in state["tasks"]
    }
    patch = update_patch(task_id=f"T{MAX_PLAN_TASKS}")

    update = apply_plan_patch(state, patch)

    assert len(update["tasks"]) == MAX_PLAN_TASKS


def test_patch_rejects_growth_beyond_max_tasks_atomically():
    state = patch_state()
    state["tasks"] = [task(f"T{index}") for index in range(1, MAX_PLAN_TASKS + 1)]
    state["task_revisions"] = {
        item["task_id"]: 1 for item in state["tasks"]
    }
    before = deepcopy(state)
    patch = insert_before_patch(
        before_task_id=f"T{MAX_PLAN_TASKS}", inserted_task_id="T65"
    )
    patch["affected_task_ids"] = ["T65", f"T{MAX_PLAN_TASKS}"]

    with pytest.raises(PatchValidationError, match="maximum|limit"):
        apply_plan_patch(state, patch)

    assert state == before


@pytest.mark.parametrize(
    "counter_items",
    [
        [("2", 1), ("T3", 0)],
        [("T3", 0), ("2", 1)],
    ],
)
def test_patch_counter_aliases_merge_by_max_without_reopening_cap(counter_items):
    state = patch_state(task_patch_count=dict(counter_items))
    before = deepcopy(state)

    with pytest.raises(PatchValidationError, match="task patch limit"):
        apply_plan_patch(state, update_patch())

    assert state == before


def test_numeric_string_counter_prefers_exact_active_task_id_over_cursor_alias():
    state = patch_state(task_patch_count={"2": 1})
    state["tasks"] = [task(task_id) for task_id in ("2", "T2", "T3", "T4")]
    state["task_revisions"] = {
        item["task_id"]: 1 for item in state["tasks"]
    }
    before = deepcopy(state)

    with pytest.raises(PatchValidationError, match="task patch limit.*2"):
        apply_plan_patch(
            state,
            update_patch(task_id="2", affected_task_ids=["2"], resume_task_id="2"),
        )

    assert state == before


@pytest.mark.parametrize("legacy_key", [2, "2"])
def test_patch_counter_keeps_legacy_cursor_alias_when_no_exact_id_exists(legacy_key):
    state = patch_state(task_patch_count={legacy_key: 1})
    before = deepcopy(state)

    with pytest.raises(PatchValidationError, match="task patch limit.*T3"):
        apply_plan_patch(state, update_patch())

    assert state == before

from types import SimpleNamespace

from src import graph as graph_module
from src.graph import route_after_execution_blocker, route_planner, route_planner_confirm, route_workflow_policy
from src.nodes import synthesis as synthesis_module


def _task(task_id, task_type="analysis", **overrides):
    task = {
        "task_id": task_id,
        "task_name": "工艺分析" if task_type != "synthesis" else "结论",
        "task_description": "总结前文已经验收的结论。",
        "task_type": task_type,
        "use_rag": False,
        "use_web": False,
        "query": "",
        "use_resources": [],
        "generate_figure": False,
        "generate_table": False,
        "visualization": None,
        "covers_sections": ["结论" if task_type == "synthesis" else "工艺分析"],
    }
    task.update(overrides)
    return task


def _state():
    return {
        "tasks": [_task("T1"), _task("T2", "synthesis")],
        "cursor": 1,
        "results": [
            {
                "task_id": "T1",
                "text_output": "氢气比例会影响聚乙烯熔融指数，知识库给出的示例范围为2%。[E1]",
                "citations": [
                    {
                        "evidence_id": "E1",
                        "file_name": "聚乙烯生产工艺.pdf",
                        "file_path": "/srv/private/聚乙烯生产工艺.pdf",
                    }
                ],
                "plan_revision": 1,
                "task_revision": 1,
            }
        ],
        "section_status": {
            "T1": {
                "status": "VERIFIED_PASS",
                "accepted_by": "verifier",
                "issues": [],
                "plan_revision": 1,
                "task_revision": 1,
            }
        },
        "plan_revision": 1,
        "task_revisions": {"T1": 1, "T2": 1},
        "accepted_evidence_gaps": {},
    }


def test_build_context_only_uses_admitted_prior_sections_and_citations():
    context = synthesis_module.build_synthesis_context(_state())

    assert [section["task_id"] for section in context["accepted_sections"]] == ["T1"]
    assert context["accepted_evidence_ids"] == ["E1"]
    assert context["accepted_citations"][0]["file_name"] == "聚乙烯生产工艺.pdf"
    assert context["known_gaps"] == []


def test_build_context_scopes_colliding_local_evidence_ids_across_tasks():
    state = _state()
    state["tasks"].insert(1, _task("T_extra", task_name="压力分析"))
    state["cursor"] = 2
    state["task_revisions"]["T_extra"] = 1
    state["results"].append(
        {
            "task_id": "T_extra",
            "text_output": "压力变化影响产品密度。[E1]",
            "citations": [{"evidence_id": "E1", "file_name": "压力资料.pdf"}],
            "plan_revision": 1,
            "task_revision": 1,
        }
    )
    state["section_status"]["T_extra"] = {
        "status": "VERIFIED_PASS",
        "accepted_by": "verifier",
        "issues": [],
        "plan_revision": 1,
        "task_revision": 1,
    }

    context = synthesis_module.build_synthesis_context(state)

    assert context["accepted_evidence_ids"] == ["E1", "E2"]
    assert [
        citation["evidence_key"] for citation in context["accepted_citations"]
    ] == ["T1:E1", "T_extra:E1"]
    assert context["accepted_sections"][1]["content"].endswith("[E2]")


def test_build_context_keeps_user_accepted_draft_out_of_synthesis_fact_pool():
    state = _state()
    state["tasks"].insert(1, _task("T_gap", task_name="证据不足章节"))
    state["cursor"] = 2
    state["task_revisions"]["T_gap"] = 1
    state["results"].append(
        {
            "task_id": "T_gap",
            "text_output": "任务要求通过RAG检索，但当前证据不足。",
            "citations": [{"evidence_id": "E9", "file_name": "草稿.pdf"}],
            "plan_revision": 1,
            "task_revision": 1,
        }
    )
    state["section_status"]["T_gap"] = {
        "status": "USER_ACCEPTED_WARNING",
        "accepted_by": "user",
        "issues": [{"code": "TOO_LONG", "description": "用户接受超长草稿"}],
        "plan_revision": 1,
        "task_revision": 1,
    }

    context = synthesis_module.build_synthesis_context(state)

    assert [section["task_id"] for section in context["verified_sections"]] == ["T1"]
    assert [section["task_id"] for section in context["warning_sections"]] == ["T_gap"]
    assert [section["task_id"] for section in context["accepted_sections"]] == ["T1"]
    assert "E9" not in context["accepted_evidence_ids"]


def test_build_context_excludes_unaccepted_and_revision_mismatched_results():
    state = _state()
    state["section_status"]["T1"]["task_revision"] = 2

    context = synthesis_module.build_synthesis_context(state)

    assert context["accepted_sections"] == []
    assert context["accepted_citations"] == []


def test_build_context_rejects_status_and_result_that_are_both_stale():
    state = _state()
    state["section_status"]["T1"]["task_revision"] = 2
    state["results"][0]["task_revision"] = 2
    state["task_revisions"]["T1"] = 3

    context = synthesis_module.build_synthesis_context(state)

    assert context["accepted_sections"] == []


def test_consistency_gate_rejects_new_facts_and_phantom_actions():
    context = synthesis_module.build_synthesis_context(_state())
    issues = synthesis_module.check_synthesis_consistency(
        "补充检索和正交试验表明，pH控制在3%时效果最好。[E9]",
        context,
    )

    assert {issue["code"] for issue in issues} == {
        "NEW_EVIDENCE_ID",
        "NEW_NUMBER",
        "NEW_TECHNICAL_IDENTIFIER",
        "PHANTOM_ACTION",
        "UNGROUNDED_CLAIM",
    }


def test_consistency_gate_rejects_real_chinese_conclusion_drift_and_heading_bypass():
    context = synthesis_module.build_synthesis_context(_state())

    issues = synthesis_module.check_synthesis_consistency(
        "### pH 3%控制策略\n目标产物结晶、产品纯度和串级控制是后续优化重点。"
        "建议建立先进控制策略并新增在线色谱监测。",
        context,
    )

    codes = {issue["code"] for issue in issues}
    assert "NEW_NUMBER" in codes
    assert "NEW_TECHNICAL_IDENTIFIER" in codes
    assert "UNGROUNDED_CLAIM" in codes


def test_consistency_gate_rejects_causal_reversal_relation_swap_and_short_action():
    context = {
        "accepted_sections": [
            {
                "content": "氢气比例升高会使熔融指数升高。反应温度影响密度。压力决定密度。"
            }
        ],
        "accepted_evidence_ids": [],
    }

    for candidate in (
        "氢气比例升高会使熔融指数降低。",
        "反应温度影响熔融指数，氢气比例影响密度。",
        "应立即停车。",
    ):
        issues = synthesis_module.check_synthesis_consistency(candidate, context)
        assert "UNGROUNDED_CLAIM" in {issue["code"] for issue in issues}


def test_consistency_gate_rejects_removing_negation_uncertainty_and_conditions():
    context = {
        "accepted_sections": [
            {
                "content": (
                    "现有证据不足，不能认为压力升高会导致密度下降。"
                    "尚无证据表明温度升高会导致分子量降低。"
                    "除非完成现场验证，否则不得立即停车。"
                )
            }
        ],
        "accepted_evidence_ids": [],
    }

    for candidate in (
        "压力升高会导致密度下降。",
        "温度升高会导致分子量降低。",
        "立即停车。",
    ):
        issues = synthesis_module.check_synthesis_consistency(candidate, context)
        assert "UNGROUNDED_CLAIM" in {issue["code"] for issue in issues}


def test_consistency_gate_rejects_rebinding_accepted_evidence_ids():
    context = {
        "accepted_sections": [
            {"content": "结论甲。[E1]\n结论乙。[E2]"}
        ],
        "accepted_evidence_ids": ["E1", "E2"],
    }

    issues = synthesis_module.check_synthesis_consistency(
        "结论甲。[E2]\n结论乙。[E1]",
        context,
    )

    assert "CITATION_REBIND" in {issue["code"] for issue in issues}


def test_consistency_gate_allows_exact_user_accepted_gap_description():
    context = synthesis_module.build_synthesis_context(_state())
    context["known_gaps"] = [
        {"task_id": "T1", "description": "知识库未提供反应压力的直接因果证据。"}
    ]

    issues = synthesis_module.check_synthesis_consistency(
        "知识库未提供反应压力的直接因果证据。",
        context,
    )

    assert issues == []


def test_synthesis_retries_once_then_fails_closed(monkeypatch):
    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content="补充检索发现pH为3%。[E9]")

    model = Model()
    modes = []

    def fake_get_llm(_config, *, json_mode):
        modes.append(json_mode)
        return model

    monkeypatch.setattr(synthesis_module, "get_llm", fake_get_llm)

    update = synthesis_module.synthesis(_state(), {})

    result = update["current_result"]
    assert model.calls == 2
    assert modes == [False]
    assert result["synthesis_audit"]["fallback_used"] is False
    assert result["status"] == "FAILED"
    assert result["error"] == "SYNTHESIS_CONSISTENCY_FAILED"
    assert result["text_output"] == ""
    assert result["citations"] == []
    assert result["tool_calls"] == []


def test_synthesis_model_failure_fails_closed(monkeypatch):
    class Model:
        def invoke(self, _messages, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        synthesis_module,
        "get_llm",
        lambda *_args, **_kwargs: Model(),
    )

    result = synthesis_module.synthesis(_state(), {})["current_result"]

    assert result["synthesis_audit"]["fallback_used"] is False
    assert result["synthesis_audit"]["model_error"] == "provider unavailable"
    assert result["status"] == "FAILED"
    assert result["text_output"] == ""


def test_synthesis_accepts_second_grounded_attempt(monkeypatch):
    responses = iter(
        [
            "产品纯度和串级控制是后续重点。",
            "氢气比例会影响聚乙烯熔融指数，知识库给出的示例范围为2%。[E1]",
        ]
    )

    class Model:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(content=next(responses))

    monkeypatch.setattr(
        synthesis_module,
        "get_llm",
        lambda *_args, **_kwargs: Model(),
    )

    result = synthesis_module.synthesis(_state(), {})["current_result"]

    assert result["synthesis_audit"]["attempts"] == 2
    assert result["synthesis_audit"]["fallback_used"] is False
    assert result["text_output"] == (
        "氢气比例会影响聚乙烯熔融指数，知识库给出的示例范围为2%。[E1]"
    )


def test_synthesis_with_no_admitted_context_does_not_call_model(monkeypatch):
    state = _state()
    state["section_status"] = {}
    called = False

    def fake_get_llm(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("model must not be called")

    monkeypatch.setattr(synthesis_module, "get_llm", fake_get_llm)

    result = synthesis_module.synthesis(state, {})["current_result"]

    assert called is False
    assert result["status"] == "FAILED"
    assert result["error"] == "NO_ACCEPTED_SECTIONS"
    assert result["text_output"] == ""


def test_graph_routes_synthesis_initial_and_rework_without_worker():
    state = _state()
    state["planner_action"] = "PROCEED"
    assert route_planner(state) == "Synthesis"
    assert route_planner_confirm(state) == "Synthesis"

    state["workflow_action"] = "REWORK"
    assert route_workflow_policy(state) == "SYNTHESIS_REWORK"
    assert route_after_execution_blocker(state) == "SYNTHESIS_REWORK"


def test_manual_verifier_retry_routes_synthesis_without_worker(monkeypatch):
    state = _state()
    monkeypatch.setattr(graph_module, "decision", lambda _state: "RETRY_WORKER")

    assert graph_module.route_manual_verifier(state) == "RETRY_SYNTHESIS"

    state["workflow_action"] = "EVIDENCE_RECOVERY"
    assert route_workflow_policy(state) == "SYNTHESIS_REWORK"
    assert route_after_execution_blocker(state) == "SYNTHESIS_REWORK"

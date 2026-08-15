import ast
import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage
import pytest

from src.nodes.worker.agent.graph import AutonomousToolNode, ToolManager, router_node

try:
    recovery_module = importlib.import_module("src.nodes.recovery")
except ModuleNotFoundError:
    recovery_module = None


def _missing_recovery(*args, **kwargs):
    raise AssertionError("src.nodes.recovery is not implemented")


decision_policy = getattr(recovery_module, "decision_policy", _missing_recovery)
evidence_recovery = getattr(recovery_module, "evidence_recovery", _missing_recovery)
automatic_planner = getattr(recovery_module, "automatic_planner", _missing_recovery)
needs_user_input = getattr(recovery_module, "needs_user_input", _missing_recovery)
plan_patcher = getattr(recovery_module, "plan_patcher", _missing_recovery)
route_after_blocker = getattr(recovery_module, "route_after_blocker", _missing_recovery)
route_policy = getattr(recovery_module, "route_policy", _missing_recovery)


def _task(task_id, **overrides):
    return {
        "task_id": task_id,
        "task_name": f"Task {task_id}",
        "task_description": f"Write {task_id}",
        "task_type": "analysis",
        "use_rag": True,
        "use_web": False,
        "generate_table": False,
        "generate_figure": False,
        "query": f"query {task_id}",
        "use_resources": [],
        **overrides,
    }


def graph_state(*, cursor=1, accepted_ids=("T1",), **overrides):
    tasks = [_task("T1"), _task("T2"), _task("T3")]
    return {
        "tasks": tasks,
        "cursor": cursor,
        "docs": [{"file_id": "F1", "name": "existing.pdf"}],
        "current_result": {
            "task_id": tasks[cursor]["task_id"],
            "text_output": "candidate",
        },
        "results": [
            {"task_id": task_id, "text_output": f"accepted {task_id}"}
            for task_id in accepted_ids
        ],
        "section_status": {
            task_id: {"status": "VERIFIED_PASS"} for task_id in accepted_ids
        },
        "assessment": {},
        "worker_state": {"next_node": "end", "retained": True},
        "task_retry_count": {},
        "evidence_recovery_count": {},
        "task_patch_count": {},
        "job_patch_count": 0,
        "plan_revision": 1,
        "task_revisions": {"T1": 1, "T2": 1, "T3": 1},
        "pending_user_action": {},
        "plan_patch_history": [],
        "verification_warnings": [],
        **overrides,
    }


def evidence_gap_assessment():
    return {
        "status": "BLOCKED",
        "issues": [
            {
                "code": "EVIDENCE_GAP",
                "category": "EVIDENCE_GAP",
                "description": "Missing authoritative evidence for catalyst life",
                "suggestion": "Search broader terminology",
                "severity": "major",
                "retrieval_query": "catalyst life deactivation mechanism",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["catalyst life evidence", "deactivation mechanism"],
    }


def test_t2_evidence_gap_never_routes_to_planner_or_resets_cursor(caplog):
    caplog.set_level("INFO", logger="src.nodes.recovery")
    state = graph_state(cursor=1, accepted_ids=("T1",))
    update = decision_policy({**state, "assessment": evidence_gap_assessment()}, {})

    assert update["workflow_action"] == "EVIDENCE_RECOVERY"
    assert update.get("cursor", state["cursor"]) == 1
    assert [item["task_id"] for item in state["results"]] == ["T1"]
    assert route_policy({**state, **update}) == "EVIDENCE_RECOVERY"
    assert (
        "Workflow policy decision: source=system task=T2 "
        "assessment_status=BLOCKED action=EVIDENCE_RECOVERY"
    ) in caplog.messages


def test_length_policy_creates_structured_execution_feedback():
    assessment = {
        "status": "FAILED",
        "issues": [
            {
                "code": "TOO_SHORT",
                "category": "CONTENT_DEFECT",
                "description": "Section is too short",
                "suggestion": "Add mechanism detail",
                "severity": "major",
            }
        ],
    }
    state = graph_state(assessment=assessment)

    update = decision_policy(state, {})

    feedback = update["worker_state"]["execution_feedback"]
    assert feedback["mode"] == "length_rewrite"
    assert feedback["issues"] == assessment["issues"]
    assert "Add mechanism detail" in feedback["instructions"]
    assert feedback["source_result"] == state["current_result"]
    assert update["worker_state"]["retained"] is True


def test_retry_exhaustion_routes_to_user_input_instead_of_continuing():
    state = graph_state(
        task_retry_count={"T2": 2},
        assessment={
            "status": "FAILED",
            "issues": [{"code": "TOO_SHORT", "category": "CONTENT_DEFECT"}],
        },
    )

    update = decision_policy(state, {})

    assert update["workflow_action"] == "NEEDS_USER_INPUT"
    assert route_policy({**state, **update}) == "NEEDS_USER_INPUT"
    assert update["section_status"]["T2"]["status"] == "ACCEPT_WITH_WARNING"
    assert [result["task_id"] for result in state["results"]] == ["T1"]


def test_evidence_recovery_builds_query_and_honors_task_web_gate():
    state = graph_state(assessment=evidence_gap_assessment())
    original_tasks = deepcopy(state["tasks"])

    update = evidence_recovery(state, {})

    feedback = update["worker_state"]["execution_feedback"]
    assert feedback["mode"] == "evidence_recovery"
    plan = feedback["recovery_plan"]
    assert plan["task_id"] == "T2"
    assert plan["plan_revision"] == 1
    assert plan["task_revision"] == 1
    assert plan["recovery_sequence"] == 1
    assert plan["evidence_queries"] == ["catalyst life deactivation mechanism"]
    assert feedback["allow_web"] is False
    assert state["tasks"] == original_tasks

    web_state = graph_state(assessment=evidence_gap_assessment())
    web_state["tasks"][1]["visualization"] = {"allow_web_fallback": True}
    web_update = evidence_recovery(web_state, {})
    assert web_update["worker_state"]["execution_feedback"]["allow_web"] is True


def test_evidence_recovery_filters_non_evidence_issues_and_orders_dependent_assets():
    state = graph_state(
        assessment={
            "status": "FAILED",
            "issues": [
                {
                    "code": "EVIDENCE_GAP",
                    "category": "EVIDENCE_GAP",
                    "description": "缺少反应压力对熔融指数影响的直接证据",
                    "suggestion": "检索反应压力与熔融指数的因果关系",
                    "severity": "major",
                    "retrieval_query": "聚乙烯 反应压力 熔融指数 影响机理",
                },
                {
                    "code": "MISSING_FIGURE",
                    "category": "CONTENT_DEFECT",
                    "description": "缺少正式因果图资产",
                    "suggestion": "生成因果图",
                    "severity": "major",
                },
                {
                    "code": "MISSING_TABLE",
                    "category": "CONTENT_DEFECT",
                    "description": "Markdown 表格没有结构化资产",
                    "suggestion": "转换表格资产",
                    "severity": "major",
                },
            ],
            "requirements_missing": [
                "结构化资产中未提供 table 和 figure 对象",
                "反应压力证据",
            ],
        }
    )
    state["tasks"][1]["generate_figure"] = True
    state["tasks"][1]["visualization"] = {
        "kind": "causal",
        "required_concepts": ["反应压力", "熔融指数"],
    }

    update = evidence_recovery(state, {})

    plan = update["worker_state"]["execution_feedback"]["recovery_plan"]
    assert plan["evidence_queries"] == ["聚乙烯 反应压力 熔融指数 影响机理"]
    assert not any("table" in query or "figure" in query for query in plan["evidence_queries"])
    assert {
        "asset": "causal_figure",
        "action": "regenerate",
        "after": "evidence_recovery",
    } in plan["asset_actions"]
    assert {
        "asset": "table",
        "action": "materialize",
    } in plan["asset_actions"]


def test_recovery_plan_never_falls_back_to_issue_description_for_query():
    state = graph_state(
        assessment={
            "status": "FAILED",
            "issues": [
                {
                    "code": "EVIDENCE_GAP",
                    "category": "EVIDENCE_GAP",
                    "description": "任务要求鱼眼检测方法，但正文没有完成。",
                    "suggestion": "补充知识库依据。",
                    "severity": "major",
                }
            ],
        }
    )

    plan = evidence_recovery(state, {})["worker_state"]["execution_feedback"][
        "recovery_plan"
    ]

    assert plan["evidence_queries"] == []


def test_recovery_plan_normalizes_deduplicates_and_merges_contained_queries():
    issues = []
    for query in (
        "  聚乙烯   鱼眼 定义 检测方法  ",
        "聚乙烯 鱼眼 检测方法",
        "聚乙烯 鱼眼 定义 检测方法",
    ):
        issues.append(
            {
                "code": "EVIDENCE_GAP",
                "category": "EVIDENCE_GAP",
                "description": "缺少证据",
                "suggestion": "补充证据",
                "severity": "major",
                "retrieval_query": query,
            }
        )
    state = graph_state(assessment={"status": "FAILED", "issues": issues})

    plan = evidence_recovery(state, {})["worker_state"]["execution_feedback"][
        "recovery_plan"
    ]

    assert plan["evidence_queries"] == ["聚乙烯 鱼眼 定义 检测方法"]


def test_recovery_query_merge_is_transitive_and_order_independent():
    queries = recovery_module._merge_retrieval_queries(
        ["A B", "C D", "A B C D"]
    )
    reordered = recovery_module._merge_retrieval_queries(
        ["A B C D", "C D", "A B"]
    )

    assert queries == ["A B C D"]
    assert reordered == ["A B C D"]


def test_recovery_plan_sequence_uses_the_updated_bounded_counter():
    state = graph_state(
        evidence_recovery_count={"T2": 1},
        assessment=evidence_gap_assessment(),
    )

    update = evidence_recovery(state, {})

    plan = update["worker_state"]["execution_feedback"]["recovery_plan"]
    assert plan["recovery_sequence"] == 1
    assert plan["recovery_id"] == "T2:p1:t1:evidence_recovery:1"


def test_citation_binding_issue_reuses_evidence_without_generating_rag_query():
    state = graph_state(
        assessment={
            "status": "FAILED",
            "issues": [
                {
                    "code": "MISSING_INLINE_CITATION",
                    "category": "EVIDENCE_GAP",
                    "description": "正文已有结构化证据但没有绑定 [E编号]",
                    "suggestion": "把现有证据编号绑定到相邻论断",
                    "severity": "major",
                }
            ],
            "requirements_met": [],
            "requirements_missing": ["正文中的证据编号绑定"],
        }
    )

    update = evidence_recovery(state, {})

    plan = update["worker_state"]["execution_feedback"]["recovery_plan"]
    assert plan["evidence_queries"] == []


@pytest.mark.parametrize(
    "invalid_flags",
    [
        {"use_web": "false", "tool_requirements": ["SpiderTool"]},
        {"allow_web_fallback": 1, "tool_requirements": ["spider_tool"]},
        {
            "visualization": {"allow_web_fallback": "true"},
            "tool_requirements": ["spider_tool"],
        },
    ],
)
def test_evidence_recovery_fails_closed_for_invalid_web_flag_types(invalid_flags):
    state = graph_state(assessment=evidence_gap_assessment())
    state["tasks"][1].update(invalid_flags)

    update = evidence_recovery(state, {})
    feedback = update["worker_state"]["execution_feedback"]
    execution_task, _, _ = AutonomousToolNode._prepare_execution_task(
        state["tasks"][1], update["worker_state"]
    )

    assert feedback["allow_web"] is False
    assert execution_task["use_web"] is False
    assert "spider_tool" not in {
        canonical_tool_name(requirement)
        for requirement in execution_task.get("tool_requirements", [])
    }


@pytest.mark.parametrize(
    "allowed_flags",
    [
        {"use_web": True},
        {"allow_web_fallback": True},
        {"visualization": {"allow_web_fallback": True}},
    ],
)
def test_evidence_recovery_preserves_explicit_web_authorization(allowed_flags):
    state = graph_state(assessment=evidence_gap_assessment())
    state["tasks"][1].update(allowed_flags)

    update = evidence_recovery(state, {})

    assert update["worker_state"]["execution_feedback"]["allow_web"] is True


def test_plan_patcher_validates_and_applies_only_local_patch(monkeypatch):
    state = graph_state(
        docs=[{"name": "evidence.csv", "path": "/job/evidence.csv"}],
        assessment={
            "status": "BLOCKED",
            "issues": [
                {
                    "code": "RESOURCE_NOT_ASSIGNED",
                    "category": "LOCAL_PLAN_DEFECT",
                    "description": "Existing evidence is not assigned",
                    "suggestion": "Assign evidence.csv to T2",
                    "resource_name": "evidence.csv",
                    "severity": "major",
                }
            ],
        },
    )
    patch = {
        "base_plan_revision": 1,
        "reason_code": "RESOURCE_NOT_ASSIGNED",
        "reason": "T2 needs the existing evidence file.",
        "affected_task_ids": ["T2"],
        "operations": [
            {
                "op": "update_task",
                "task_id": "T2",
                "changes": {"use_resources": ["evidence.csv"]},
            }
        ],
        "resume_task_id": "T2",
        "expected_resolution": "T2 can read the evidence.",
    }

    class Model:
        def invoke(self, messages):
            return SimpleNamespace(content=f"```json\n{json.dumps(patch)}\n```")

    monkeypatch.setattr(recovery_module, "get_llm", lambda *args, **kwargs: Model())

    update = plan_patcher(state, {})

    assert update["workflow_action"] == "REWORK"
    assert update["tasks"][1]["use_resources"] == ["/job/evidence.csv"]
    assert [item["task_id"] for item in update["results"]] == ["T1"]
    assert update["cursor"] == 1
    assert update["job_patch_count"] == 1


def test_plan_patcher_model_or_validation_error_requires_user_input(monkeypatch):
    state = graph_state(assessment={"status": "BLOCKED", "issues": []})

    class Model:
        def invoke(self, messages):
            return SimpleNamespace(content='{"tasks": []}')

    monkeypatch.setattr(recovery_module, "get_llm", lambda *args, **kwargs: Model())

    update = plan_patcher(state, {})

    assert update["workflow_action"] == "NEEDS_USER_INPUT"
    assert update["pending_user_action"]["category"] == "PLAN_PATCH_ERROR"
    assert "guidance" in update["pending_user_action"]
    assert update["job_patch_count"] == 0
    assert "tasks" not in update


def test_plan_patcher_prompt_contains_each_task2_operation_schema():
    prompt_path = Path(__file__).parents[1] / "src" / "prompts" / "planner_patch.md"
    prompt = prompt_path.read_text(encoding="utf-8")

    assert '"op": "update_task"' in prompt
    assert '"op": "move_before"' in prompt
    assert '"op": "insert_before"' in prompt
    assert '"before_task_id"' in prompt
    assert '"task_type": "analysis"' in prompt
    assert "task_name, task_description, task_type, query, use_rag, use_web" in prompt
    assert "completed or accepted task" in prompt


def test_needs_user_input_interrupt_payload_and_incremental_resume(monkeypatch):
    captured = {}
    resumed_doc = {"file_id": "F2", "name": "new.pdf"}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"text": "Use the uploaded source", "docs": [resumed_doc]}

    monkeypatch.setattr(recovery_module, "interrupt", fake_interrupt)
    state = graph_state(
        pending_user_action={
            "category": "EXTERNAL_BLOCKER",
            "task_id": "T2",
            "issues": [{"code": "RESOURCE_UNAVAILABLE"}],
        }
    )

    update = needs_user_input(state, {})

    assert captured["type"] == "needs_user_input"
    assert captured["category"] == "EXTERNAL_BLOCKER"
    assert captured["affected_task"] == "T2"
    assert captured["blocker_status"] == "ACTIVE"
    assert captured["blocker_id"]
    assert captured["accepted_choices"]
    assert update["workflow_action"] == "REWORK"
    assert update["docs"] == [resumed_doc]
    assert "cursor" not in update
    assert route_after_blocker({**state, **update}) == "REWORK"
    assert update["worker_state"]["execution_feedback"]["instructions"] == "Use the uploaded source"


def test_evidence_blocker_resume_returns_to_evidence_recovery_without_cursor_reset(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"text": "broaden search", "docs": []},
    )
    state = graph_state(
        cursor=1,
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        },
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "EVIDENCE_RECOVERY"
    assert "cursor" not in update
    assert "docs" in update and update["docs"] == []


def test_evidence_blocker_exposes_specific_user_resolution_choices(monkeypatch):
    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "ACCEPT_EVIDENCE_GAP", "text": "", "docs": []}

    monkeypatch.setattr(recovery_module, "interrupt", fake_interrupt)
    state = graph_state(
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        }
    )

    update = needs_user_input(state, {})

    assert captured["accepted_choices"] == [
        "UPLOAD_RESOURCES",
        "AUTHORIZE_WEB",
        "ADJUST_REQUIREMENT",
        "ACCEPT_EVIDENCE_GAP",
    ]
    assert "catalyst life" in captured["guidance_text"]
    assert update["workflow_action"] == "NEXT"
    assert [result["task_id"] for result in update["results"]] == ["T1", "T2"]
    assert update["section_status"]["T2"]["status"] == "USER_ACCEPTED_GAP"
    assert update["report_status"] == "BLOCKED"


def test_accepting_evidence_gap_reworks_remaining_content_defect(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"action": "ACCEPT_EVIDENCE_GAP", "text": "", "docs": []},
    )
    evidence_issue = evidence_gap_assessment()["issues"][0]
    length_issue = {
        "code": "TOO_LONG",
        "category": "CONTENT_DEFECT",
        "description": "正文超过最高字数要求",
        "suggestion": "压缩正文",
        "severity": "major",
    }
    state = graph_state(
        assessment={"status": "FAILED", "issues": [evidence_issue, length_issue]},
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": [evidence_issue, length_issue],
            "accepted_choices": [
                "UPLOAD_RESOURCES",
                "AUTHORIZE_WEB",
                "ADJUST_REQUIREMENT",
                "ACCEPT_EVIDENCE_GAP",
            ],
        },
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "LENGTH_REWRITE"
    assert "results" not in update
    assert update["task_retry_count"] == {"T2": 1}
    feedback = update["worker_state"]["execution_feedback"]
    assert [issue["code"] for issue in feedback["issues"]] == ["TOO_LONG"]
    waiver = update["accepted_evidence_gaps"]["T2"]
    assert waiver["plan_revision"] == 1
    assert waiver["task_revision"] == 1
    assert [issue["code"] for issue in waiver["issues"]] == ["EVIDENCE_GAP"]
    assert update["section_status"]["T2"]["status"] == "BLOCKED"


def test_length_rewrite_feedback_is_tool_free_and_preserves_source_result():
    state = graph_state(
        assessment={
            "status": "FAILED",
            "issues": [
                {
                    "code": "TOO_LONG",
                    "category": "CONTENT_DEFECT",
                    "description": "正文超过最高字数要求",
                    "suggestion": "压缩正文",
                    "severity": "major",
                }
            ],
        }
    )

    update = recovery_module.decision_policy(state, {})

    assert update["workflow_action"] == "LENGTH_REWRITE"
    feedback = update["worker_state"]["execution_feedback"]
    assert feedback["mode"] == "length_rewrite"
    assert feedback["source_result"] == state["current_result"]


def test_user_can_explicitly_accept_content_warning_as_draft(monkeypatch, caplog):
    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "ACCEPT_AS_DRAFT", "text": "", "docs": []}

    monkeypatch.setattr(recovery_module, "interrupt", fake_interrupt)
    caplog.set_level("INFO", logger="src.nodes.recovery")
    state = graph_state(
        pending_user_action={
            "category": "CONTENT_DEFECT",
            "task_id": "T2",
            "issues": [{"code": "TOO_SHORT", "description": "篇幅不足"}],
            "accepted_choices": [
                "REWORK",
                "ADJUST_REQUIREMENT",
                "ACCEPT_AS_DRAFT",
                "DONE",
            ],
        }
    )

    update = needs_user_input(state, {})

    assert "ACCEPT_AS_DRAFT" in captured["accepted_choices"]
    assert update["workflow_action"] == "NEXT"
    assert [result["task_id"] for result in update["results"]] == ["T1", "T2"]
    assert update["section_status"]["T2"]["status"] == "USER_ACCEPTED_WARNING"
    assert (
        "User blocker decision: blocker=T2:p1:legacy task=T2 category=CONTENT_DEFECT "
        "choice=ACCEPT_AS_DRAFT action=NEXT uploaded_files=false"
    ) in caplog.messages


def test_terminal_task_accept_as_draft_commits_once_and_finishes(monkeypatch, caplog):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"action": "ACCEPT_AS_DRAFT", "text": "", "docs": []},
    )
    caplog.set_level("INFO", logger="src.nodes.recovery")
    state = graph_state(
        cursor=2,
        accepted_ids=("T1", "T2"),
        pending_user_action={
            "category": "CONTENT_DEFECT",
            "task_id": "T3",
            "issues": [{"code": "CONTENT_DEFECT", "description": "结论存在缺口"}],
            "accepted_choices": ["REWORK", "ACCEPT_AS_DRAFT", "DONE"],
        },
    )
    state["section_status"] = {
        task_id: {
            "status": "VERIFIED_PASS",
            "accepted_by": "verifier",
            "issues": [],
            "plan_revision": 1,
            "task_revision": 1,
        }
        for task_id in ("T1", "T2")
    }

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "DONE"
    assert [result["task_id"] for result in update["results"]] == ["T1", "T2", "T3"]
    assert update["section_status"]["T3"]["status"] == "USER_ACCEPTED_WARNING"
    assert update["report_status"] == "DRAFT_WITH_GAPS"
    assert route_after_blocker({**state, **update}) == "DONE"
    assert (
        "User blocker decision: blocker=T3:p1:legacy task=T3 category=CONTENT_DEFECT "
        "choice=ACCEPT_AS_DRAFT action=DONE uploaded_files=false"
    ) in caplog.messages


def test_authorize_web_is_execution_only_and_explicit(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"action": "AUTHORIZE_WEB", "text": "", "docs": []},
    )
    state = graph_state(
        web_authorized=False,
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        },
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "REWORK"
    assert update["web_authorized"] is True
    assert update["tasks"][1]["use_web"] is True
    assert state["tasks"][1]["use_web"] is False
    feedback = update["worker_state"]["execution_feedback"]
    assert feedback["allow_web"] is True
    assert feedback["recovery_plan"]["evidence_queries"] == [
        "catalyst life deactivation mechanism"
    ]


def test_special_resume_choice_cannot_bypass_pending_accepted_choices(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"action": "AUTHORIZE_WEB", "text": "", "docs": []},
    )
    state = graph_state(
        web_authorized=False,
        pending_user_action={
            "category": "EXTERNAL_BLOCKER",
            "task_id": "T2",
            "issues": [{"code": "RESOURCE_UNAVAILABLE"}],
            "accepted_choices": ["REWORK"],
        },
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "REWORK"
    assert "web_authorized" not in update
    assert "tasks" not in update


def test_upload_resources_uses_category_appropriate_resume_route(monkeypatch):
    ingested = []
    monkeypatch.setattr(
        recovery_module,
        "_ingest_uploaded_evidence",
        lambda docs: ingested.extend(docs),
    )
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {
            "action": "UPLOAD_RESOURCES",
            "text": "",
            "docs": [{"file_id": "F2", "name": "new.pdf"}],
        },
    )
    state = graph_state(
        pending_user_action={
            "category": "EXTERNAL_BLOCKER",
            "task_id": "T2",
            "issues": [{"code": "RESOURCE_UNAVAILABLE"}],
            "accepted_choices": ["UPLOAD_RESOURCES", "REWORK"],
        }
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "REWORK"
    assert [doc["file_id"] for doc in update["docs"]] == ["F1", "F2"]
    assert ingested == []


def test_evidence_upload_is_ingested_and_assigned_before_recovery(monkeypatch):
    uploaded = {
        "file_id": "F2",
        "name": "standard.pdf",
        "path": "/tmp/standard.pdf",
    }
    ingested = []
    monkeypatch.setattr(
        recovery_module,
        "_ingest_uploaded_evidence",
        lambda docs: ingested.extend(docs),
    )
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {
            "text": "我已上传附件，请结合附件继续处理当前任务。",
            "docs": [uploaded],
        },
    )
    state = graph_state(
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        }
    )

    update = needs_user_input(state, {})

    assert ingested == [uploaded]
    assert update["workflow_action"] == "EVIDENCE_RECOVERY"
    assert update["tasks"][1]["use_rag"] is True
    assert update["tasks"][1]["use_resources"] == ["/tmp/standard.pdf"]


def test_uploaded_evidence_helper_calls_rag_ingestion(monkeypatch):
    from src.rag import service as rag_service_module

    calls = []

    class Service:
        def ingest(self, paths):
            calls.append(paths)
            return {
                "success": True,
                "loaded_files": 1,
                "loaded_with_warnings_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "total_chunks": 4,
            }

    monkeypatch.setattr(rag_service_module, "ChemicalRAGService", Service)

    result = recovery_module._ingest_uploaded_evidence(
        [{"path": "/tmp/standard.pdf"}]
    )

    assert calls == [["/tmp/standard.pdf"]]
    assert result["total_chunks"] == 4


def test_adjust_requirement_changes_only_current_task_and_bumps_revision(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {
            "action": "ADJUST_REQUIREMENT",
            "text": "仅报告知识库已有定义，并明确列出缺口。",
            "docs": [],
        },
    )
    state = graph_state(
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        }
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "REWORK"
    assert "仅报告知识库已有定义" in update["tasks"][1]["task_description"]
    assert update["tasks"][0] == state["tasks"][0]
    assert update["task_revisions"]["T2"] == 2
    assert state["task_revisions"]["T2"] == 1


def test_adjust_requirement_rejects_missing_concrete_requirement(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {
            "action": "ADJUST_REQUIREMENT",
            "text": "调整任务要求",
            "docs": [],
        },
    )
    state = graph_state(
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        }
    )

    with pytest.raises(ValueError, match="调整任务要求："):
        needs_user_input(state, {})


def test_streamlit_text_resume_understands_user_facing_evidence_choices(monkeypatch):
    state = graph_state(
        web_authorized=False,
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        },
    )
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"text": "授权公开网络检索", "docs": []},
    )

    web_update = needs_user_input(state, {})

    assert web_update["web_authorized"] is True
    assert web_update["worker_state"]["execution_feedback"]["allow_web"] is True

    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {
            "text": "调整任务要求：仅报告知识库已有证据及缺口",
            "docs": [],
        },
    )

    adjusted = needs_user_input(state, {})

    assert adjusted["workflow_action"] == "REWORK"
    assert adjusted["tasks"][1]["task_description"].endswith(
        "用户明确调整的要求：仅报告知识库已有证据及缺口"
    )


def test_evidence_blocker_honors_explicit_rework_resume(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {
            "action": "REWORK",
            "text": "Rewrite only from the evidence already available",
            "docs": [],
        },
    )
    state = graph_state(
        pending_user_action={
            "category": "EVIDENCE_GAP",
            "task_id": "T2",
            "issues": evidence_gap_assessment()["issues"],
        }
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "REWORK"
    assert (
        update["worker_state"]["execution_feedback"]["instructions"]
        == "Rewrite only from the evidence already available"
    )


def test_explicit_next_resume_commits_current_result_before_advancing(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"action": "NEXT", "text": "accept current result", "docs": []},
    )
    state = graph_state(
        pending_user_action={
            "category": "EXTERNAL_BLOCKER",
            "task_id": "T2",
            "issues": [{"code": "RESOURCE_UNAVAILABLE"}],
        }
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "NEXT"
    assert [result["task_id"] for result in update["results"]] == ["T1", "T2"]


def test_resume_dict_text_alias_honors_only_an_accepted_action(monkeypatch):
    monkeypatch.setattr(
        recovery_module,
        "interrupt",
        lambda payload: {"text": "带限制继续", "docs": []},
    )
    state = graph_state(
        pending_user_action={
            "category": "EXTERNAL_BLOCKER",
            "task_id": "T2",
            "issues": [{"code": "RESOURCE_UNAVAILABLE"}],
            "accepted_choices": ["REWORK", "NEXT", "DONE"],
        }
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "NEXT"
    assert [result["task_id"] for result in update["results"]] == ["T1", "T2"]


def test_resume_plain_string_alias_routes_to_done(monkeypatch):
    monkeypatch.setattr(recovery_module, "interrupt", lambda payload: "结束")
    state = graph_state(
        pending_user_action={
            "category": "EXTERNAL_BLOCKER",
            "task_id": "T2",
            "issues": [{"code": "RESOURCE_UNAVAILABLE"}],
            "accepted_choices": ["REWORK", "NEXT", "DONE"],
        }
    )

    update = needs_user_input(state, {})

    assert update["workflow_action"] == "DONE"
    assert "results" not in update
    assert [result["task_id"] for result in state["results"]] == ["T1"]


def test_resume_aliases_are_deterministic_and_restricted_to_accepted_choices(
    monkeypatch,
):
    aliases = {
        "NEXT": ["NEXT", "CONTINUE", "继续", "带限制继续", "接受当前结果", "跳过"],
        "DONE": ["DONE", "完成", "结束"],
        "REWORK": ["REWORK", "RETRY", "返工", "重试"],
        "EVIDENCE_RECOVERY": ["EVIDENCE_RECOVERY", "证据恢复", "扩大检索"],
    }
    accepted_choices = list(aliases)

    for expected, values in aliases.items():
        for value in values:
            monkeypatch.setattr(recovery_module, "interrupt", lambda payload, value=value: value)
            update = needs_user_input(
                graph_state(
                    pending_user_action={
                        "category": "EXTERNAL_BLOCKER",
                        "task_id": "T2",
                        "accepted_choices": accepted_choices,
                    }
                ),
                {},
            )
            assert update["workflow_action"] == expected

    monkeypatch.setattr(recovery_module, "interrupt", lambda payload: "跳过")
    rejected = needs_user_input(
        graph_state(
            pending_user_action={
                "category": "EXTERNAL_BLOCKER",
                "task_id": "T2",
                "accepted_choices": ["REWORK"],
            }
        ),
        {},
    )
    assert rejected["workflow_action"] == "REWORK"
    assert "results" not in rejected


def test_worker_feedback_is_execution_only_and_web_is_explicitly_gated():
    task = _task(
        "T2",
        use_web=True,
        tool_requirements=["chemical_knowledge_base_tool", "spider_tool"],
        visualization={"allow_web_fallback": True},
    )
    original = deepcopy(task)
    worker_state = {
        "retained": True,
        "execution_feedback": {
            "mode": "evidence_recovery",
            "issues": [{"code": "EVIDENCE_GAP"}],
            "instructions": "Use synonyms and preserve citations",
            "recovery_plan": {
                "recovery_id": "T2:p1:t1:evidence_recovery:1",
                "task_id": "T2",
                "plan_revision": 1,
                "task_revision": 1,
                "recovery_sequence": 1,
                "evidence_queries": ["catalyst lifetime deactivation"],
                "asset_actions": [],
                "length_target": None,
                "scope_constraints": [],
            },
            "allow_web": False,
        },
    }

    execution_task, instructions, cleaned_state = AutonomousToolNode._prepare_execution_task(
        task,
        worker_state,
        {"plan_revision": 1, "task_revision": 1},
    )

    assert task == original
    assert execution_task["query"] == "query T2"
    assert execution_task["_recovery_queries"] == ["catalyst lifetime deactivation"]
    assert execution_task["use_web"] is False
    assert "spider_tool" not in execution_task["tool_requirements"]
    assert execution_task["visualization"]["allow_web_fallback"] is False
    assert "Use synonyms" in instructions
    assert "T2:p1:t1:evidence_recovery:1" in instructions
    assert "execution_feedback" not in cleaned_state
    assert cleaned_state["retained"] is True


def test_worker_discards_stale_recovery_plan_after_checkpoint_resume():
    task = _task("T2", use_web=True, tool_requirements=["spider_tool"])
    worker_state = {
        "retained": True,
        "execution_feedback": {
            "mode": "evidence_recovery",
            "instructions": "stale instructions",
            "allow_web": False,
            "recovery_plan": {
                "recovery_id": "T2:p1:t1:evidence_recovery:1",
                "task_id": "T2",
                "plan_revision": 1,
                "task_revision": 1,
                "recovery_sequence": 1,
                "evidence_queries": ["stale query"],
                "asset_actions": [],
                "length_target": None,
                "scope_constraints": [],
            },
        },
    }

    execution_task, instructions, cleaned_state = AutonomousToolNode._prepare_execution_task(
        task,
        worker_state,
        {"plan_revision": 2, "task_revision": 1},
    )

    assert execution_task == task
    assert instructions == ""
    assert "execution_feedback" not in cleaned_state


def test_worker_preserves_task_web_access_when_feedback_has_no_web_override():
    task = _task(
        "T2",
        use_web=True,
        tool_requirements=["spider_tool"],
        visualization={"allow_web_fallback": True},
    )
    worker_state = {
        "execution_feedback": {
            "mode": "rework",
            "issues": [{"code": "TOO_SHORT"}],
            "instructions": "Expand the analysis",
        }
    }

    execution_task, _, _ = AutonomousToolNode._prepare_execution_task(task, worker_state)

    assert execution_task["use_web"] is True
    assert "spider_tool" in execution_task["tool_requirements"]
    assert execution_task["visualization"]["allow_web_fallback"] is True


def test_explicit_web_denial_cannot_be_reenabled_from_task_description():
    initialized = []

    class SpiderTool:
        def __init__(self, config):
            initialized.append("spider")
            self.name = "spider_tool"

        def validate_task(self, task):
            return True

    class Config:
        ENABLED_TOOLS = ("SpiderTool",)

    task = _task(
        "T2",
        use_rag=False,
        use_web=True,
        task_description="使用最新公开信息补充分析",
    )
    worker_state = {
        "execution_feedback": {
            "mode": "evidence_recovery",
            "issues": [{"code": "EVIDENCE_GAP"}],
            "instructions": "Use local evidence only",
            "allow_web": False,
        }
    }
    execution_task, _, _ = AutonomousToolNode._prepare_execution_task(task, worker_state)
    manager = ToolManager.__new__(ToolManager)
    manager.config = Config()
    manager.tool_classes = {"SpiderTool": SpiderTool}

    tools = manager.get_available_tools_for_task(execution_task)

    assert tools == []
    assert initialized == []


def test_worker_router_preserves_feedback_until_execution_node_consumes_it():
    feedback = {
        "mode": "rework",
        "issues": [{"code": "TOO_SHORT"}],
        "instructions": "Expand the section",
    }
    state = graph_state(worker_state={"execution_feedback": feedback, "retained": True})

    update = router_node(state)

    assert update["worker_state"]["next_node"] == "autonomous_tool_node"
    assert update["worker_state"]["execution_feedback"] == feedback
    assert update["worker_state"]["retained"] is True


def _worker_node_for_process_test(*, fail=False):
    node = AutonomousToolNode.__new__(AutonomousToolNode)

    def create_tools(task):
        if fail:
            raise RuntimeError("tool setup failed")
        return []

    node.tool_manager = SimpleNamespace(create_langchain_tools=create_tools)
    node.llm_client = SimpleNamespace()
    node._build_system_prompt = lambda task, tools: "system"
    node._build_task_prompt = lambda task: "task"
    node._prefetch_rag = lambda task, tools: []
    node._execute_tool_loop = lambda *args, **kwargs: ("completed", [], {})
    node._prepare_concept_graph = lambda task, calls: (
        SimpleNamespace(records=[]),
        {},
        None,
    )
    node._create_task_result = lambda task, *args, **kwargs: {
        "task_id": task["task_id"],
        "status": "COMPLETED",
    }
    node._create_error_result = lambda task, cursor, error: {
        "task_id": task["task_id"],
        "status": "FAILED",
        "error": error,
    }
    return node


def _state_with_web_recovery_feedback():
    return graph_state(
        worker_state={
            "retained": True,
            "execution_feedback": {
                "mode": "evidence_recovery",
                "instructions": "Search the permitted web source",
                "allow_web": True,
            },
        },
        all_results=[],
        tool_execution_history=[],
    )


def test_worker_success_return_strips_execution_only_web_marker():
    update = _worker_node_for_process_test().process(
        _state_with_web_recovery_feedback()
    )

    assert update["current_task"]["task_id"] == "T2"
    assert update["current_task"]["use_web"] is True
    assert "_recovery_allow_web" not in update["current_task"]


def test_worker_error_return_strips_execution_only_web_marker():
    update = _worker_node_for_process_test(fail=True).process(
        _state_with_web_recovery_feedback()
    )

    assert update["current_task"]["task_id"] == "T2"
    assert update["current_task"]["use_web"] is True
    assert "_recovery_allow_web" not in update["current_task"]


def test_auto_graph_has_no_replan_route_to_planner():
    graph_path = Path(__file__).parents[1] / "src" / "graph.py"
    module = ast.parse(graph_path.read_text(encoding="utf-8"))
    workflow_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkFlowBase"
    )
    build_method = next(
        node
        for node in workflow_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build"
    )
    source = ast.get_source_segment(graph_path.read_text(encoding="utf-8"), build_method)

    assert '"REPLAN": "Planner"' not in source
    assert '"Verifier", "DecisionPolicy"' in source
    assert '"DecisionPolicy"' in source
    assert '"PlanPatcher"' in source
    assert '"NeedsUserInput"' in source
    assert '"RETRY_VERIFIER": "Verifier"' in source


def test_verifier_contract_failure_routes_only_back_to_verifier_once():
    state = graph_state(
        assessment={
            "status": "FAILED",
            "issues": [
                {
                    "code": "ASSESSMENT_CONTRACT_ERROR",
                    "category": "VERIFIER_FAILURE",
                }
            ],
        }
    )

    update = decision_policy(state, {})

    assert update["workflow_action"] == "RETRY_VERIFIER"
    assert route_policy({**state, **update}) == "RETRY_VERIFIER"
    assert "worker_state" not in update
    assert update["task_retry_count"] == {}


def test_automatic_planner_filters_legacy_replan_without_resetting_cursor(monkeypatch):
    planner_module = importlib.import_module("src.nodes.planner")
    legacy_replan = AIMessage(
        content=json.dumps({"to": "Planner", "type": "REPLAN"})
    )
    proceed = AIMessage(
        content=json.dumps({"to": "Planner", "type": "PROCEED"})
    )
    state = graph_state(
        cursor=1,
        decision="REPLAN",
        messages=[proceed, legacy_replan],
    )

    def forbidden_full_replan(*args, **kwargs):
        raise AssertionError("automatic workflow invoked full replan")

    monkeypatch.setattr(
        planner_module,
        "_build_tasks_from_replan_feedback",
        forbidden_full_replan,
    )

    update = automatic_planner(state, {})

    assert update["planner_action"] == "PROCEED"
    assert update["cursor"] == 2
    assert update["cursor"] != 0
    assert state["decision"] == "REPLAN"
    assert state["messages"] == [proceed, legacy_replan]


def test_auto_graph_uses_guarded_planner_but_manual_keeps_legacy_planner():
    graph_path = Path(__file__).parents[1] / "src" / "graph.py"
    source = graph_path.read_text(encoding="utf-8")

    assert "automatic_planner if self.use_auto_verifier else planner" in source


def test_graph_schema_persists_recovery_handoff_fields():
    graph_path = Path(__file__).parents[1] / "src" / "graph.py"
    graph_source = graph_path.read_text(encoding="utf-8")
    state_path = Path(__file__).parents[1] / "src" / "state.py"
    state_source = state_path.read_text(encoding="utf-8")

    assert "state_schema=State" in graph_source
    assert "WorkflowState" not in graph_source
    assert "assessment:" in state_source
    assert "continuation_action:" in state_source
    assert "verification_warning:" in state_source

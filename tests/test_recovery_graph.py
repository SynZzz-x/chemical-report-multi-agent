import ast
import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

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
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["catalyst life evidence", "deactivation mechanism"],
    }


def test_t2_evidence_gap_never_routes_to_planner_or_resets_cursor():
    state = graph_state(cursor=1, accepted_ids=("T1",))
    update = decision_policy({**state, "assessment": evidence_gap_assessment()}, {})

    assert update["workflow_action"] == "EVIDENCE_RECOVERY"
    assert update.get("cursor", state["cursor"]) == 1
    assert [item["task_id"] for item in state["results"]] == ["T1"]
    assert route_policy({**state, **update}, {}) == "EVIDENCE_RECOVERY"


def test_rework_policy_creates_structured_execution_feedback():
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
    assert feedback["mode"] == "rework"
    assert feedback["issues"] == assessment["issues"]
    assert "Add mechanism detail" in feedback["instructions"]
    assert update["worker_state"]["retained"] is True


def test_accept_with_warning_routes_through_continuation_action():
    state = graph_state(
        task_retry_count={"T2": 2},
        assessment={
            "status": "FAILED",
            "issues": [{"code": "TOO_SHORT", "category": "CONTENT_DEFECT"}],
        },
    )

    update = decision_policy(state, {})

    assert update["workflow_action"] == "ACCEPT_WITH_WARNING"
    assert route_policy({**state, **update}, {}) == "NEXT"


def test_evidence_recovery_builds_query_and_honors_task_web_gate():
    state = graph_state(assessment=evidence_gap_assessment())
    original_tasks = deepcopy(state["tasks"])

    update = evidence_recovery(state, {})

    feedback = update["worker_state"]["execution_feedback"]
    assert feedback["mode"] == "evidence_recovery"
    assert "catalyst life evidence" in feedback["recovery_query"]
    assert feedback["allow_web"] is False
    assert state["tasks"] == original_tasks

    web_state = graph_state(assessment=evidence_gap_assessment())
    web_state["tasks"][1]["visualization"] = {"allow_web_fallback": True}
    web_update = evidence_recovery(web_state, {})
    assert web_update["worker_state"]["execution_feedback"]["allow_web"] is True


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
    assert "task_name, task_description, query, use_rag, use_web" in prompt


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
    assert captured["accepted_choices"]
    assert update["workflow_action"] == "REWORK"
    assert update["docs"] == [resumed_doc]
    assert "cursor" not in update
    assert route_after_blocker({**state, **update}, {}) == "REWORK"
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
    assert [result["task_id"] for result in update["results"]] == ["T1", "T2"]


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
            "recovery_query": "catalyst lifetime deactivation",
            "allow_web": False,
        },
    }

    execution_task, instructions, cleaned_state = AutonomousToolNode._prepare_execution_task(
        task, worker_state
    )

    assert task == original
    assert execution_task["query"] == "catalyst lifetime deactivation"
    assert execution_task["use_web"] is False
    assert "spider_tool" not in execution_task["tool_requirements"]
    assert execution_task["visualization"]["allow_web_fallback"] is False
    assert "Use synonyms" in instructions
    assert "execution_feedback" not in cleaned_state
    assert cleaned_state["retained"] is True


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
    source = graph_path.read_text(encoding="utf-8")

    assert "class WorkflowState(State, total=False):" in source
    assert "assessment:" in source
    assert "continuation_action:" in source
    assert "state_schema=WorkflowState" in source

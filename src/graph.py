"""LangGraph状态及路由定义"""
from typing import Any

from langgraph.graph import START, END, StateGraph
from .state import State, ConfigSchema
from .nodes.intake import intake
from .nodes.planner import planner, planner_confirm
from .nodes.quality_review import quality_review
from .nodes.verifier_manual import verifier_manual, decision
from .nodes.recovery import (
    automatic_planner,
    decision_policy,
    evidence_recovery,
    needs_user_input,
    plan_patcher,
    route_after_blocker,
    route_policy,
)
from .nodes.summarizer_v2 import summarizer
from .nodes.exiting import exiting
from .nodes.task_controller import route_task_controller, task_controller
from .nodes.artifact_commit import artifact_commit
from .nodes.legacy_verifier import legacy_auto_verifier, legacy_manual_verifier
# from .nodes.worker.agent.graph import router_node, execute_task_node, generate_result_node, route_decision
from .nodes.worker.agent.graph import create_worker_workflow


class WorkflowState(State, total=False):
    """Graph-local Task 3 channels not yet shared with legacy node schemas."""

    assessment: dict[str, Any]
    continuation_action: str
    verification_warning: dict[str, Any]


def route_planner(state: State):
    action = state.get("planner_action")
    if action == "PROCEED":
        return "TaskController"
    return "Planner_Confirm"


def route_planner_confirm(state: State):
    if state.get("planner_action") == "FULL_REPLAN_RETRY":
        return "Planner"
    if state.get("planner_action") in {
        "INTAKE_SUMMARY_REFINED",
        "FULL_REPLAN_REFINED",
        "FULL_REPLAN_ERROR",
    }:
        return "Planner_Confirm"
    return "TaskController"


_MANUAL_VERIFIER_ROUTES = {
    "REWORK": "TaskController",
    "RETRY_WORKER": "TaskController",
    "FULL_REPLAN": "Planner",
    "NEXT": "TaskController",
    "DONE": "TaskController",
}


class WorkFlowBase(StateGraph):
    """
    工作流基类
    """
    def __init__(self, use_auto_verifier=False):
        super().__init__(state_schema=WorkflowState, config_schema=ConfigSchema)
        self.use_auto_verifier = use_auto_verifier
        self._build()

    def _build(self):
        """
        构建工作流图
        """
        self.add_edge(START, "Intake")

        self.add_node("Intake", intake)
        self.add_edge("Intake", "Planner")

        self.add_node(
            "Planner",
            automatic_planner if self.use_auto_verifier else planner,
        )
        self.add_conditional_edges(
            "Planner",
            route_planner
        )

        self.add_node("Planner_Confirm", planner_confirm)
        self.add_conditional_edges(
            "Planner_Confirm",
            route_planner_confirm,
            {
                "Planner": "Planner",
                "Planner_Confirm": "Planner_Confirm",
                "TaskController": "TaskController",
            },
        )

        self.add_node("TaskController", task_controller)
        self.add_conditional_edges(
            "TaskController",
            route_task_controller,
            {
                "Worker": "Worker",
                "Summarizer": "Summarizer",
                "NeedsUserInput": "NeedsUserInput",
            },
        )
        
        # Worker子图
        worker = create_worker_workflow()
        self.add_node("Worker", worker)
        self.add_node("ArtifactCommit", artifact_commit)
        self.add_edge("Worker", "ArtifactCommit")
        self.add_edge("ArtifactCommit", "QualityReview")

        if self.use_auto_verifier:
            self.add_node("QualityReview", quality_review, metadata={"type":"auto"})
            self.add_node(
                "Verifier",
                legacy_auto_verifier,
                metadata={"type": "legacy_auto_checkpoint"},
            )
            self.add_node("DecisionPolicy", decision_policy)
            self.add_edge("QualityReview", "DecisionPolicy")
            self.add_edge("Verifier", "DecisionPolicy")
            self.add_conditional_edges(
                "DecisionPolicy",
                route_policy,
                {
                    "NEXT": "TaskController",
                    "DONE": "TaskController",
                    "REWORK": "TaskController",
                    "EVIDENCE_RECOVERY": "EvidenceRecovery",
                    "PLAN_PATCH": "PlanPatcher",
                    "NEEDS_USER_INPUT": "NeedsUserInput",
                    "RETRY_VERIFIER": "QualityReview",
                },
            )

        else:
            self.add_node("QualityReview", verifier_manual, metadata={"type":"manual"})
            self.add_node(
                "Verifier",
                legacy_manual_verifier,
                metadata={"type": "legacy_manual_checkpoint"},
            )
            self.add_conditional_edges(
                "QualityReview",
                decision,
                _MANUAL_VERIFIER_ROUTES,
            )
            self.add_conditional_edges(
                "Verifier",
                decision,
                _MANUAL_VERIFIER_ROUTES,
            )

        # Recovery/HumanReview nodes are shared by automatic and manual modes.
        self.add_node("EvidenceRecovery", evidence_recovery)
        self.add_edge("EvidenceRecovery", "TaskController")

        self.add_node("PlanPatcher", plan_patcher)
        self.add_conditional_edges(
            "PlanPatcher",
            route_after_blocker,
            {
                "REWORK": "TaskController",
                "NEEDS_USER_INPUT": "NeedsUserInput",
            },
        )

        self.add_node("NeedsUserInput", needs_user_input)
        self.add_conditional_edges(
            "NeedsUserInput",
            route_after_blocker,
            {
                "REWORK": "TaskController",
                "EVIDENCE_RECOVERY": "EvidenceRecovery",
                "NEXT": "TaskController",
                "DONE": "TaskController",
            },
        )

        self.add_node("Summarizer", summarizer)
        self.add_edge("Summarizer", "Exit")

        self.add_node("Exit", exiting)
        self.add_edge("Exit", END)


class WorkFlow(WorkFlowBase):
    """
    默认手动审核工作流
    """
    def __init__(self):
        super().__init__(use_auto_verifier=False)


class WorkFlowAuto(WorkFlowBase):
    """
    自动审核工作流
    """
    def __init__(self):
        super().__init__(use_auto_verifier=True)

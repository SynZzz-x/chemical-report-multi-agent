"""LangGraph状态及路由定义"""

from langgraph.graph import START, END, StateGraph
from .state import State, ConfigSchema
from .nodes.intake import intake
from .nodes.planner import planner, planner_confirm
from .nodes.verifier import verifier as verifier_auto
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
from .nodes.asset_recovery import asset_recovery, route_after_asset_recovery
from .nodes.summarizer_v2 import summarizer
from .nodes.synthesis import synthesis
from .nodes.exiting import exiting
# from .nodes.worker.agent.graph import router_node, execute_task_node, generate_result_node, route_decision
from .nodes.worker.agent.graph import create_worker_workflow


def _execution_node(state: State) -> str:
    tasks = state.get("tasks") or []
    cursor = int(state.get("cursor", 0) or 0)
    if 0 <= cursor < len(tasks):
        task = tasks[cursor]
        if isinstance(task, dict) and task.get("task_type") == "synthesis":
            return "Synthesis"
    return "Worker"


def route_planner(state: State):
    action = state.get("planner_action")
    if action == "PROCEED":
        return _execution_node(state)
    return "Planner_Confirm"


def route_planner_confirm(state: State):
    if state.get("planner_action") in {
        "FULL_REPLAN_RETRY",
        "INITIAL_PLAN_RETRY",
    }:
        return "Planner"
    if state.get("planner_action") == "INITIAL_PLAN_CANCELLED":
        return "Exit"
    if state.get("planner_action") in {
        "INTAKE_SUMMARY_REFINED",
        "FULL_REPLAN_REFINED",
        "FULL_REPLAN_ERROR",
    }:
        return "Planner_Confirm"
    return _execution_node(state)


def route_workflow_policy(state: State) -> str:
    route = route_policy(state)
    if route == "SYNTHESIS_REWRITE":
        return "SYNTHESIS_REWORK"
    if (
        route in {"REWORK", "EVIDENCE_RECOVERY"}
        and _execution_node(state) == "Synthesis"
    ):
        return "SYNTHESIS_REWORK"
    return route


def route_after_execution_blocker(state: State) -> str:
    route = route_after_blocker(state)
    if route == "SYNTHESIS_REWRITE":
        return "SYNTHESIS_REWORK"
    if (
        route in {"REWORK", "EVIDENCE_RECOVERY"}
        and _execution_node(state) == "Synthesis"
    ):
        return "SYNTHESIS_REWORK"
    return route


def route_manual_verifier(state: State) -> str:
    route = decision(state)
    if route == "RETRY_WORKER" and _execution_node(state) == "Synthesis":
        return "RETRY_SYNTHESIS"
    return route


_MANUAL_VERIFIER_ROUTES = {
    "RETRY_WORKER": "Worker",
    "RETRY_SYNTHESIS": "Synthesis",
    "FULL_REPLAN": "Planner",
    "NEXT": "Planner",
    "DONE": "Summarizer",
}


class WorkFlowBase(StateGraph):
    """
    工作流基类
    """
    def __init__(self, use_auto_verifier=False):
        super().__init__(state_schema=State, config_schema=ConfigSchema)
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
                "Worker": "Worker",
                "Synthesis": "Synthesis",
                "Exit": "Exit",
            },
        )
        
        # Worker子图
        worker = create_worker_workflow()
        self.add_node("Worker", worker)
        self.add_edge("Worker", "Verifier")
        self.add_node("Synthesis", synthesis)
        self.add_edge("Synthesis", "Verifier")

        if self.use_auto_verifier:
            self.add_node("Verifier", verifier_auto, metadata={"type":"auto"})
            self.add_node("DecisionPolicy", decision_policy)
            self.add_edge("Verifier", "DecisionPolicy")
            self.add_conditional_edges(
                "DecisionPolicy",
                route_workflow_policy,
                {
                    "NEXT": "Planner",
                    "DONE": "Summarizer",
                    "REWORK": "Worker",
                    "ASSET_RECOVERY": "AssetRecovery",
                    "LENGTH_REWRITE": "Worker",
                    "SYNTHESIS_REWORK": "Synthesis",
                    "EVIDENCE_RECOVERY": "EvidenceRecovery",
                    "PLAN_PATCH": "PlanPatcher",
                    "NEEDS_USER_INPUT": "NeedsUserInput",
                    "RETRY_VERIFIER": "Verifier",
                    "FATAL_SYSTEM": "Exit",
                },
            )

            self.add_node("EvidenceRecovery", evidence_recovery)
            self.add_edge("EvidenceRecovery", "Worker")

            self.add_node("AssetRecovery", asset_recovery)
            self.add_conditional_edges(
                "AssetRecovery",
                route_after_asset_recovery,
                {
                    "RETRY_VERIFIER": "Verifier",
                    "REWORK": "Worker",
                },
            )

            self.add_node("PlanPatcher", plan_patcher)
            self.add_conditional_edges(
                "PlanPatcher",
                route_after_execution_blocker,
                {
                    "REWORK": "Worker",
                    "LENGTH_REWRITE": "Worker",
                    "SYNTHESIS_REWORK": "Synthesis",
                    "NEEDS_USER_INPUT": "NeedsUserInput",
                },
            )

            self.add_node("NeedsUserInput", needs_user_input)
            self.add_conditional_edges(
                "NeedsUserInput",
                route_after_execution_blocker,
                {
                    "REWORK": "Worker",
                    "ASSET_RECOVERY": "AssetRecovery",
                    "LENGTH_REWRITE": "Worker",
                    "SYNTHESIS_REWORK": "Synthesis",
                    "EVIDENCE_RECOVERY": "EvidenceRecovery",
                    "RETRY_VERIFIER": "Verifier",
                    "NEXT": "Planner",
                    "DONE": "Summarizer",
                },
            )
        else:
            self.add_node("Verifier", verifier_manual, metadata={"type":"manual"})
            self.add_conditional_edges(
                "Verifier",
                route_manual_verifier,
                _MANUAL_VERIFIER_ROUTES,
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

"""LangGraph状态及路由定义"""
from langgraph.graph import START, END, StateGraph
from .state import State, ConfigSchema
from .nodes.intake import intake
from .nodes.planner import planner, planner_confirm
from .nodes.verifier import verifier as verifier_auto
from .nodes.verifier_manual import verifier_manual, decision
from .nodes.summarizer_v2 import summarizer
from .nodes.exiting import exiting
# from .nodes.worker.agent.graph import router_node, execute_task_node, generate_result_node, route_decision
from .nodes.worker.agent.graph import create_worker_workflow


def route_planner(state: State):
    action = state.get("planner_action")
    if action == "PROCEED":
        return "Worker"
    return "Planner_Confirm"


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

        self.add_node("Planner", planner)
        self.add_conditional_edges(
            "Planner",
            route_planner
        )

        self.add_node("Planner_Confirm", planner_confirm)
        self.add_edge("Planner_Confirm", "Worker")
        
        # Worker子图
        worker = create_worker_workflow()
        self.add_node("Worker", worker)
        self.add_edge("Worker", "Verifier")

        if self.use_auto_verifier:
            self.add_node("Verifier", verifier_auto, metadata={"type":"auto"})
        else:
            self.add_node("Verifier", verifier_manual, metadata={"type":"manual"})
            
        self.add_conditional_edges(
            "Verifier",
            decision,
            {
                "RETRY_WORKER": "Worker",
                "REPLAN": "Planner",
                "NEXT": "Planner",
                "DONE": "Summarizer",
            }
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

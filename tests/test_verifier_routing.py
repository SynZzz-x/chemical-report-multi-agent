from src.nodes.verifier_manual import decision


def test_decision_returns_state_decision():
    assert decision({"decision": "NEXT"}, {}) == "NEXT"
    assert decision({"decision": "REPLAN"}, {}) == "REPLAN"
    assert decision({"decision": "RETRY_WORKER"}, {}) == "RETRY_WORKER"
    assert decision({"decision": "DONE"}, {}) == "DONE"


def test_decision_defaults_to_done():
    assert decision({}, {}) == "DONE"

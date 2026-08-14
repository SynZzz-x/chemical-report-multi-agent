import json
from types import SimpleNamespace

import pytest

from src.nodes import verifier as auto_verifier_module
from src.recovery.policy import decide_recovery_action


def _state(*, cursor=0):
    tasks = [
        {
            "task_id": "T1",
            "task_name": "引言",
            "task_description": "撰写简短引言。",
            "generate_figure": False,
            "generate_table": False,
        },
        {
            "task_id": "T2",
            "task_name": "质量指标体系",
            "task_description": "必须生成质量指标表格。",
            "generate_figure": False,
            "generate_table": True,
        },
    ]
    current = tasks[cursor]
    return {
        "tasks": tasks,
        "cursor": cursor,
        "current_result": {
            "task_id": current["task_id"],
            "status": "COMPLETED",
            "text_output": f"{current['task_name']}正文",
            "tables": [],
            "figures": [],
            "citations": [{"evidence_id": "E1"}],
            "sources_used": ["process.docx"],
        },
        "results": [{"task_id": "T0", "text_output": "accepted"}],
        "messages": [],
        "task_retry_count": {"T1": 1},
        "replan_count": 1,
        "plan_revision": 4,
    }


def _run(monkeypatch, state, assessment):
    captured = {}

    class Model:
        def invoke(self, payload):
            captured.update(payload)
            return SimpleNamespace(content=json.dumps(assessment, ensure_ascii=False))

    monkeypatch.setattr(auto_verifier_module, "get_llm", lambda *args, **kwargs: Model())
    update = auto_verifier_module.verifier(
        state,
        {"configurable": {"use_llm": True}},
    )
    return update, captured


def test_verifier_is_assessment_only_and_classifies_evidence_gap(monkeypatch):
    state = _state()
    state["current_result"]["text_output"] += "[E1]"
    assessment = {
        "status": "BLOCKED",
        "current_section": "引言",
        "issues": [
            {
                "code": "MISSING_EVIDENCE",
                "category": "EVIDENCE_GAP",
                "description": "关键结论缺少知识库依据",
                "suggestion": "扩大检索覆盖并补充引用",
                "severity": "major",
                "retrieval_query": "聚乙烯 关键结论 知识库依据",
            }
        ],
        "requirements_met": ["包含背景"],
        "requirements_missing": ["关键结论来源"],
        "recommended_decision": "REPLAN",
    }

    update, _ = _run(monkeypatch, state, assessment)

    assert set(update) == {"assessment"}
    assert "decision" not in update
    assert "recommended_decision" not in update["assessment"]
    assert update["assessment"]["issues"][0]["category"] == "EVIDENCE_GAP"
    assert (
        update["assessment"]["issues"][0]["retrieval_query"]
        == "聚乙烯 关键结论 知识库依据"
    )
    assert update["assessment"]["requirements_missing"] == ["关键结论来源"]


def test_sanitizer_checks_current_task_requirements_not_first_task():
    state = _state(cursor=1)
    assessment = {
        "status": "FAILED",
        "current_section": "质量指标体系",
        "issues": [
            {
                "code": "MISSING_TABLE",
                "category": "CONTENT_DEFECT",
                "description": "缺少任务要求的表格",
                "suggestion": "补充表格",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["质量指标表格"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)

    assert sanitized["status"] == "FAILED"
    assert sanitized["issues"][0]["code"] == "MISSING_TABLE"


def test_verifier_receives_full_task_and_asset_context(monkeypatch):
    assessment = {
        "status": "PASS",
        "current_section": "质量指标体系",
        "issues": [],
        "requirements_met": ["质量指标表格"],
        "requirements_missing": [],
    }

    _, captured = _run(monkeypatch, _state(cursor=1), assessment)

    assert "必须生成质量指标表格" in captured["task_requirements"]
    assets = json.loads(captured["worker_assets"])
    assert assets["citations"] == [{"evidence_id": "E1"}]
    assert assets["tables"] == []
    assert "actual_length" in assets


def test_verifier_receives_effective_source_policy(monkeypatch):
    state = _state()
    state["tasks"][0].update({"use_rag": True, "use_web": False})
    state["web_authorized"] = False

    _, captured = _run(
        monkeypatch,
        state,
        {
            "status": "PASS",
            "current_section": "引言",
            "issues": [],
            "requirements_met": ["内容完整"],
            "requirements_missing": [],
        },
    )

    assert json.loads(captured["source_policy"]) == {
        "rag_allowed": True,
        "web_authorized": False,
        "web_allowed": False,
    }


def test_sanitizer_rewrites_unauthorized_web_demand_as_source_gap():
    state = _state()
    state["tasks"][0].update({"use_rag": True, "use_web": False})
    state["web_authorized"] = False
    assessment = {
        "status": "FAILED",
        "current_section": "引言",
        "issues": [
            {
                "code": "EVIDENCE_GAP",
                "category": "EVIDENCE_GAP",
                "description": "未从其他权威标准文献补充鱼眼检测方法。",
                "suggestion": "应查询公开网络中的权威标准。",
                "severity": "major",
                "retrieval_query": "聚乙烯 鱼眼 检测方法",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["鱼眼检测方法"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)

    issue = sanitized["issues"][0]
    assert issue["code"] == "EVIDENCE_GAP"
    assert issue["description"] == (
        "当前已授权来源不足以支持该证据要求：聚乙烯 鱼眼 检测方法"
    )
    assert "授权公开网络检索" in issue["suggestion"]
    assert "应查询" not in issue["suggestion"]


def test_deterministic_length_failure_overrides_llm_pass(monkeypatch):
    state = _state()
    state["tasks"][0]["task_description"] = "撰写引言，字数：20-30字。"
    state["current_result"].update(
        {
            "text_output": "太短。",
            "citations": [],
            "word_count": 999,
        }
    )
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [],
        "requirements_met": ["内容完整"],
        "requirements_missing": [],
    }

    update, captured = _run(monkeypatch, state, assessment)

    issue = next(
        item for item in update["assessment"]["issues"] if item["code"] == "TOO_SHORT"
    )
    assert update["assessment"]["status"] == "FAILED"
    assert issue["actual"] == 2
    assert issue["required_min"] == 20
    assets = json.loads(captured["worker_assets"])
    assert assets["actual_length"] == 2
    assert assets["length_target"] == {"min": 20, "max": 30}


def test_deterministic_length_replaces_llm_estimate_for_the_same_issue(monkeypatch):
    state = _state()
    state["tasks"][0]["task_description"] = "字数：20-30字。"
    state["current_result"].update({"text_output": "太短。", "citations": []})
    assessment = {
        "status": "FAILED",
        "current_section": "引言",
        "issues": [
            {
                "code": "TOO_SHORT",
                "category": "CONTENT_DEFECT",
                "description": "估计只有约5字。",
                "suggestion": "扩写。",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["正文篇幅"],
    }

    update, _ = _run(monkeypatch, state, assessment)

    issues = [
        issue for issue in update["assessment"]["issues"] if issue["code"] == "TOO_SHORT"
    ]
    assert len(issues) == 1
    assert issues[0]["actual"] == 2
    assert issues[0]["required_min"] == 20


def test_deterministic_length_removes_generic_duplicate_issue(monkeypatch):
    state = _state()
    state["tasks"][0]["task_description"] = "字数：20-30字。"
    state["current_result"].update({"text_output": "太短。", "citations": []})
    assessment = {
        "status": "FAILED",
        "current_section": "引言",
        "issues": [
            {
                "code": "CONTENT_DEFECT",
                "category": "CONTENT_DEFECT",
                "description": "正文篇幅不足，未达到20字的最低要求。",
                "suggestion": "扩写到20-30字。",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["正文篇幅"],
    }

    update, _ = _run(monkeypatch, state, assessment)

    assert [issue["code"] for issue in update["assessment"]["issues"]] == [
        "TOO_SHORT"
    ]


def test_invalid_retrieval_query_is_dropped_without_reusing_description():
    state = _state()
    assessment = {
        "status": "FAILED",
        "current_section": "引言",
        "issues": [
            {
                "code": "EVIDENCE_GAP",
                "category": "EVIDENCE_GAP",
                "description": "任务要求某项证据但正文没有完成。",
                "suggestion": "补充证据。",
                "severity": "major",
                "retrieval_query": [],
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["证据"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)

    assert sanitized["issues"][0]["code"] == "EVIDENCE_GAP"
    assert "retrieval_query" not in sanitized["issues"][0]


def test_verifier_audit_keeps_raw_issues_separate_from_sanitized_policy_issues(
    monkeypatch, tmp_path
):
    log_path = tmp_path / "verifier.jsonl"
    monkeypatch.setattr(auto_verifier_module, "LOG_PATH", str(log_path))
    state = _state()
    state["current_result"].update({"text_output": "正文。[E1]"})
    raw_issue = {
        "code": "MISSING_EVIDENCE",
        "category": "EVIDENCE_GAP",
        "description": "缺少直接证据",
        "suggestion": "补充证据",
        "severity": "major",
        "retrieval_query": "聚乙烯 反应压力 质量影响",
    }

    update, _ = _run(
        monkeypatch,
        state,
        {
            "status": "FAILED",
            "current_section": "引言",
            "issues": [raw_issue],
            "requirements_met": [],
            "requirements_missing": ["直接证据"],
        },
    )

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["raw_issues"] == [raw_issue]
    assert entry["assessment"] == update["assessment"]


def test_deterministic_asset_gate_requires_formal_assets_after_materialization():
    state = _state(cursor=1)
    state["current_result"].update({"citations": [], "tables": [], "figures": []})
    assessment = {
        "status": "PASS",
        "current_section": "质量指标体系",
        "issues": [],
        "requirements_met": ["正文完整"],
        "requirements_missing": [],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)
    checked = auto_verifier_module._apply_deterministic_validation(sanitized, state)

    assert checked["status"] == "FAILED"
    assert any(issue["code"] == "MISSING_TABLE" for issue in checked["issues"])


def test_causal_figure_missing_from_insufficient_evidence_is_not_a_second_root_issue():
    state = _state()
    state["tasks"][0].update(
        {
            "generate_figure": True,
            "visualization": {
                "kind": "causal",
                "required_concepts": ["反应压力", "熔融指数"],
            },
        }
    )
    state["current_result"].update(
        {
            "citations": [],
            "figures": [],
            "evidence_coverage": {
                "status": "insufficient",
                "uncovered_concepts": ["反应压力"],
            },
        }
    )

    checked = auto_verifier_module._apply_deterministic_validation(
        {
            "status": "PASS",
            "current_section": "引言",
            "issues": [],
            "requirements_met": [],
            "requirements_missing": [],
        },
        state,
    )

    assert not any(issue["code"] == "MISSING_FIGURE" for issue in checked["issues"])


def test_deterministic_length_check_emits_too_long():
    state = _state()
    state["tasks"][0]["task_description"] = "不超过3字。"
    state["current_result"].update({"text_output": "聚乙烯生产工艺", "citations": []})

    checked = auto_verifier_module._apply_deterministic_validation(
        {
            "status": "PASS",
            "current_section": "引言",
            "issues": [],
            "requirements_met": [],
            "requirements_missing": [],
        },
        state,
    )

    issue = next(issue for issue in checked["issues"] if issue["code"] == "TOO_LONG")
    assert issue["actual"] == 7
    assert issue["required_max"] == 3


def test_verifier_prints_short_issue_summary(monkeypatch, capsys):
    state = _state()
    state["current_result"]["text_output"] += "[E1]"
    assessment = {
        "status": "FAILED",
        "current_section": "引言",
        "issues": [
            {
                "code": "EVIDENCE_GAP",
                "category": "EVIDENCE_GAP",
                "description": "关键结论缺少可追溯来源",
                "suggestion": "补充引用",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["关键结论来源"],
    }

    _run(monkeypatch, state, assessment)

    output = capsys.readouterr().out
    assert "issue_count=1" in output
    assert "EVIDENCE_GAP: 关键结论缺少可追溯来源" in output


def test_verifier_rejects_unknown_inline_evidence_id(monkeypatch):
    state = _state()
    state["current_result"]["text_output"] = "温度影响熔融指数。[E404]"
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [],
        "requirements_met": ["内容完整"],
        "requirements_missing": [],
    }

    update, _ = _run(monkeypatch, state, assessment)

    assert update["assessment"]["status"] == "FAILED"
    assert update["assessment"]["issues"][0]["code"] == "INVALID_CITATION_ID"
    assert "E404" in update["assessment"]["issues"][0]["description"]


def test_verifier_requires_inline_binding_when_citations_are_available(monkeypatch):
    state = _state()
    state["current_result"]["text_output"] = "温度影响熔融指数。"
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [],
        "requirements_met": ["内容完整"],
        "requirements_missing": [],
    }

    update, _ = _run(monkeypatch, state, assessment)

    assert update["assessment"]["status"] == "FAILED"
    assert update["assessment"]["issues"][0]["code"] == "MISSING_INLINE_CITATION"


def test_verifier_rejects_inline_id_when_structured_citations_are_empty(monkeypatch):
    state = _state()
    state["current_result"]["citations"] = []
    state["current_result"]["text_output"] = "温度影响熔融指数。[E404]"
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [],
        "requirements_met": ["内容完整"],
        "requirements_missing": [],
    }

    update, _ = _run(monkeypatch, state, assessment)

    assert update["assessment"]["status"] == "FAILED"
    assert update["assessment"]["issues"][0]["code"] == "INVALID_CITATION_ID"


def test_sanitizer_rejects_pass_assessment_that_still_contains_issues():
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [
            {
                "code": "TOO_SHORT",
                "category": "CONTENT_DEFECT",
                "description": "内容过短",
                "suggestion": "扩写正文",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["正文深度"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, _state())

    assert sanitized["status"] == "FAILED"


def test_sanitizer_downgrades_pass_with_missing_requirements_to_structured_failure():
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [],
        "requirements_met": ["包含背景"],
        "requirements_missing": ["关键结论来源", "正文深度"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, _state())
    decision = decide_recovery_action(_state(), sanitized)

    assert sanitized["status"] == "FAILED"
    assert [issue["code"] for issue in sanitized["issues"]] == [
        "REQUIREMENT_MISSING",
        "REQUIREMENT_MISSING",
    ]
    assert decision["workflow_action"] != "DONE"


def test_sanitizer_keeps_malformed_failed_assessment_from_becoming_pass():
    assessment = {
        "status": "BLOCKED",
        "current_section": "引言",
        "issues": [None, 42],
        "requirements_met": [],
        "requirements_missing": ["外部依赖"],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, _state())

    assert sanitized["status"] == "FAILED"
    assert sanitized["issues"][0]["code"] == "ASSESSMENT_CONTRACT_ERROR"


def test_synthesis_verifier_ignores_markdown_heading_requirement():
    state = _state()
    state["tasks"][0].update(
        {
            "task_type": "synthesis",
            "task_name": "结论",
            "task_description": "总结前文，不输出标题。",
        }
    )
    assessment = {
        "status": "FAILED",
        "current_section": "结论",
        "issues": [
            {
                "code": "CONTENT_DEFECT",
                "category": "CONTENT_DEFECT",
                "description": "缺少任务要求的Markdown章节标题‘## 结论’。",
                "suggestion": "补充Markdown标题。",
                "severity": "major",
            }
        ],
        "requirements_met": [],
        "requirements_missing": [],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)

    assert sanitized["status"] == "PASS"
    assert sanitized["issues"] == []


@pytest.mark.parametrize("code", ["LLM_ERROR", "LLM_NOT_ENABLED"])
def test_verifier_service_failures_retry_verifier_once_then_require_user_input(
    monkeypatch, code
):
    state = _state()
    original_results = list(state["results"])
    if code == "LLM_ERROR":
        class FailingModel:
            def invoke(self, payload):
                raise RuntimeError("verification unavailable")

        monkeypatch.setattr(
            auto_verifier_module,
            "get_llm",
            lambda *args, **kwargs: FailingModel(),
        )
        config = {"configurable": {"use_llm": True}}
    else:
        monkeypatch.setattr(
            auto_verifier_module,
            "get_app_config",
            lambda: SimpleNamespace(deepseek_api_key=None),
        )
        config = {"configurable": {"use_llm": False}}

    sanitized = auto_verifier_module.verifier(state, config)["assessment"]
    first = decide_recovery_action(state, sanitized)
    second = decide_recovery_action({**state, **first}, sanitized)

    assert first["workflow_action"] == "RETRY_VERIFIER"
    assert first["verifier_retry_count"] == {"T1": 1}
    assert first["task_retry_count"] == {"T1": 1}
    assert second["workflow_action"] == "NEEDS_USER_INPUT"
    assert second["pending_user_action"]["category"] == "VERIFIER_FAILURE"
    assert second["task_retry_count"] == {"T1": 1}
    assert "results" not in second
    assert state["results"] == original_results


def test_assessment_contract_error_never_consumes_worker_content_retries():
    state = _state()
    malformed = {
        "status": "FAILED",
        "issues": [],
        "requirements_missing": [],
    }
    sanitized = auto_verifier_module._sanitize_assessment(malformed, state)

    first = decide_recovery_action(state, sanitized)
    second = decide_recovery_action({**state, **first}, sanitized)

    assert sanitized["issues"][0]["category"] == "VERIFIER_FAILURE"
    assert first["workflow_action"] == "RETRY_VERIFIER"
    assert first["task_retry_count"] == state["task_retry_count"]
    assert second["workflow_action"] == "NEEDS_USER_INPUT"
    assert second["task_retry_count"] == state["task_retry_count"]


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("requirements_missing", "citation"),
        ("issues", {"code": "LLM_ERROR"}),
        ("requirements_met", {"citation": True}),
    ],
)
def test_malformed_assessment_collections_use_bounded_verifier_retry_only(
    field, malformed_value
):
    state = _state()
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [],
        "requirements_met": [],
        "requirements_missing": [],
        field: malformed_value,
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)
    first = decide_recovery_action(state, sanitized)
    second = decide_recovery_action({**state, **first}, sanitized)

    assert sanitized["status"] == "FAILED"
    assert sanitized["issues"] == [
        {
            "code": "ASSESSMENT_CONTRACT_ERROR",
            "category": "VERIFIER_FAILURE",
            "description": "Verifier returned malformed collection fields.",
            "suggestion": "Retry automatic verification with a valid structured assessment.",
            "severity": "error",
        }
    ]
    assert first["workflow_action"] == "RETRY_VERIFIER"
    assert first["task_retry_count"] == state["task_retry_count"]
    assert second["workflow_action"] == "NEEDS_USER_INPUT"
    assert second["task_retry_count"] == state["task_retry_count"]


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("issues", [{}]),
        ("requirements_missing", [{}]),
        ("requirements_met", [{}]),
        (
            "issues",
            [
                {
                    "code": "TOO_SHORT",
                    "category": "CONTENT_DEFECT",
                    "description": "内容过短",
                    "suggestion": "扩写",
                    "severity": "major",
                },
                {},
            ],
        ),
        ("requirements_missing", ["citation", {}]),
        ("requirements_met", ["background", {}]),
    ],
)
def test_malformed_assessment_elements_fail_the_entire_contract(
    field, malformed_value
):
    state = _state()
    assessment = {
        "status": "PASS",
        "current_section": "引言",
        "issues": [],
        "requirements_met": [],
        "requirements_missing": [],
        field: malformed_value,
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, state)
    first = decide_recovery_action(state, sanitized)
    second = decide_recovery_action({**state, **first}, sanitized)

    assert sanitized["status"] == "FAILED"
    assert sanitized["issues"][0]["code"] == "ASSESSMENT_CONTRACT_ERROR"
    assert sanitized["issues"][0]["category"] == "VERIFIER_FAILURE"
    assert first["workflow_action"] == "RETRY_VERIFIER"
    assert first["task_retry_count"] == state["task_retry_count"]
    assert second["workflow_action"] == "NEEDS_USER_INPUT"
    assert second["task_retry_count"] == state["task_retry_count"]


@pytest.mark.parametrize(
    "malformed_issue",
    [
        {
            "code": "TOO_SHORT",
            "category": 1,
            "description": "内容过短",
            "suggestion": "扩写",
            "severity": "major",
        },
        {
            "code": "TOO_SHORT",
            "category": "CONTENT_DEFECT",
            "description": [],
            "suggestion": "扩写",
            "severity": "major",
        },
        {
            "code": "TOO_SHORT",
            "category": "CONTENT_DEFECT",
            "description": "内容过短",
            "suggestion": None,
            "severity": "major",
        },
    ],
)
def test_issue_detail_types_are_part_of_the_assessment_contract(malformed_issue):
    assessment = {
        "status": "FAILED",
        "issues": [malformed_issue],
        "requirements_met": [],
        "requirements_missing": [],
    }

    sanitized = auto_verifier_module._sanitize_assessment(assessment, _state())

    assert sanitized["issues"][0]["code"] == "ASSESSMENT_CONTRACT_ERROR"
    assert sanitized["issues"][0]["category"] == "VERIFIER_FAILURE"

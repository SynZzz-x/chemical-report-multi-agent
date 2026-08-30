from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evidence.claims import derive_claims, find_uncited_material_claims
from src.evidence.text_projection import (
    normalize_evidence_text,
    presentation_evidence_excerpt,
    semantic_evidence_excerpt,
)
from src.nodes import verifier as auto_verifier_module


E6_TREND_EVIDENCE = {
    "evidence_id": "E6",
    "title": "聚乙烯质量趋势记录",
    "locator": "表 6",
    "supporting_text": (
        "趋势记录显示反应温度与熔融指数同步升高，"
        "但未记录实验设计、分析误差或公用工程波动。"
    ),
}


@pytest.mark.parametrize("text", [
    "氢气/乙烯比是控制MFR档位最直接、最灵敏的调节手段。",
    "共聚单体/乙烯比是密度档位的核心控制变量。",
])
def test_uncited_material_factual_assertion_is_detected(text):
    assert find_uncited_material_claims(text) == [{"text": text, "claim_type": "factual"}]


@pytest.mark.parametrize("text", [
    "建议优先检查氢气比。",
    "现有证据不足，因此不作判断。",
    "### 核心控制变量",
    "| 核心 | 变量 |",
    "必须",
])
def test_non_factual_or_non_prose_material_words_do_not_trigger(text):
    assert find_uncited_material_claims(text) == []


@pytest.mark.parametrize("text", [
    "| 参数 | 结论 |\n| --- | --- |\n| 共聚单体/乙烯比 | 是密度档位的核心控制变量。 |",
    "~~~text\n共聚单体/乙烯比是密度档位的核心控制变量。\n~~~",
    "`共聚单体/乙烯比是密度档位的核心控制变量。`",
    "``共聚单体/乙烯比是密度档位的核心控制变量。`仍是代码``",
    "参数 | 结论\n--- | ---\n共聚单体/乙烯比 | 是密度档位的核心控制变量。",
])
def test_uncited_detector_masks_structural_and_code_content_before_splitting(text):
    assert find_uncited_material_claims(text) == []


def test_prose_with_pipe_without_markdown_separator_remains_checked():
    text = "共聚单体/乙烯比是密度档位的核心控制变量 | 需要相邻引用。"
    assert find_uncited_material_claims(text) == [{"text": text, "claim_type": "factual"}]


def test_heading_is_masked_without_hiding_following_material_prose():
    text = "### 核心控制变量\n共聚单体/乙烯比是密度档位的核心控制变量。"
    assert find_uncited_material_claims(text) == [{"text": text.split("\n", 1)[1], "claim_type": "factual"}]


def test_material_inference_requires_premise_citation():
    text = "据此可推测反应压力升高会显著扩大MWD。"
    assert find_uncited_material_claims(text) == [{"text": text, "claim_type": "inference"}]


def test_generic_explicit_inference_without_strong_material_assertion_is_not_detected():
    assert find_uncited_material_claims("综合来看，聚合反应是一个复杂过程。") == []


def test_uncited_material_quantitative_assertion_is_detected():
    text = "催化剂浓度提高 10%。"
    assert find_uncited_material_claims(text) == [{"text": text, "claim_type": "factual"}]


def test_material_sentence_does_not_inherit_previous_citation():
    text = "氢气影响熔指。[E1] 共聚单体/乙烯比是密度档位的核心控制变量。"
    assert find_uncited_material_claims(text) == [{
        "text": "共聚单体/乙烯比是密度档位的核心控制变量。",
        "claim_type": "factual",
    }]


def test_ascii_period_preserves_citation_adjacency_for_detector_and_claim_derivation():
    citations = [{"evidence_id": "E1", "title": "趋势", "supporting_text": "氢气影响熔指。"}]
    text = "氢气影响熔指.[E1] 共聚单体/乙烯比是密度档位的核心控制变量。"

    assert find_uncited_material_claims(text) == [{
        "text": "共聚单体/乙烯比是密度档位的核心控制变量。",
        "claim_type": "factual",
    }]
    assert derive_claims(text, citations)[0]["text"] == "氢气影响熔指.[E1]"


@pytest.mark.parametrize("first_sentence", ["氢气影响MFR.", "反应温度为80°C."])
def test_ascii_period_after_acronym_or_unit_preserves_citation_adjacency(first_sentence):
    citations = [{"evidence_id": "E1", "title": "趋势", "supporting_text": "已知。"}]
    text = f"{first_sentence}[E1] 共聚单体/乙烯比是密度档位的核心控制变量。"

    assert find_uncited_material_claims(text) == [{
        "text": "共聚单体/乙烯比是密度档位的核心控制变量。",
        "claim_type": "factual",
    }]
    assert derive_claims(text, citations)[0]["text"] == f"{first_sentence}[E1]"


def test_ascii_period_whitespace_boundary_does_not_share_preceding_citation():
    citations = [{"evidence_id": "E1", "title": "趋势", "supporting_text": "氢气影响熔指。"}]
    text = "氢气影响熔指[E1]. MFR是密度档位的核心控制变量。"

    assert find_uncited_material_claims(text) == [{
        "text": "MFR是密度档位的核心控制变量。",
        "claim_type": "factual",
    }]
    assert derive_claims(text, citations) == [{
        "claim_id": "C1",
        "text": "氢气影响熔指[E1].",
        "claim_type": "factual",
        "evidence_ids": ["E1"],
        "evidence": [{
            "evidence_id": "E1",
            "title": "趋势",
            "locator": "",
            "semantic_evidence_excerpt": "氢气影响熔指。",
        }],
    }]


@pytest.mark.parametrize("text", [
    "密度为 0.25 [E1]",
    "来源为 https://example.com [E1]",
])
def test_ascii_sentence_handling_does_not_split_decimal_or_url_tokens(text):
    citations = [{"evidence_id": "E1", "title": "来源", "supporting_text": "已知。"}]

    assert derive_claims(text, citations)[0]["text"] == text


def _assessment(*, code: str | None = None) -> dict:
    if code is None:
        return {
            "status": "PASS",
            "current_section": "证据核验",
            "issues": [],
            "requirements_met": [],
            "requirements_missing": [],
        }
    return {
        "status": "FAILED",
        "current_section": "证据核验",
        "issues": [
            {
                "code": code,
                "category": "EVIDENCE_GAP",
                "description": "证据不能完整支持该论断。",
                "suggestion": "缩小论断范围或补充直接证据。",
                "severity": "major",
                "requirement_ids": [],
            }
        ],
        "requirements_met": [],
        "requirements_missing": ["论断支持"],
    }


@pytest.fixture
def fake_verifier(monkeypatch):
    calls: list[str] = []
    pending: list[dict] = []
    template = (
        Path(__file__).parents[1] / "src" / "prompts" / "verifier.md"
    ).read_text(encoding="utf-8")

    class Model:
        def invoke(self, prompt, **_kwargs):
            calls.append(template.format(**prompt))
            return SimpleNamespace(
                content=json.dumps(pending.pop(0), ensure_ascii=False)
            )

    monkeypatch.setattr(
        auto_verifier_module,
        "get_llm",
        lambda *_args, **_kwargs: Model(),
    )

    def run(content: str, citations: list[dict], response: dict):
        pending.append(response)
        report_sources = []
        for citation in citations:
            if not str(citation.get("evidence_id") or "").strip():
                continue
            title = str(citation.get("title") or "").strip()
            if title and title not in report_sources:
                report_sources.append(title)
        state = {
            "tasks": [
                {
                    "task_id": "T1",
                    "task_name": "证据核验",
                    "task_description": "核验报告论断的证据支持。",
                    "task_type": "analysis",
                    "use_rag": True,
                    "use_web": False,
                    "requirement_ids": [],
                }
            ],
            "cursor": 0,
            "current_result": {
                "task_id": "T1",
                "status": "COMPLETED",
                "text_output": content,
                "tables": [],
                "figures": [],
                "citations": citations,
                "report_sources": report_sources,
            },
            "requirement_registry": [],
        }
        return auto_verifier_module.verifier(
            state,
            {"configurable": {"use_llm": True}},
        )

    return SimpleNamespace(run=run, calls=calls)


@pytest.mark.parametrize(
    "claim",
    [
        "单变量验证，每次调整仅改变一个因素，用数据验证假设。[E6]",
        "先排除分析误差、公用工程波动等系统性因素。[E6]",
    ],
)
def test_existing_e6_does_not_make_unsupported_claim_pass(claim, fake_verifier):
    result = fake_verifier.run(
        claim,
        [E6_TREND_EVIDENCE],
        _assessment(code="CLAIM_UNSUPPORTED"),
    )

    assert result["assessment"]["status"] == "FAILED"
    assert result["assessment"]["issues"][0]["code"] in {
        "CLAIM_UNSUPPORTED",
        "CLAIM_EVIDENCE_MISMATCH",
    }
    assert '"claim_id": "C1"' in fake_verifier.calls[-1]
    assert claim in fake_verifier.calls[-1]
    assert E6_TREND_EVIDENCE["supporting_text"] in fake_verifier.calls[-1]


def test_supported_factual_claim_is_sent_as_a_claim_evidence_pair(fake_verifier):
    citation = {
        "evidence_id": "E1",
        "title": "温度趋势",
        "locator": "2.1",
        "supporting_text": "反应温度升高会提高熔融指数。",
    }

    result = fake_verifier.run(
        "反应温度升高会提高熔融指数。[E1]",
        [citation],
        _assessment(),
    )

    assert result["assessment"]["status"] == "PASS"
    assert '"claim_id": "C1"' in fake_verifier.calls[-1]
    assert '"claim_type": "factual"' in fake_verifier.calls[-1]
    assert citation["supporting_text"] in fake_verifier.calls[-1]
    assert all(
        code in fake_verifier.calls[-1]
        for code in (
            "CLAIM_UNSUPPORTED",
            "CLAIM_PARTIALLY_SUPPORTED",
            "CLAIM_EVIDENCE_MISMATCH",
            "UNLABELED_INFERENCE",
        )
    )


def test_explicit_inference_is_labeled_in_the_semantic_prompt(fake_verifier):
    citation = {
        "evidence_id": "E1",
        "title": "温度趋势",
        "locator": "2.1",
        "supporting_text": "温度升高时熔融指数随之上升。",
    }

    result = fake_verifier.run(
        "据此可推测升温可能降低分子量。[E1]",
        [citation],
        _assessment(),
    )

    assert result["assessment"]["status"] == "PASS"
    assert '"claim_id": "C1"' in fake_verifier.calls[-1]
    assert '"claim_type": "inference"' in fake_verifier.calls[-1]


def test_factual_looking_derived_conclusion_preserves_unlabeled_inference_code(
    fake_verifier,
):
    citation = {
        "evidence_id": "E1",
        "title": "温度趋势",
        "locator": "2.1",
        "supporting_text": "温度升高时熔融指数随之上升。",
    }

    result = fake_verifier.run(
        "升温通过链转移降低分子量。[E1]",
        [citation],
        _assessment(code="UNLABELED_INFERENCE"),
    )

    assert result["assessment"]["issues"][0]["code"] == "UNLABELED_INFERENCE"
    assert result["assessment"]["issues"][0]["category"] == "EVIDENCE_GAP"
    assert '"claim_id": "C1"' in fake_verifier.calls[-1]


def test_evidence_gap_disclosure_without_citation_reaches_semantic_prompt(
    fake_verifier,
):
    disclosure = "当前知识库未检出压力对分子量分布的直接证据。"

    result = fake_verifier.run(disclosure, [], _assessment())

    assert result["assessment"]["status"] == "PASS"
    assert '"claim_id": "C1"' in fake_verifier.calls[-1]
    assert '"claim_type": "evidence_gap"' in fake_verifier.calls[-1]
    assert '"evidence_ids": []' in fake_verifier.calls[-1]


def test_temperature_mechanism_can_remain_partially_supported(fake_verifier):
    citation = {
        "evidence_id": "E1",
        "title": "温度影响",
        "locator": "3.4",
        "supporting_text": "反应温度会影响产品分子量。",
    }

    result = fake_verifier.run(
        "反应温度升高通过促进链转移降低分子量。[E1]",
        [citation],
        _assessment(code="CLAIM_PARTIALLY_SUPPORTED"),
    )

    assert result["assessment"]["issues"][0]["code"] == (
        "CLAIM_PARTIALLY_SUPPORTED"
    )
    assert result["assessment"]["issues"][0]["category"] == "EVIDENCE_GAP"
    assert '"claim_id": "C1"' in fake_verifier.calls[-1]


def test_invented_identity_preflight_makes_zero_semantic_model_calls(
    fake_verifier, monkeypatch
):
    monkeypatch.setattr(
        auto_verifier_module,
        "derive_claims",
        lambda *_args, **_kwargs: pytest.fail(
            "citation identity preflight must run before claim derivation"
        ),
    )
    result = fake_verifier.run(
        "未知来源的确定结论。[E404]",
        [],
        _assessment(),
    )

    assert result["assessment"]["status"] == "FAILED"
    assert result["assessment"]["issues"][0]["code"] == "INVALID_CITATION_ID"
    assert fake_verifier.calls == []


@pytest.mark.parametrize(
    "citations",
    [
        [],
        [{"title": "missing id"}],
    ],
)
def test_malformed_citation_identity_preflight_fails_without_semantic_call(
    citations, fake_verifier, monkeypatch
):
    monkeypatch.setattr(
        auto_verifier_module,
        "derive_claims",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed citation preflight must run before claim derivation"
        ),
    )

    result = fake_verifier.run("结论。[E-bad]", citations, _assessment())

    assert result["assessment"]["status"] == "FAILED"
    assert result["assessment"]["issues"][0]["code"] == "INVALID_CITATION_ID"
    assert result["assessment"]["issues"][0]["category"] == "EVIDENCE_GAP"
    assert fake_verifier.calls == []


@pytest.mark.parametrize(
    "ordinary_bracket",
    [
        "[E-mail]",
        "[E = mc²]",
        "[E: enabled]",
        "[E]",
        "[E_critical]",
    ],
)
def test_non_citation_e_brackets_do_not_trigger_identity_preflight(
    ordinary_bracket, fake_verifier
):
    result = fake_verifier.run(
        f"正文包含普通标记 {ordinary_bracket}。",
        [],
        _assessment(),
    )

    assert result["assessment"]["status"] == "PASS"
    assert len(fake_verifier.calls) == 1


@pytest.mark.parametrize(
    "malformed_marker",
    ["[E12x]", "[E-foo1]", "[E:1]", "[E=1]", "[E/1]", "[E.1]"],
)
def test_e_prefixed_tokens_with_digits_are_malformed_citation_intent(
    malformed_marker, fake_verifier, monkeypatch
):
    monkeypatch.setattr(
        auto_verifier_module,
        "derive_claims",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed digit-bearing evidence token must fail in preflight"
        ),
    )

    result = fake_verifier.run(f"结论。{malformed_marker}", [], _assessment())

    assert result["assessment"]["status"] == "FAILED"
    assert result["assessment"]["issues"][0]["code"] == "INVALID_CITATION_ID"
    assert fake_verifier.calls == []


def test_semantic_excerpt_preserves_late_support_while_presentation_stays_short():
    raw = "背景" * 2200 + "氢气乙烯比升高导致分子量下降。"

    assert "分子量下降" in semantic_evidence_excerpt(raw)
    assert len(presentation_evidence_excerpt(raw)) <= 240


def test_text_projection_normalizes_noise_without_mutating_source_object():
    source = {"supporting_text": "  表格|||\n\n反应温度\t影响熔融指数。  "}
    original = deepcopy(source)

    assert normalize_evidence_text(source) == "表格| 反应温度 影响熔融指数。"
    assert source == original


def test_evidence_gap_without_citation_is_derived_locally():
    claims = derive_claims(
        "当前知识库未检出反应压力对 Mw/Mn 的直接证据，本报告不作推断。",
        [],
    )

    assert claims == [{
        "claim_id": "C1",
        "text": "当前知识库未检出反应压力对 Mw/Mn 的直接证据，本报告不作推断。",
        "claim_type": "evidence_gap",
        "evidence_ids": [],
        "evidence": [],
    }]


def test_cited_factual_claim_projects_its_validated_evidence():
    citations = [{
        "evidence_id": "E1",
        "title": "聚乙烯工艺说明",
        "locator": "4.1 反应条件",
        "supporting_text": "反应温度升高会提高熔融指数。",
    }]

    claims = derive_claims("反应温度升高会提高熔融指数。[E1]", citations)

    assert claims == [{
        "claim_id": "C1",
        "text": "反应温度升高会提高熔融指数。[E1]",
        "claim_type": "factual",
        "evidence_ids": ["E1"],
        "evidence": [{
            "evidence_id": "E1",
            "title": "聚乙烯工艺说明",
            "locator": "4.1 反应条件",
            "semantic_evidence_excerpt": "反应温度升高会提高熔融指数。",
        }],
    }]


def test_explicit_inference_and_recommendation_use_explicit_markers():
    citations = [
        {"evidence_id": "E1", "title": "趋势", "supporting_text": "温度影响熔指。"},
        {"evidence_id": "E2", "title": "操作", "supporting_text": "应记录温度。"},
    ]

    claims = derive_claims(
        "综合来看，升温可能降低分子量。[E1]\n\n工程上可考虑降低温度以稳定产品。[E2]",
        citations,
    )

    assert [claim["claim_type"] for claim in claims] == ["inference", "recommendation"]
    assert [claim["evidence_ids"] for claim in claims] == [["E1"], ["E2"]]


def test_compound_cited_claim_stays_one_claim_with_multiple_known_evidence_ids():
    citations = [
        {"evidence_id": "E2", "title": "压力", "supporting_text": "压力影响密度。"},
        {"evidence_id": "E1", "title": "温度", "supporting_text": "温度影响熔指。"},
    ]

    claims = derive_claims(
        "升温提高熔融指数且升压提高密度。[E2, E1]",
        citations,
    )

    assert len(claims) == 1
    assert claims[0]["evidence_ids"] == ["E2", "E1"]
    assert [item["evidence_id"] for item in claims[0]["evidence"]] == ["E2", "E1"]


def test_single_line_break_inside_a_cited_sentence_keeps_the_full_claim():
    citations = [{"evidence_id": "E1", "title": "趋势", "supporting_text": "温度影响熔指和密度。"}]

    claims = derive_claims("升温提高熔融指数\n并提高密度。[E1]", citations)

    assert len(claims) == 1
    assert "升温提高熔融指数" in claims[0]["text"]
    assert "并提高密度。[E1]" in claims[0]["text"]


def test_adjacent_normalized_citations_remain_bound_to_one_sentence():
    citations = [
        {"evidence_id": "E1", "title": "温度", "supporting_text": "温度影响熔指。"},
        {"evidence_id": "E2", "title": "压力", "supporting_text": "压力影响密度。"},
    ]

    claims = derive_claims("升温提高熔融指数并且升压提高密度。[E1][E2]", citations)

    assert len(claims) == 1
    assert claims[0]["text"] == "升温提高熔融指数并且升压提高密度。[E1][E2]"
    assert claims[0]["evidence_ids"] == ["E1", "E2"]


def test_claim_text_is_bounded_while_retaining_a_late_citation():
    citations = [{"evidence_id": "E1", "title": "已知", "supporting_text": "已知证据。"}]

    claims = derive_claims("背景" * 1200 + "结论。[E1]", citations)

    assert len(claims) == 1
    assert len(claims[0]["text"]) <= 1000
    assert "[E1]" in claims[0]["text"]


def test_unknown_citation_is_kept_as_a_claim_but_never_resolved():
    citations = [{"evidence_id": "E1", "title": "已知", "supporting_text": "已知证据。"}]

    claims = derive_claims("未知来源的结论。[E99]", citations)

    assert claims == [{
        "claim_id": "C1",
        "text": "未知来源的结论。[E99]",
        "claim_type": "factual",
        "evidence_ids": [],
        "evidence": [],
    }]

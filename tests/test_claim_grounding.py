from copy import deepcopy

from src.evidence.claims import derive_claims
from src.evidence.text_projection import (
    normalize_evidence_text,
    presentation_evidence_excerpt,
    semantic_evidence_excerpt,
)


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

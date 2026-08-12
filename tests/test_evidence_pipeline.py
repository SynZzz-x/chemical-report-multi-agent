import pytest

from src.evidence.coordinator import EvidenceCoordinator
from src.evidence.coverage import assess_coverage
from src.evidence.models import EvidenceBundle, EvidenceRecord
from src.evidence.normalizer import normalize_rag_tool_calls
from src.evidence.web import _host_matches, _is_public_ip


def _rag_call(*evidence):
    return {
        "tool": "chemical_knowledge_base_tool",
        "success": True,
        "parameters": {"query": "反应温度 熔融指数"},
        "full_result": {
            "retrieval_available": True,
            "has_evidence": True,
            "evidence": list(evidence),
        },
    }


def test_rag_evidence_normalization_preserves_provenance_and_deduplicates():
    item = {
        "title": "聚乙烯生产工艺与质量控制概述",
        "source": "/srv/docs/polyethylene.docx",
        "section_path": "4.1 反应条件",
        "pages": {"start": 8, "end": 9},
        "chunk_ids": ["child_1"],
        "content": "反应温度升高会降低聚合物分子量并提高熔融指数。",
    }

    bundle = normalize_rag_tool_calls([_rag_call(item), _rag_call(dict(item))])

    assert len(bundle.records) == 1
    record = bundle.records[0]
    assert record.source_type == "rag"
    assert record.title == item["title"]
    assert record.file_path == item["source"]
    assert record.locator == item["section_path"]
    assert record.chunk_ids == ("child_1",)


def test_coverage_requires_every_requested_concept():
    bundle = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺说明",
                supporting_text="反应温度影响熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    report = assess_coverage(bundle, ["反应温度", "熔融指数", "密度"])

    assert report.status == "insufficient"
    assert report.covered_concepts == ("反应温度", "熔融指数")
    assert report.uncovered_concepts == ("密度",)
    assert report.web_fallback_required is True


def test_coverage_uses_auditable_aliases_for_equivalent_process_terms():
    bundle = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺说明",
                supporting_text="反应温度和反应压力会影响聚合过程。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    report = assess_coverage(bundle, ["聚合温度", "聚合压力"])

    assert report.status == "sufficient"
    assert report.covered_concepts == ("聚合温度", "聚合压力")
    assert report.uncovered_concepts == ()


@pytest.mark.parametrize(
    ("evidence_text", "required_concept"),
    [
        ("装置记录了催化剂类型。", "催化剂体系"),
        ("循环气中的氢气含量发生变化。", "氢气加入量"),
        ("产品共聚单体含量为控制指标。", "共聚单体加入量"),
    ],
)
def test_coverage_does_not_equate_distinct_process_properties(
    evidence_text,
    required_concept,
):
    bundle = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺说明",
                supporting_text=evidence_text,
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    report = assess_coverage(bundle, [required_concept])

    assert report.status == "insufficient"
    assert report.uncovered_concepts == (required_concept,)


class _FakeWebProvider:
    def __init__(self):
        self.queries = []

    def search(self, queries):
        self.queries.extend(queries)
        return (
            EvidenceRecord(
                evidence_id="web-temp",
                source_type="web",
                title="公开技术资料",
                supporting_text="乙烯分压变化会影响聚合速率与产品密度。",
                url="https://example.org/polyethylene-pressure",
                accessed_at="2026-08-04T00:00:00Z",
                source_quality="institutional",
            ),
        )


def test_coordinator_skips_web_when_rag_is_sufficient():
    provider = _FakeWebProvider()
    coordinator = EvidenceCoordinator(web_provider=provider, max_web_queries=3)
    bundle = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺说明",
                supporting_text="反应温度直接影响熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    completed, coverage = coordinator.complete(
        bundle,
        required_concepts=["反应温度", "熔融指数"],
        web_queries=["反应温度 熔融指数"],
        allow_web_fallback=True,
    )

    assert provider.queries == []
    assert completed == bundle
    assert coverage.status == "sufficient"


def test_coordinator_uses_web_only_for_uncovered_concepts():
    provider = _FakeWebProvider()
    coordinator = EvidenceCoordinator(web_provider=provider, max_web_queries=3)
    bundle = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺说明",
                supporting_text="反应温度直接影响熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
        )
    )

    completed, coverage = coordinator.complete(
        bundle,
        required_concepts=["反应温度", "熔融指数", "乙烯分压", "密度"],
        web_queries=["反应温度 熔融指数", "乙烯分压 产品密度"],
        allow_web_fallback=True,
    )

    assert provider.queries == ["乙烯分压 产品密度"]
    assert len(completed.records) == 2
    assert completed.records[1].evidence_id == "E2"
    assert coverage.status == "sufficient"


def test_web_security_helpers_reject_private_addresses_and_domain_spoofs():
    assert _is_public_ip("127.0.0.1") is False
    assert _is_public_ip("169.254.169.254") is False
    assert _is_public_ip("8.8.8.8") is True
    assert _host_matches("agency.gov.cn", "gov.cn") is True
    assert _host_matches("agency.gov.cn.evil.example", "gov.cn") is False

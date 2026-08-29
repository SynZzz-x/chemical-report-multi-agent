import copy
from pathlib import Path

from docx import Document

from src.evidence import reporting
from src.evidence.projection import canonical_source_identity
from src.utils.md_to_docx import md_to_docx
from src.utils.md_to_pdf import md_to_pdf


def test_long_four_column_evidence_appendix_renders_to_pdf(tmp_path):
    citations = [
        {
            "evidence_id": f"E{index}",
            "title": "聚乙烯生产工艺与质量控制概述",
            "file_path": (
                "/home/zsy/my_project/agent_project/chemical-report-multi-agent/"
                "document/聚乙烯生产工艺与质量控制概述.docx"
            ),
            "locator": f"第{index}章/反应条件与质量指标/详细定位",
            "supporting_text": "温度、压力和催化剂对产品质量的证据摘要。" * 12,
            "section_title": f"第{index}节",
        }
        for index in range(1, 35)
    ]
    evidence = reporting.format_grouped_evidence_appendix(
        citations,
        heading_level=2,
    )
    markdown = f"# 报告\n\n{evidence}\n"
    output = tmp_path / "evidence.pdf"

    md_to_pdf(markdown, str(output))

    assert output.exists()
    assert output.stat().st_size > 0
    assert "/home/zsy" not in evidence


def test_grouped_evidence_appendix_is_stable_safe_bounded_and_read_only():
    long_support = "背景 ||| 表格噪声 ||| " + "温度影响分子量。" * 80
    citations = [
        {
            "evidence_id": "E10",
            "source_type": "rag",
            "title": "/cache/users/u-17/jobs/j-42/process.docx",
            "file_path": "/a/process.docx",
            "locator": "§5 关键工艺参数 / page 6 / table 2",
            "section_title": "工艺分析",
            "supporting_text": long_support,
            "chunk_id": "rag_7f_internal",
        },
        {
            "evidence_id": "E1",
            "source_type": "rag",
            "file_path": "/a/process.docx",
            "locator": "§4 原料条件",
            "section_title": "工艺分析",
            "supporting_text": "原料条件影响产品质量。",
        },
        {
            "evidence_id": "E2",
            "source_type": "rag",
            "file_path": "/a/process.docx",
            "locator": "page 3",
            "section_title": "背景",
            "supporting_text": "氢气影响熔指。",
        },
        {
            "evidence_id": "E4",
            "source_type": "rag",
            "title": "质量指南",
            "file_path": "/b/quality.pdf",
            "locator": "table 1",
            "section_title": "质量控制",
            "supporting_text": "压力影响产品质量。",
        },
        {
            "evidence_id": "E3",
            "source_type": "rag",
            "file_path": "/other/process.docx",
            "locator": "/cache/users/u-17/chunks/rag_99",
            "section_title": "附录",
            "supporting_text": "重复来源。",
        },
    ]
    original = copy.deepcopy(citations)

    assert hasattr(reporting, "format_grouped_evidence_appendix")
    first = reporting.format_grouped_evidence_appendix(
        citations,
        heading_level=2,
        heading_title="证据索引",
    )
    second = reporting.format_grouped_evidence_appendix(
        citations,
        heading_level=2,
        heading_title="证据索引",
    )

    assert first == second
    assert citations == original
    assert first.count("### process.docx") == 1
    assert first.index("### process.docx") < first.index("### 质量指南")
    assert (
        first.index("[E1]")
        < first.index("[E2]")
        < first.index("[E3]")
        < first.index("[E10]")
    )
    assert "§5 关键工艺参数 / page 6 / table 2" in first
    assert "table 1" in first
    assert "/cache" not in first
    assert "/a/process.docx" not in first
    assert "u-17" not in first
    assert "j-42" not in first
    assert "rag_7f_internal" not in first
    assert "rag_99" not in first
    assert "|||" not in first
    assert long_support not in first
    assert len(first[first.index("背景"):]) < len(long_support)
    assert canonical_source_identity(citations[0]) == canonical_source_identity(
        citations[4]
    )


def test_docx_heading_number_never_starts_with_zero_when_parent_is_missing(tmp_path):
    output = tmp_path / "heading.docx"

    md_to_docx("# 报告\n\n### 反应温度\n\n正文。", str(output))

    document = Document(output)
    headings = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.style.name.startswith("Heading")
    ]
    assert "1.1 反应温度" in headings
    assert all(not heading.startswith("0.") for heading in headings)

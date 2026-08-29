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
            "section_title": "/cache/users/u-17/jobs/j-42/internal-section",
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


def test_grouped_appendix_never_displays_canonical_web_identity_as_label():
    citations = [
        {
            "evidence_id": "E1",
            "source_type": "web",
            "url": (
                "https://internal.example/cache/users/u-secret/report"
                "?conversation_id=c-secret&job_id=j-secret"
            ),
            "supporting_text": "公开结论。",
        },
        {
            "evidence_id": "E2",
            "source_type": "rag",
            "title": "cache/users/u-label",
            "file_path": "/cache/users/u-label",
            "supporting_text": "知识库结论。",
        },
        {
            "evidence_id": "E3",
            "source_type": "rag",
            "title": "/cache/users/alice",
            "file_path": "/cache/users/alice",
            "supporting_text": "第一条匿名来源。",
        },
        {
            "evidence_id": "E4",
            "source_type": "rag",
            "file_name": "/cache/jobs/12345",
            "file_path": "/cache/jobs/12345",
            "supporting_text": "第二条匿名来源。",
        },
    ]
    original = copy.deepcopy(citations)

    appendix = reporting.format_grouped_evidence_appendix(citations)

    assert citations == original
    assert "#### 网页来源" in appendix
    assert appendix.count("#### 知识库文档") == 3
    assert "internal.example" not in appendix
    assert "u-secret" not in appendix
    assert "c-secret" not in appendix
    assert "j-secret" not in appendix
    assert "conversation_id" not in appendix
    assert "job_id" not in appendix
    assert "u-label" not in appendix
    assert "alice" not in appendix
    assert "12345" not in appendix


def test_grouped_appendix_redacts_internal_references_from_all_display_text():
    citations = [
        {
            "evidence_id": "E1",
            "source_type": "rag",
            "title": "公开工艺手册",
            "file_path": "/srv/cache/users/u-private/source.docx",
            "locator": (
                "§5 / page 6 / table 2；cache/users/u-rel/jobs/j-rel/chunk.txt；"
                "/Users/alice/private/report.docx；file:///tmp/job-j-file/report.pdf；"
                "https://internal.example/report?conversation_id=c-query；"
                "https://example.org/report?userId=u-query-camel&"
                "conversation-id=c-query-dash"
            ),
            "section_title": (
                "工艺分析 cache/conversations/c-rel/jobs/j-section/notes.md"
            ),
            "supporting_text": (
                "公开摘要；来源 /home/alice/cache/users/u-summary/input.docx；"
                "chunk_id=rag_7f_internal；user_id=u-inline；"
                "job_id: j-colon；userId=u-camel；conversation-id=c-dash；"
                "https://internal.example/chunk?job_id=j-query"
            ),
        }
    ]
    original = copy.deepcopy(citations)

    appendix = reporting.format_grouped_evidence_appendix(citations)

    assert citations == original
    assert "§5 / page 6 / table 2" in appendix
    assert "工艺分析" in appendix
    assert "公开摘要" in appendix
    for internal_value in (
        "cache/users",
        "cache/conversations",
        "/Users/alice",
        "/home/alice",
        "file://",
        "internal.example",
        "u-rel",
        "j-rel",
        "j-file",
        "c-query",
        "u-query-camel",
        "c-query-dash",
        "c-rel",
        "j-section",
        "u-summary",
        "rag_7f_internal",
        "u-inline",
        "j-query",
        "j-colon",
        "u-camel",
        "c-dash",
        "conversation_id",
        "chunk_id",
        "user_id",
        "job_id",
        "userId",
        "conversation-id",
        "example.org/report?userId",
    ):
        assert internal_value not in appendix


def test_grouped_appendix_preserves_scientific_text_and_public_url_exactly():
    citations = [
        {
            "evidence_id": "E1",
            "source_type": "web",
            "title": "公开工艺数据",
            "url": "https://example.org/report",
            "locator": "§5 / page 6 / table 2 / https://example.org/report",
            "section_title": "传质速率 kg/m/s",
            "supporting_text": (
                "速率单位 kg/m/s、mol/L/min；比较 RAG1/RAG2 与 "
                "RAG-1/RAG-2 assay。"
            ),
        },
        {
            "evidence_id": "E2",
            "source_type": "rag",
            "title": "/public/plots/j-curve.pdf",
            "file_path": "/public/plots/j-curve.pdf",
            "supporting_text": "公开曲线文件。",
        }
    ]
    original = copy.deepcopy(citations)

    appendix = reporting.format_grouped_evidence_appendix(citations)

    assert citations == original
    assert "https://example.org/report" in appendix
    assert "§5 / page 6 / table 2" in appendix
    assert "kg/m/s" in appendix
    assert "mol/L/min" in appendix
    assert "RAG1/RAG2" in appendix
    assert "RAG-1/RAG-2 assay" in appendix
    assert "#### j-curve.pdf" in appendix


def test_grouped_appendix_redacts_url_fragments_decoded_values_and_private_ips():
    citations = [
        {
            "evidence_id": "E1",
            "source_type": "web",
            "title": "URL 边界测试",
            "locator": (
                "§5；https://example.org/report#jobId=j-frag&"
                "conversation-id=c-frag&user_id=u-frag；"
                "https://example.org/go?redirect=%2Fcache%2Fusers%2Falice%2Fjobs%2Fj-9；"
                "http://127.0.0.1/private/report"
            ),
            "supporting_text": "公开说明 page 6 table 2。",
        }
    ]
    original = copy.deepcopy(citations)

    appendix = reporting.format_grouped_evidence_appendix(citations)

    assert citations == original
    assert "§5" in appendix
    assert "page 6 table 2" in appendix
    for internal_value in (
        "j-frag",
        "c-frag",
        "u-frag",
        "jobId",
        "conversation-id",
        "user_id",
        "redirect=",
        "%2Fcache",
        "alice",
        "j-9",
        "127.0.0.1",
    ):
        assert internal_value not in appendix


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

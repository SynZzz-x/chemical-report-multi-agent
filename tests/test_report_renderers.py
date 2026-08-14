from pathlib import Path

from docx import Document

from src.utils.md_to_docx import md_to_docx
from src.utils.md_to_pdf import md_to_pdf


def test_long_four_column_evidence_appendix_renders_to_pdf(tmp_path):
    rows = "\n".join(
        "| [E{index}] | {source} | {locator} | {summary} |".format(
            index=index,
            source=(
                "/home/zsy/my_project/agent_project/chemical-report-multi-agent/"
                "document/聚乙烯生产工艺与质量控制概述.docx"
            ),
            locator=f"第{index}章/反应条件与质量指标/详细定位",
            summary="温度、压力和催化剂对产品质量的证据摘要。" * 12,
        )
        for index in range(1, 35)
    )
    markdown = (
        "# 报告\n\n## 证据来源\n\n"
        "| 证据编号 | 来源 | 定位 | 摘要 |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n"
    )
    output = tmp_path / "evidence.pdf"

    md_to_pdf(markdown, str(output))

    assert output.exists()
    assert output.stat().st_size > 0


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

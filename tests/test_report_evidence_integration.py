from src.evidence.models import EvidenceBundle, EvidenceRecord
from src.evidence.identity import normalize_sections_evidence
from src.evidence.reporting import append_missing_figures, format_evidence_table
from src.nodes.worker.agent.graph import AutonomousToolNode


def _rag_call():
    return {
        "tool": "chemical_knowledge_base_tool",
        "success": True,
        "parameters": {"query": "反应温度"},
        "full_result": {
            "evidence": [
                {
                    "title": "聚乙烯工艺说明",
                    "source": "/srv/docs/process.docx",
                    "section_path": "4.1 反应条件",
                    "content": "反应温度影响熔融指数。",
                    "chunk_ids": ["C1"],
                }
            ]
        },
    }


def test_task_result_derives_citations_and_sources_from_rag_calls():
    node = AutonomousToolNode.__new__(AutonomousToolNode)

    result = node._create_task_result(
        {
            "task_id": "T3",
            "task_name": "反应条件",
            "use_resources": ["/srv/data/process.csv"],
        },
        2,
        "这是由检索证据支持的正文。" * 10,
        [_rag_call()],
        {"chemical_knowledge_base_tool": 1},
        1.0,
        True,
        False,
    )

    assert result["citations"][0]["evidence_id"] == "E1"
    assert result["sources_used"] == [
        "/srv/data/process.csv",
        "/srv/docs/process.docx",
    ]


def test_evidence_table_contains_document_locator_and_web_url():
    bundle = EvidenceBundle(
        records=(
            EvidenceRecord(
                evidence_id="E1",
                source_type="rag",
                title="工艺说明",
                locator="4.1 反应条件",
                supporting_text="温度影响熔融指数。",
                file_path="/srv/docs/process.docx",
            ),
            EvidenceRecord(
                evidence_id="E2",
                source_type="web",
                title="公开资料",
                supporting_text="压力影响密度。",
                url="https://example.org/source",
                accessed_at="2026-08-04T00:00:00Z",
            ),
        )
    )

    table = format_evidence_table([record.model_dump(mode="json") for record in bundle.records])

    assert "[E1]" in table
    assert "4.1 反应条件" in table
    assert "https://example.org/source" in table


def test_report_reference_projection_is_four_columns_and_hides_absolute_paths():
    table = format_evidence_table(
        [
            {
                "evidence_id": "E1",
                "title": "聚乙烯工艺说明",
                "file_path": "/home/zsy/private/docs/process.docx",
                "supporting_text": "温度影响熔融指数。",
                "section_title": "工艺分析",
            }
        ],
        heading_level=2,
        include_section=True,
    )

    assert "| 证据编号 | 来源与支撑章节 | 定位 | 摘要 |" in table
    assert "[E1] | process.docx（支撑章节：工艺分析） | process.docx |" in table
    assert "/home/zsy" not in table


def test_missing_concept_figure_is_appended_deterministically():
    markdown = append_missing_figures(
        "## 反应条件\n\n正文",
        [
            {
                "path": "/srv/cache/charts/concept_T3.png",
                "description": "参数影响关系图（关系证据：[E1]）",
            }
        ],
    )

    assert "![参数影响关系图](/srv/cache/charts/concept_T3.png)" in markdown
    assert "<description>参数影响关系图（关系证据：[E1]）</description>" in markdown


def test_cross_task_local_ids_receive_distinct_deterministic_display_ids():
    sections = [
        {
            "task_id": "T1",
            "text": "温度结论 [E8]。",
            "citations": [{"evidence_id": "E8", "title": "温度资料"}],
            "figures": [{"path": "/tmp/a.png", "evidence_ids": ["E8"]}],
        },
        {
            "task_id": "T2",
            "text": "压力结论 [E8]。",
            "citations": [{"evidence_id": "E8", "title": "压力资料"}],
            "figures": [{"path": "/tmp/b.png", "evidence_ids": ["E8"]}],
            "tables": [
                {
                    "title": "压力表",
                    "evidence_id": "E8",
                    "metadata": {"evidence_ids": ["E8"]},
                }
            ],
            "graph_spec": {
                "edges": [{"source": "压力", "target": "密度", "evidence_ids": ["E8"]}]
            },
        },
    ]

    normalized, display_map = normalize_sections_evidence(sections)

    assert display_map == {"T1:E8": "E1", "T2:E8": "E2"}
    assert len(normalized[0]["citations"]) == len(normalized[1]["citations"]) == 1
    assert normalized[0]["citations"][0]["evidence_key"] == "T1:E8"
    assert normalized[1]["citations"][0]["evidence_key"] == "T2:E8"
    assert normalized[0]["text"] == "温度结论 [E1]。"
    assert normalized[1]["text"] == "压力结论 [E2]。"
    assert normalized[0]["figures"][0]["evidence_ids"] == ["E1"]
    assert normalized[1]["figures"][0]["evidence_ids"] == ["E2"]
    assert normalized[1]["tables"][0]["evidence_id"] == "E2"
    assert normalized[1]["tables"][0]["metadata"]["evidence_ids"] == ["E2"]
    assert normalized[1]["graph_spec"]["edges"][0]["evidence_ids"] == ["E2"]

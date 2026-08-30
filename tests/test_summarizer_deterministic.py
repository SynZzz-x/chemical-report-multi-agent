import json
from copy import deepcopy
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from PIL import Image as PILImage

from src.nodes import summarizer_v2, synthesis as synthesis_module


def _task(task_id, name):
    return {"task_id": task_id, "task_name": name}


def _status(value, issues=None):
    return {
        "status": value,
        "accepted_by": "user" if value.startswith("USER_") else "verifier",
        "issues": list(issues or []),
        "plan_revision": 1,
        "task_revision": 1,
    }


def _state(*, statuses, results=None):
    return {
        "user_id": "u1",
        "conversation_id": "c1",
        "job_id": "j1",
        "tasks": [_task("T1", "引言"), _task("T2", "工艺分析")],
        "results": list(
            results
            or [
                {
                    "task_id": "T2",
                    "text_output": "### 分析方法\n\n工艺分析正文 [E1]。",
                    "plan_revision": 1,
                    "task_revision": 1,
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "title": "工艺手册",
                            "locator": "第2章",
                            "supporting_text": "温度与质量关系",
                        }
                    ],
                },
                {
                    "task_id": "T1",
                    "text_output": "## 引言\n\n### 背景\n\n引言正文 [E1]。",
                    "plan_revision": 1,
                    "task_revision": 1,
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "title": "质量指南",
                            "locator": "第1章",
                            "supporting_text": "质量控制背景",
                        }
                    ],
                },
            ]
        ),
        "section_status": statuses,
        "messages": [
            AIMessage(
                content=json.dumps(
                    {
                        "from": "Intake",
                        "to": "Planner",
                        "title": "聚乙烯质量报告",
                    },
                    ensure_ascii=False,
                )
            )
        ],
    }


def _install_render_stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda state, config: str(tmp_path),
    )

    def write_pdf(markdown, output_path, **kwargs):
        Path(output_path).write_text("pdf", encoding="utf-8")

    def write_docx(markdown, output_path, **kwargs):
        Path(output_path).write_text("docx", encoding="utf-8")

    monkeypatch.setattr(summarizer_v2.md_to_pdf, "md_to_pdf", write_pdf)
    monkeypatch.setattr(summarizer_v2.md_to_docx, "md_to_docx", write_docx)
    monkeypatch.setattr(
        summarizer_v2,
        "get_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deterministic summarizer must not call an LLM")
        ),
        raising=False,
    )


def test_blocked_report_does_not_create_delivery_files(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status(
            "ACCEPT_WITH_WARNING",
            [{"code": "TOO_SHORT", "description": "篇幅不足"}],
        ),
    }
    state = _state(statuses=statuses)
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("blocked report must not create a report directory")
        ),
    )

    update = summarizer_v2.summarizer(state, {})

    result = update["final_result"]
    assert result["report_status"] == "BLOCKED"
    assert result["attachments"] == []
    assert result["path"] is None
    assert result["blocking_sections"][0]["task_id"] == "T2"
    assert "篇幅不足" in result["blocking_sections"][0]["issues"][0]["description"]
    assert not (tmp_path / "report").exists()


def test_final_citation_conflict_blocks_before_delivery_paths(monkeypatch):
    state = _state(
        statuses={"T1": _status("VERIFIED_PASS"), "T2": _status("VERIFIED_PASS")},
        results=[
            {
                "task_id": "T1",
                "text_output": "冲突正文 [E1]。",
                "plan_revision": 1,
                "task_revision": 1,
                "citations": [
                    {
                        "evidence_id": "E1",
                        "evidence_key": "shared-final-display-id",
                        "file_path": "/docs/a.docx",
                        "locator": "1",
                        "supporting_text": "甲",
                    },
                ],
            },
            {
                "task_id": "T2",
                "text_output": "正常正文 [E1]。",
                "plan_revision": 1,
                "task_revision": 1,
                "citations": [
                    {
                        "evidence_id": "E1",
                        "evidence_key": "shared-final-display-id",
                        "file_path": "/docs/c.docx",
                        "locator": "3",
                        "supporting_text": "丙",
                    }
                ],
            },
        ],
    )
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("gate must precede path resolution")
        ),
    )

    update = summarizer_v2.summarizer(state, {})

    assert update["report_status"] == "BLOCKED"
    assert update["final_result"]["attachments"] == []
    assert (
        update["final_result"]["blocking_sections"][0]["issues"][0]["code"]
        == "FINAL_CITATION_INTEGRITY"
    )


def test_markdown_assembly_is_byte_deterministic():
    state = _state(statuses={"T1": _status("VERIFIED_PASS"), "T2": _status("VERIFIED_PASS")})
    sections, _ = summarizer_v2.normalize_sections_evidence(state["results"])

    first = summarizer_v2._assemble_markdown(state, sections, "READY_FOR_FINAL")
    second = summarizer_v2._assemble_markdown(state, sections, "READY_FOR_FINAL")

    assert first == second


def test_normalized_synthesis_is_delivered_with_original_citation_scopes(monkeypatch, tmp_path):
    state = _state(statuses={task_id: _status("VERIFIED_PASS") for task_id in ("T1", "T2")})
    state["results"] = [
        {"task_id": task_id, "text_output": text, "plan_revision": 1, "task_revision": 1,
         "citations": [{"evidence_id": "E8", "file_path": path, "supporting_text": text.split("[")[0]}]}
        for task_id, text, path in (
            ("T1", "温度影响分子量。[E8]", "/docs/a.docx"),
            ("T2", "压力影响密度。[E8]", "/docs/b.docx"),
        )
    ]
    state["tasks"].append({**_task("TS", "结论"), "task_type": "synthesis"})
    state["cursor"] = 2
    state["plan_revision"] = 1
    state["task_revisions"] = {task_id: 1 for task_id in ("T1", "T2", "TS")}
    original_results = deepcopy(state["results"])
    monkeypatch.setattr(synthesis_module, "get_llm", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(synthesis_module, "with_completion_budget", lambda model, purpose: (model, 1000))
    monkeypatch.setattr(synthesis_module, "invoke_llm", lambda *_args, **_kwargs: AIMessage(
        content="温度影响分子量。[E1]\n压力影响密度。[E2]"
    ))
    synthesis_result = synthesis_module.synthesis(state, {})["current_result"]
    assert synthesis_result["status"] == "COMPLETED"
    assert [item["evidence_key"] for item in synthesis_result["citations"]] == ["T1:E8", "T2:E8"]
    state["results"].append(synthesis_result)
    state["section_status"]["TS"] = _status("VERIFIED_PASS")
    original_state = deepcopy({**state, "messages": [message.content for message in state["messages"]]})
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "READY_FOR_FINAL"
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")
    assert "温度影响分子量。[E1]" in markdown
    assert "压力影响密度。[E2]" in markdown
    assert {**state, "messages": [message.content for message in state["messages"]]} == original_state
    assert state["results"][:2] == original_results


@pytest.mark.parametrize("reverse", [False, True])
def test_visible_id_conflict_blocks_before_remap_and_delivery(monkeypatch, tmp_path, reverse):
    citations = [
        {"evidence_id": "E1", "evidence_key": "T1:E8", "local_evidence_id": "E8",
         "file_path": "/docs/a.docx", "supporting_text": "甲"},
        {"evidence_id": "E1", "evidence_key": "T2:E9", "local_evidence_id": "E9",
         "file_path": "/docs/b.docx", "supporting_text": "乙"},
    ]
    if reverse:
        citations.reverse()
    state = _state(statuses={"T1": _status("VERIFIED_PASS")}, results=[{
        "task_id": "T1", "text_output": "正文 [E1]。", "citations": citations,
        "plan_revision": 1, "task_revision": 1,
    }])
    state["tasks"] = state["tasks"][:1]
    calls = []
    original_normalize = summarizer_v2.normalize_sections_evidence

    def record_normalize(sections):
        calls.append("normalize")
        return original_normalize(sections)

    monkeypatch.setattr(summarizer_v2, "normalize_sections_evidence", record_normalize)
    _install_render_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(summarizer_v2, "get_session_cache_dir", lambda *_args: (
        calls.append("paths") or str(tmp_path)
    ))

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["attachments"] == []
    assert result["path"] is None
    assert calls == []
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("with_prior_section", [False, True])
def test_unknown_raw_marker_blocks_before_normalization_paths_and_artifacts(
    monkeypatch, tmp_path, with_prior_section
):
    state = _state(statuses={"T1": _status("VERIFIED_PASS")}, results=[{
        "task_id": "T1",
        "text_output": "已绑定结论 [E8]。未知结论 [E1]。",
        "plan_revision": 1,
        "task_revision": 1,
        "citations": [{"evidence_id": "E8", "file_path": "/docs/a.docx",
                       "supporting_text": "已绑定结论。"}],
    }])
    state["tasks"] = state["tasks"][:1]
    if with_prior_section:
        state["tasks"].insert(0, _task("T0", "前节"))
        state["section_status"]["T0"] = _status("VERIFIED_PASS")
        state["results"].insert(0, {
            "task_id": "T0", "text_output": "前节结论 [E1]。",
            "plan_revision": 1, "task_revision": 1,
            "citations": [{"evidence_id": "E1", "file_path": "/docs/previous.docx",
                           "supporting_text": "前节结论。"}],
        })
    calls = []
    original_normalize = summarizer_v2.normalize_sections_evidence

    def record_normalize(sections):
        calls.append("normalize")
        return original_normalize(sections)

    monkeypatch.setattr(summarizer_v2, "normalize_sections_evidence", record_normalize)
    _install_render_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(summarizer_v2, "get_session_cache_dir", lambda *_args: (
        calls.append("paths") or str(tmp_path)
    ))

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["attachments"] == []
    assert result["path"] is None
    assert "LOCAL_CITATION_BINDING_MISSING" in result["blocking_sections"][0]["issues"][0]["description"]
    assert calls == []
    assert not (tmp_path / "report").exists()


def test_lost_body_marker_cannot_be_masked_by_surviving_appendix(monkeypatch, tmp_path):
    state = _state(statuses={"T1": _status("VERIFIED_PASS")}, results=[{
        "task_id": "T1", "text_output": "正文 [E1]。", "plan_revision": 1, "task_revision": 1,
        "citations": [{"evidence_id": "E1", "file_path": "/docs/a.docx", "supporting_text": "甲"}],
    }])
    state["tasks"] = state["tasks"][:1]
    _install_render_stubs(monkeypatch, tmp_path)
    original_strip = summarizer_v2._strip_duplicate_leading_heading
    monkeypatch.setattr(summarizer_v2, "_strip_duplicate_leading_heading", lambda text, title: (
        original_strip(text, title).replace("[E1]", "")
    ))
    assembled = []
    original_assemble = summarizer_v2._assemble_markdown

    def capture_assembly(*args, **kwargs):
        markdown = original_assemble(*args, **kwargs)
        assembled.append(markdown)
        return markdown

    monkeypatch.setattr(summarizer_v2, "_assemble_markdown", capture_assembly)
    path_calls = []
    monkeypatch.setattr(summarizer_v2, "get_session_cache_dir", lambda *_args: (
        path_calls.append(True) or str(tmp_path)
    ))

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert len(assembled) == 1
    assert "正文 。" in assembled[0]
    assert "[E1]" in assembled[0]  # The evidence appendix still contains the marker.
    assert result["report_status"] == "BLOCKED"
    assert result["attachments"] == []
    assert result["path"] is None
    assert path_calls == []
    assert not (tmp_path / "report").exists()


def test_assembly_body_spans_are_invocation_local_and_markdown_stays_plain_str(monkeypatch, tmp_path):
    state = _state(statuses={"T1": _status("VERIFIED_PASS"), "T2": _status("VERIFIED_PASS")})
    state["tasks"][0]["covers_sections"] = ["1. 引言"]
    state["tasks"][1]["covers_sections"] = ["3. 工艺分析"]
    state["messages"] = [AIMessage(content=json.dumps({
        "from": "Intake", "to": "Planner", "title": "报告",
        "sections": ["1. 引言", "2. 证据来源", "3. 工艺分析"],
    }, ensure_ascii=False))]
    original_state = deepcopy({**state, "messages": [message.content for message in state["messages"]]})
    _install_render_stubs(monkeypatch, tmp_path)
    assemblies = []
    original_assemble = summarizer_v2._assemble_markdown

    def record_assembly(*args, **kwargs):
        markdown = original_assemble(*args, **kwargs)
        assemblies.append((markdown, kwargs.get("body_spans")))
        return markdown

    monkeypatch.setattr(summarizer_v2, "_assemble_markdown", record_assembly)
    rendered = []

    def render(markdown, output_path, **kwargs):
        rendered.append(markdown)
        Path(output_path).write_text("artifact", encoding="utf-8")

    monkeypatch.setattr(summarizer_v2.md_to_pdf, "md_to_pdf", render)
    monkeypatch.setattr(summarizer_v2.md_to_docx, "md_to_docx", render)

    update = summarizer_v2.summarizer(state, {})

    assert update["report_status"] == "READY_FOR_FINAL"
    assert len(assemblies) == 1
    markdown, body_spans = assemblies[0]
    assert body_spans is not None
    assert len(body_spans) == 2
    first_body, second_body = [markdown[start:end] for start, end in body_spans]
    assert "引言正文 [E1]。" in first_body
    assert "工艺分析正文 [E2]。" in second_body
    assert "证据来源" not in first_body + second_body
    assert body_spans[0][1] < markdown.index("## 2. 证据来源") < body_spans[1][0]
    assert type(markdown) is str
    assert len(rendered) == 2 and all(type(value) is str and value == markdown for value in rendered)
    assert Path(update["final_result"]["attachments"][0]).read_text(encoding="utf-8") == markdown
    assert {**state, "messages": [message.content for message in state["messages"]]} == original_state
    assert '"body_spans":' not in json.dumps(update, ensure_ascii=False, default=str)
    assert "body_spans" not in markdown


def test_assembly_body_spans_do_not_classify_body_by_heading_name():
    state = _state(statuses={"T1": _status("VERIFIED_PASS")})
    body_spans = []
    sections = [{
        "task_id": "T1", "title": "引言", "text": "### 证据来源\n\n正文 [E1]。",
        "citations": [{"evidence_id": "E1", "title": "实际来源", "supporting_text": "甲"}],
    }]

    markdown = summarizer_v2._assemble_markdown(
        state, sections, "READY_FOR_FINAL", body_spans=body_spans
    )

    assert len(body_spans) == 1
    start, end = body_spans[0]
    assert "### 证据来源\n\n正文 [E1]。" in markdown[start:end]
    assert "实际来源" not in markdown[start:end]


def test_ready_report_is_assembled_in_task_order_without_llm(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    _install_render_stubs(monkeypatch, tmp_path)

    update = summarizer_v2.summarizer(_state(statuses=statuses), {})

    result = update["final_result"]
    markdown_path = Path(result["attachments"][0])
    markdown = markdown_path.read_text(encoding="utf-8")
    assert result["report_status"] == "READY_FOR_FINAL"
    assert [Path(path).name for path in result["attachments"]] == [
        "report.md",
        "report.pdf",
        "report.docx",
    ]
    assert Path(result["path"]).name == "report.docx"
    assert markdown.index("## 引言") < markdown.index("## 工艺分析")
    assert markdown.count("## 引言") == 1
    assert "### 背景" in markdown
    assert "### 分析方法" in markdown
    assert markdown.count("## 证据来源") == 1
    assert markdown.count("### 质量指南") == 1
    assert markdown.count("### 工艺手册") == 1
    assert "**[E1] 第1章**" in markdown
    assert "支撑章节：引言" in markdown
    assert "摘要：质量控制背景" in markdown
    assert "**[E2] 第2章**" in markdown
    assert "支撑章节：工艺分析" in markdown
    assert "摘要：温度与质量关系" in markdown
    assert "工艺分析正文 [E2]" in markdown
    assert "LLM Generation Failed" not in markdown


def test_reference_section_is_projected_from_citations_at_outline_position(
    monkeypatch, tmp_path
):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["tasks"] = [
        {**state["tasks"][0], "covers_sections": ["1. 引言"]},
        {**state["tasks"][1], "covers_sections": ["3. 工艺分析"]},
    ]
    state["results"][0]["text_output"] = "## 工艺分析\n\n### 分析方法\n\n工艺分析正文 [E1]。"
    state["messages"] = [
        AIMessage(
            content=json.dumps(
                {
                    "from": "Intake",
                    "to": "Planner",
                    "title": "聚乙烯质量报告",
                    "sections": [
                        "1. 引言",
                        "2. 知识库依据与参考文件说明",
                        "3. 工艺分析",
                    ],
                },
                ensure_ascii=False,
            )
        )
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    reference_heading = "## 2. 知识库依据与参考文件说明"
    assert reference_heading in markdown
    assert markdown.index("引言正文") < markdown.index(reference_heading)
    assert markdown.index(reference_heading) < markdown.index("工艺分析正文")
    assert "质量指南" in markdown
    assert "工艺手册" in markdown
    assert "清单为空" not in markdown


def test_knowledge_base_file_list_is_deterministically_aggregated(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["tasks"] = [
        {**state["tasks"][0], "covers_sections": ["1. 引言"]},
        {**state["tasks"][1], "covers_sections": ["2. 工艺分析"]},
    ]
    state["results"][0]["citations"] = [
        {
            "evidence_id": "E1",
            "source_type": "rag",
            "title": "聚乙烯生产工艺与质量控制概述",
            "file_path": "/srv/private/聚乙烯生产工艺与质量控制概述.docx",
            "locator": "第2章",
            "supporting_text": "工艺说明",
        },
        {
            "evidence_id": "E2",
            "source_type": "rag",
            "title": "聚乙烯生产工艺与质量控制概述",
            "file_path": "/srv/private/聚乙烯生产工艺与质量控制概述.docx",
            "locator": "第3章",
            "supporting_text": "质量说明",
        },
    ]
    state["results"][1]["citations"] = [
        {
            "evidence_id": "E1",
            "source_type": "rag",
            "title": "/srv/private/聚乙烯生产工艺与质量控制概述.docx",
            "file_path": "/srv/private/聚乙烯生产工艺与质量控制概述.docx",
            "locator": "第1章",
            "supporting_text": "背景说明",
        }
    ]
    state["messages"] = [
        AIMessage(
            content=json.dumps(
                {
                    "from": "Intake",
                    "to": "Planner",
                    "title": "聚乙烯质量报告",
                    "sections": [
                        "1. 引言",
                        "2. 工艺分析",
                        "3. 知识库文件清单",
                    ],
                },
                ensure_ascii=False,
            )
        )
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert "## 3. 知识库文件清单" in markdown
    assert "| 文件名称 | 来源类型 | 支撑章节 | 证据条数 |" in markdown
    assert markdown.count("| 聚乙烯生产工艺与质量控制概述.docx | rag |") == 1
    assert "引言、工艺分析" in markdown
    assert "| 2 |" in markdown
    assert "### 证据索引" in markdown
    assert markdown.count("#### 聚乙烯生产工艺与质量控制概述.docx") == 1
    assert "| 证据编号 | 定位 | 支撑章节 | 摘要 |" not in markdown
    assert "**[E1] rag / /srv/private" not in markdown
    assert "**[E1] 第1章**" in markdown
    assert "第1章" in markdown
    assert "/srv/private" not in markdown


def test_reference_list_projects_only_actually_cited_evidence(monkeypatch, tmp_path):
    statuses = {"T1": _status("VERIFIED_PASS"), "T2": _status("VERIFIED_PASS")}
    state = _state(statuses=statuses)
    state["tasks"] = [
        {**state["tasks"][0], "covers_sections": ["1. 引言"]},
        {**state["tasks"][1], "covers_sections": ["2. 工艺分析"]},
    ]
    state["results"] = [
        {
            "task_id": "T1",
            "text_output": "正文使用第一和第三条证据。[E1, 3]。",
            "plan_revision": 1,
            "task_revision": 1,
            "citations": [
                {
                    "evidence_id": "E1",
                    "source_type": "rag",
                    "file_path": "/a/source_a.docx",
                    "supporting_text": "a1",
                },
                {
                    "evidence_id": "E2",
                    "source_type": "rag",
                    "file_path": "/a/source_a.docx",
                    "supporting_text": "a2",
                },
                {
                    "evidence_id": "E3",
                    "source_type": "rag",
                    "file_path": "/b/source_b.pdf",
                    "supporting_text": "b",
                },
                {
                    "evidence_id": "E4",
                    "source_type": "rag",
                    "file_path": "/c/unused.txt",
                    "supporting_text": "unused",
                },
            ],
        },
        {
            "task_id": "T2",
            "text_output": "本节没有证据引用。",
            "plan_revision": 1,
            "task_revision": 1,
            "citations": [
                {
                    "evidence_id": "E1",
                    "source_type": "rag",
                    "file_path": "/d/also_unused.docx",
                    "supporting_text": "unused",
                }
            ],
        },
    ]
    state["messages"] = [
        AIMessage(
            content=json.dumps(
                {
                    "from": "Intake",
                    "to": "Planner",
                    "title": "报告",
                    "sections": ["1. 引言", "2. 工艺分析", "3. 知识库文件清单"],
                },
                ensure_ascii=False,
            )
        )
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert markdown.count("| source_a.docx | rag |") == 1
    assert markdown.count("| source_b.pdf | rag |") == 1
    assert "unused.txt" not in markdown
    assert "also_unused.docx" not in markdown


def test_accepted_missing_figure_removes_dangling_reference_and_caption(
    monkeypatch, tmp_path
):
    statuses = {
        "T1": _status(
            "USER_ACCEPTED_WARNING",
            [{"code": "MISSING_FIGURE", "description": "图形无法生成"}],
        )
    }
    state = _state(
        statuses=statuses,
        results=[
            {
                "task_id": "T1",
                "text_output": (
                    "## 引言\n\n相关因果关系见图1。\n\n另一个说明如图1所示。"
                    "\n\n图1 聚乙烯质量影响关系\n\n其余正文保留。"
                ),
                "plan_revision": 1,
                "task_revision": 1,
                "citations": [],
                "figures": [],
            }
        ],
    )
    state["tasks"] = [{"task_id": "T1", "task_name": "引言", "generate_figure": True}]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert result["report_status"] == "DRAFT_WITH_GAPS"
    assert "见图1" not in markdown
    assert "如图1所示" not in markdown
    assert "图1 聚乙烯质量影响关系" not in markdown
    assert "其余正文保留" in markdown
    assert "图形缺口" in markdown


def test_existing_figure_preserves_reference_caption_and_asset(
    monkeypatch, tmp_path
):
    figure_path = tmp_path / "figure.png"
    PILImage.new("RGB", (8, 8), color="white").save(figure_path)
    state = _state(
        statuses={"T1": _status("VERIFIED_PASS")},
        results=[
            {
                "task_id": "T1",
                "text_output": "## 引言\n\n相关因果关系见图1。\n\n图1 聚乙烯质量影响关系",
                "plan_revision": 1,
                "task_revision": 1,
                "citations": [],
                "figures": [
                    {"path": str(figure_path), "description": "图1 聚乙烯质量影响关系"}
                ],
            }
        ],
    )
    state["tasks"] = [{"task_id": "T1", "task_name": "引言", "generate_figure": True}]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert "见图1" in markdown
    assert "图1 聚乙烯质量影响关系" in markdown
    assert f"]({figure_path})" in markdown
    assert "图形缺口" not in markdown


def test_system_asset_degradation_is_deliverable_with_visible_non_user_warning(
    monkeypatch, tmp_path
):
    state = _state(
        statuses={
            "T1": {
                **_status("ACCEPT_WITH_WARNING"),
                "accepted_by": "system",
                "issues": [{"code": "MISSING_FIGURE"}],
            }
        },
        results=[
            {
                "task_id": "T1",
                "text_output": "正文如图1所示。\n\n图1 可选关系图\n\n保留结论。",
                "plan_revision": 1,
                "task_revision": 1,
                "citations": [],
                "figures": [],
            }
        ],
    )
    state["tasks"] = [{"task_id": "T1", "task_name": "引言", "generate_figure": True}]
    state["degraded_issue_registry"] = [
        {
            "issue_id": "degraded-1",
            "task_id": "T1",
            "task_revision": 1,
            "failure_class": "DEGRADABLE_QUALITY",
            "subtype": "MISSING_FIGURE",
            "reason": "MISSING_FIGURE",
            "affected_claims": [],
            "affected_requirement_ids": [],
            "attempted_repairs": [],
            "final_fallback": "commit_supported_content_with_warning",
            "status": "active",
            "metadata": {},
        }
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert result["report_status"] == "DRAFT_WITH_GAPS"
    assert "系统记录的交付限制" in markdown
    assert "MISSING_FIGURE" in markdown
    assert "用户明确接受" not in markdown
    assert "如图1所示" not in markdown
    assert "图1 可选关系图" not in markdown
    assert "保留结论" in markdown


def test_nested_reference_projection_restores_parent_container(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["tasks"] = [
        {**state["tasks"][0], "covers_sections": ["1. 引言"]},
        {**state["tasks"][1], "covers_sections": ["2. 工艺分析"]},
    ]
    state["messages"] = [
        AIMessage(
            content=json.dumps(
                {
                    "from": "Intake",
                    "to": "Planner",
                    "title": "聚乙烯质量报告",
                    "sections": [
                        "1. 引言",
                        "2. 工艺分析",
                        "5. 附录",
                        "5.1 证据来源",
                    ],
                },
                ensure_ascii=False,
            )
        )
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert markdown.count("## 5. 附录") == 1
    assert markdown.count("### 5.1 证据来源") == 1
    assert markdown.index("## 5. 附录") < markdown.index("### 5.1 证据来源")


def test_report_assembly_restores_container_heading_for_grouped_tasks(
    monkeypatch,
    tmp_path,
):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["tasks"] = [
        {
            "task_id": "T1",
            "task_name": "背景与范围",
            "covers_sections": ["1.1 报告目的"],
        },
        {
            "task_id": "T2",
            "task_name": "编制依据",
            "covers_sections": ["1.2 编制依据"],
        },
    ]
    state["results"] = [
        {
            "task_id": "T1",
            "text_output": "### 1.1 报告目的\n\n目的正文。",
            "plan_revision": 1,
            "task_revision": 1,
        },
        {
            "task_id": "T2",
            "text_output": "### 1.2 编制依据\n\n依据正文。",
            "plan_revision": 1,
            "task_revision": 1,
        },
    ]
    state["messages"] = [
        AIMessage(
            content=json.dumps(
                {
                    "from": "Intake",
                    "to": "Planner",
                    "title": "聚乙烯质量报告",
                    "sections": [
                        "1. 引言",
                        "1.1 报告目的",
                        "1.2 编制依据",
                    ],
                },
                ensure_ascii=False,
            )
        )
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert markdown.count("## 1. 引言") == 1
    assert "## 背景与范围" not in markdown
    assert "## 编制依据" not in markdown
    assert markdown.index("### 1.1 报告目的") < markdown.index("### 1.2 编制依据")


def test_single_covered_top_level_section_keeps_outline_heading(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["tasks"] = [
        {**state["tasks"][0], "covers_sections": ["1. 引言"]},
        {**state["tasks"][1], "covers_sections": ["2. 工艺分析"]},
    ]
    state["messages"] = [
        AIMessage(
            content=json.dumps(
                {
                    "from": "Intake",
                    "to": "Planner",
                    "title": "聚乙烯质量报告",
                    "sections": ["1. 引言", "2. 工艺分析"],
                },
                ensure_ascii=False,
            )
        )
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert markdown.count("## 1. 引言") == 1
    assert markdown.count("## 2. 工艺分析") == 1


def test_report_assembly_restores_full_nested_container_path(monkeypatch, tmp_path):
    state = _state(statuses={"T1": _status("VERIFIED_PASS")})
    state["tasks"] = [
        {
            "task_id": "T1",
            "task_name": "温度机理",
            "covers_sections": ["1.1.1 温度影响"],
        }
    ]
    state["results"] = [
        {
            "task_id": "T1",
            "text_output": "#### 1.1.1 温度影响\n\n正文。",
            "plan_revision": 1,
            "task_revision": 1,
        }
    ]
    state["messages"] = [
        AIMessage(
            content=json.dumps(
                {
                    "from": "Intake",
                    "to": "Planner",
                    "title": "聚乙烯质量报告",
                    "sections": [
                        "1. 参数分析",
                        "1.1 反应条件",
                        "1.1.1 温度影响",
                    ],
                },
                ensure_ascii=False,
            )
        )
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert markdown.index("## 1. 参数分析") < markdown.index("### 1.1 反应条件")
    assert markdown.index("### 1.1 反应条件") < markdown.index("#### 1.1.1 温度影响")


def test_report_assembly_uses_top_level_content_heading_not_task_name(
    monkeypatch,
    tmp_path,
):
    state = _state(statuses={"T1": _status("VERIFIED_PASS")})
    state["tasks"] = [
        {
            "task_id": "T1",
            "task_name": "背景执行单元",
            "covers_sections": ["1. 引言"],
        }
    ]
    state["results"] = [
        {
            "task_id": "T1",
            "text_output": "## 1. 引言\n\n正文。",
            "plan_revision": 1,
            "task_revision": 1,
        }
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert markdown.count("## 1. 引言") == 1
    assert "## 背景执行单元" not in markdown


def test_user_accepted_gap_generates_named_draft_with_visible_warning(
    monkeypatch, tmp_path
):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status(
            "USER_ACCEPTED_GAP",
            [{"code": "EVIDENCE_GAP", "description": "缺少装置级控制范围"}],
        ),
    }
    _install_render_stubs(monkeypatch, tmp_path)

    update = summarizer_v2.summarizer(_state(statuses=statuses), {})

    result = update["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")
    assert result["report_status"] == "DRAFT_WITH_GAPS"
    assert [Path(path).name for path in result["attachments"]] == [
        "report_draft_with_gaps.md",
        "report_draft_with_gaps.pdf",
        "report_draft_with_gaps.docx",
    ]
    assert "未完成草稿：已接受的证据缺口或内容风险" in markdown
    assert "工艺分析" in markdown
    assert "缺少装置级控制范围" in markdown


def test_user_accepted_verifier_contract_error_hides_internal_schema_terms(
    monkeypatch, tmp_path
):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status(
            "USER_ACCEPTED_WARNING",
            [
                {
                    "code": "ASSESSMENT_CONTRACT_ERROR",
                    "description": "自动校验未能完成。",
                }
            ],
        ),
    }
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(_state(statuses=statuses), {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert "自动校验未能完成" in markdown
    assert "malformed" not in markdown.casefold()
    assert "collection fields" not in markdown.casefold()
    assert "json" not in markdown.casefold()


def test_user_accepted_missing_figure_is_omitted_from_draft(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status(
            "USER_ACCEPTED_WARNING",
            [{"code": "MISSING_FIGURE", "description": "要求的因果图未能生成"}],
        ),
    }
    state = _state(statuses=statuses)
    state["tasks"][1]["generate_figure"] = True
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert result["report_status"] == "DRAFT_WITH_GAPS"
    assert "要求的因果图未能生成" in markdown
    assert result["blocking_sections"] == []


def test_user_accepted_missing_table_is_omitted_from_draft(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status(
            "USER_ACCEPTED_WARNING",
            [{"code": "MISSING_TABLE", "description": "要求的表格未能生成"}],
        ),
    }
    state = _state(statuses=statuses)
    state["tasks"][1]["generate_table"] = True
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert result["report_status"] == "DRAFT_WITH_GAPS"
    assert "要求的表格未能生成" in markdown
    assert result["blocking_sections"] == []


def test_user_acceptance_of_unrelated_issue_does_not_waive_missing_figure(
    monkeypatch,
):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status(
            "USER_ACCEPTED_WARNING",
            [{"code": "TOO_LONG", "description": "正文超过篇幅要求"}],
        ),
    }
    state = _state(statuses=statuses)
    state["tasks"][1]["generate_figure"] = True
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("new missing asset must block delivery")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["issues"][0]["code"] == (
        "MISSING_FIGURE_ASSET"
    )


def test_missing_admitted_result_blocks_instead_of_silently_omitting_section(
    monkeypatch, tmp_path
):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(
        statuses=statuses,
        results=[
            {
                "task_id": "T1",
                "text_output": "引言正文",
                "plan_revision": 1,
                "task_revision": 1,
            }
        ],
    )
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("incomplete admitted results must not render")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["task_id"] == "T2"
    assert result["blocking_sections"][0]["issues"][0]["code"] == "MISSING_RESULT"


def test_unknown_section_status_is_reported_as_a_blocker(monkeypatch):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("UNKNOWN_STATUS"),
    }
    state = _state(statuses=statuses)
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unknown acceptance state must not render")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["task_id"] == "T2"
    assert result["blocking_sections"][0]["status"] == "UNKNOWN_STATUS"


def test_revision_mismatch_blocks_stale_verified_result(monkeypatch):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": {**_status("VERIFIED_PASS"), "task_revision": 2},
    }
    state = _state(statuses=statuses)
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("stale result must not render")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["issues"][0]["code"] == "REVISION_MISMATCH"


def test_similar_but_different_leading_heading_is_preserved():
    text = "## 质量异常案例分析\n\n正文"

    assert summarizer_v2._strip_duplicate_leading_heading(
        text, "质量异常原因分析"
    ) == text


def test_missing_figure_file_blocks_delivery(monkeypatch):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["results"][0]["figures"] = [
        {"path": "/missing/causal.png", "description": "因果图"}
    ]
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("missing figure must block delivery")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["issues"][0]["code"] == "MISSING_FIGURE_FILE"


def test_figure_evidence_ids_are_visible_in_report(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    figure_path = tmp_path / "causal.png"
    PILImage.new("RGB", (2, 2), "white").save(figure_path)
    state["results"][0]["figures"] = [
        {
            "path": str(figure_path),
            "description": "工艺参数关系图",
            "evidence_ids": ["E1"],
        }
    ]
    _install_render_stubs(monkeypatch, tmp_path)

    result = summarizer_v2.summarizer(state, {})["final_result"]
    markdown = Path(result["attachments"][0]).read_text(encoding="utf-8")

    assert "关系证据：[E2]" in markdown


def test_corrupt_image_file_blocks_delivery(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    figure_path = tmp_path / "fake.png"
    figure_path.write_bytes(b"not a png")
    state["results"][0]["figures"] = [{"path": str(figure_path)}]
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("corrupt image must block")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["issues"][0]["code"] == "INVALID_FIGURE_CONTENT"


def test_user_accepted_missing_figure_does_not_waive_corrupt_image(
    monkeypatch, tmp_path
):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status(
            "USER_ACCEPTED_WARNING",
            [{"code": "MISSING_FIGURE", "description": "接受缺少图形"}],
        ),
    }
    state = _state(statuses=statuses)
    state["tasks"][1]["generate_figure"] = True
    figure_path = tmp_path / "fake.png"
    figure_path.write_bytes(b"not a png")
    state["results"][0]["figures"] = [{"path": str(figure_path)}]
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("corrupt accepted asset must still block")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["issues"][0]["code"] == (
        "INVALID_FIGURE_CONTENT"
    )


def test_truncated_jpeg_that_passes_verify_is_not_renderable(tmp_path):
    figure_path = tmp_path / "truncated.jpg"
    PILImage.new("RGB", (20, 20), "white").save(figure_path, format="JPEG")
    figure_path.write_bytes(figure_path.read_bytes()[:-1])

    assert summarizer_v2._is_renderable_local_image(str(figure_path)) is False


def test_image_extension_must_match_decoded_format(tmp_path):
    figure_path = tmp_path / "renamed.jpg"
    PILImage.new("RGB", (2, 2), "white").save(figure_path, format="PNG")

    assert summarizer_v2._is_renderable_local_image(str(figure_path)) is False


def test_renderer_failures_are_reported_without_false_success(monkeypatch, tmp_path):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda state, config: str(tmp_path),
    )
    monkeypatch.setattr(
        summarizer_v2.md_to_pdf,
        "md_to_pdf",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pdf failed")),
    )
    monkeypatch.setattr(
        summarizer_v2.md_to_docx,
        "md_to_docx",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("docx failed")),
    )

    result = summarizer_v2.summarizer(_state(statuses=statuses), {})["final_result"]

    assert result["delivery_status"] == "PARTIAL"
    assert [Path(path).name for path in result["attachments"]] == ["report.md"]
    assert Path(result["path"]).name == "report.md"
    assert "部分" in result["summary"]
    assert {error["format"] for error in result["artifact_errors"]} == {"pdf", "docx"}


@pytest.mark.parametrize(
    "figure",
    [
        {"description": "没有路径"},
        {"path": "https://example.com/figure.png", "description": "远程图片"},
        "not-a-figure-object",
    ],
)
def test_unrenderable_figure_asset_blocks_delivery(monkeypatch, figure):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["results"][0]["figures"] = [figure]
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unrenderable figure must block")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["issues"][0]["code"] == "INVALID_FIGURE_ASSET"


def test_unmaterializable_table_asset_blocks_delivery(monkeypatch):
    statuses = {
        "T1": _status("VERIFIED_PASS"),
        "T2": _status("VERIFIED_PASS"),
    }
    state = _state(statuses=statuses)
    state["tasks"][1]["generate_table"] = True
    state["results"][0]["tables"] = [{"title": "空表"}]
    monkeypatch.setattr(
        summarizer_v2,
        "get_session_cache_dir",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("invalid table must block")
        ),
    )

    result = summarizer_v2.summarizer(state, {})["final_result"]

    assert result["report_status"] == "BLOCKED"
    assert result["blocking_sections"][0]["issues"][0]["code"] == "INVALID_TABLE_ASSET"

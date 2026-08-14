import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from PIL import Image as PILImage

from src.nodes import summarizer_v2


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
    assert "| 引言 | [E1]" in markdown
    assert "| 工艺分析 | [E2]" in markdown
    assert "工艺分析正文 [E2]" in markdown
    assert "LLM Generation Failed" not in markdown


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

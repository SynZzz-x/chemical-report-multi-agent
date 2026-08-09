from pathlib import Path

import pytest

from src.nodes import summarizer_v2 as module


def _passed_report_state():
    artifact = {
        "artifact_id": "A1",
        "task_id": "T1",
        "text_output": "已审核正文",
        "content": "已审核正文",
        "figures": [],
        "tables": [],
        "citations": [],
    }
    return {
        "user_id": "u1",
        "job_id": "j1",
        "tasks": [{"task_id": "T1", "task_name": "第一章"}],
        "task_records": {
            "T1": {
                "task_id": "T1",
                "sequence": 0,
                "status": "PASSED",
                "attempt_count": 1,
                "active_artifact_id": "A1",
                "dependencies": [],
            }
        },
        "active_artifact_ids": {"T1": "A1"},
        "artifacts": {"A1": artifact},
        "review_records": [
            {
                "review_id": "R1",
                "task_id": "T1",
                "artifact_id": "A1",
                "reviewer": "quality_review_agent",
                "status": "PASS",
                "issues": [],
            }
        ],
        "results": [
            {
                "task_id": "T1",
                "artifact_id": "stale-A1",
                "text_output": "不应进入报告的旧正文",
            }
        ],
        "messages": [],
    }


def test_report_llm_calls_disable_json_mode(monkeypatch):
    calls = []

    class Model:
        def invoke(self, *args, **kwargs):
            return type("Response", (), {"content": "正文"})()

    monkeypatch.setattr(
        module,
        "get_llm",
        lambda config, json_mode=True: calls.append(json_mode) or Model(),
    )

    module._generate_report_evaluation("report", {})
    module._generate_section_content(
        {
            "title": "章节",
            "text": "正文",
            "figures": [],
            "tables": [],
            "citations": [],
        },
        {},
    )

    assert calls == [False, False]


def test_summarizer_rejects_incomplete_task_ledger():
    state = {
        "tasks": [{"task_id": "T1"}],
        "task_records": {
            "T1": {
                "task_id": "T1",
                "sequence": 0,
                "status": "RUNNING",
                "attempt_count": 1,
                "dependencies": [],
            }
        },
    }

    with pytest.raises(RuntimeError, match="not passed"):
        module.summarizer(state, {})


def test_summarizer_rejects_passed_artifact_without_pass_review():
    state = _passed_report_state()
    state["review_records"] = []

    with pytest.raises(RuntimeError, match="PASS review"):
        module.summarizer(state, {})


def test_failed_pdf_is_not_advertised_as_attachment(monkeypatch, tmp_path):
    state = _passed_report_state()
    monkeypatch.setattr(module, "get_session_cache_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(
        module,
        "_generate_section_content",
        lambda section, config: f"## {section['title']}\n\n{section['text']}",
    )
    monkeypatch.setattr(module, "_generate_report_evaluation", lambda *args: "评价")
    monkeypatch.setattr(module.md_rewrite, "rewrite_markdown", lambda text: text)
    monkeypatch.setattr(
        module.md_to_pdf,
        "md_to_pdf",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("wide table")),
    )
    monkeypatch.setattr(
        module.md_to_docx,
        "md_to_docx",
        lambda content, path: Path(path).write_bytes(b"docx"),
    )

    result = module.summarizer(state, {})

    manifest = result["report_manifest"]
    assert manifest["included_artifact_ids"] == ["A1"]
    assert manifest["pdf_status"] == "FAILED"
    assert manifest["docx_status"] == "SUCCEEDED"
    assert manifest["generation_errors"]["pdf"] == "wide table"
    assert all(
        not path.endswith(".pdf")
        for path in result["final_result"]["attachments"]
    )
    assert "已审核正文" in Path(manifest["md_path"]).read_text(encoding="utf-8")
    assert "不应进入报告的旧正文" not in Path(manifest["md_path"]).read_text(
        encoding="utf-8"
    )


def test_report_manifest_is_persisted_with_successful_outputs_only(
    monkeypatch, tmp_path
):
    state = _passed_report_state()
    state["conversation_id"] = "c1"
    monkeypatch.setattr(module, "get_session_cache_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(
        module,
        "_generate_section_content",
        lambda section, config: f"## {section['title']}\n\n{section['text']}",
    )
    monkeypatch.setattr(module, "_generate_report_evaluation", lambda *args: "评价")
    monkeypatch.setattr(module.md_rewrite, "rewrite_markdown", lambda text: text)
    monkeypatch.setattr(
        module.md_to_pdf,
        "md_to_pdf",
        lambda content, path, **kwargs: Path(path).write_bytes(b"pdf"),
    )
    monkeypatch.setattr(
        module.md_to_docx,
        "md_to_docx",
        lambda content, path: Path(path).write_bytes(b"docx"),
    )

    class Store:
        def __init__(self):
            self.calls = []

        def put(self, namespace, key, value, index=False):
            self.calls.append((namespace, key, value, index))

    store = Store()
    result = module.summarizer(state, {}, store=store)

    manifest = result["report_manifest"]
    assert store.calls == [
        (("u1", "report_jobs", "j1", "reports"), manifest["report_id"], manifest, False)
    ]
    assert set(result["final_result"]["attachments"]) == {
        manifest["md_path"],
        manifest["rewritten_md_path"],
        manifest["pdf_path"],
        manifest["docx_path"],
    }


def test_legacy_passed_results_are_migrated_to_artifacts_and_reviews(
    monkeypatch, tmp_path
):
    state = {
        "user_id": "u1",
        "job_id": "legacy-j1",
        "tasks": [{"task_id": "T1", "task_name": "旧章节"}],
        "results": [{"task_id": "T1", "text_output": "旧版已接受正文"}],
        "messages": [],
    }
    monkeypatch.setattr(module, "get_session_cache_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(
        module,
        "_generate_section_content",
        lambda section, config: f"## {section['title']}\n\n{section['text']}",
    )
    monkeypatch.setattr(module, "_generate_report_evaluation", lambda *args: "评价")
    monkeypatch.setattr(module.md_rewrite, "rewrite_markdown", lambda text: text)
    monkeypatch.setattr(
        module.md_to_pdf,
        "md_to_pdf",
        lambda content, path, **kwargs: Path(path).write_bytes(b"pdf"),
    )
    monkeypatch.setattr(
        module.md_to_docx,
        "md_to_docx",
        lambda content, path: Path(path).write_bytes(b"docx"),
    )

    result = module.summarizer(state, {})

    assert result["report_manifest"]["included_artifact_ids"][0].startswith(
        "artifact_legacy_"
    )
    assert result["active_artifact_ids"]["T1"] in result["artifacts"]
    assert result["review_records"][0]["status"] == "PASS"

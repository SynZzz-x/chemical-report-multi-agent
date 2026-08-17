from pathlib import Path


def test_first_markdown_path_preserves_admitted_artifact_order(tmp_path):
    from src.report_preview import first_markdown_path

    docx = tmp_path / "report.docx"
    draft = tmp_path / "report_draft_with_gaps.md"
    fallback = tmp_path / "report.md"

    assert first_markdown_path([docx, draft, fallback]) == draft


def test_first_markdown_path_returns_none_without_markdown(tmp_path):
    from src.report_preview import first_markdown_path

    assert first_markdown_path([tmp_path / "report.docx", tmp_path / "report.pdf"]) is None


def test_read_markdown_preview_returns_nonempty_utf8_content(tmp_path):
    from src.report_preview import read_markdown_preview

    report = tmp_path / "report.md"
    report.write_text("# 最终报告\n\n正文", encoding="utf-8")

    assert read_markdown_preview(report) == "# 最终报告\n\n正文"


def test_read_markdown_preview_tolerates_missing_or_empty_artifact(tmp_path):
    from src.report_preview import read_markdown_preview

    empty = tmp_path / "empty.md"
    empty.write_text("  \n", encoding="utf-8")

    assert read_markdown_preview(None) is None
    assert read_markdown_preview(tmp_path / "missing.md") is None
    assert read_markdown_preview(empty) is None


def test_streamlit_renders_preview_from_safe_report_paths_before_downloads():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )

    assert "def _final_markdown_path() -> Path | None:" in source
    assert "first_markdown_path(_report_paths_from_state())" in source
    assert "def _render_report_preview() -> None:" in source
    assert "content = read_markdown_preview(_final_markdown_path())" in source
    assert (
        "_render_history()\n_render_report_preview()\n_render_report_downloads()"
        in source
    )

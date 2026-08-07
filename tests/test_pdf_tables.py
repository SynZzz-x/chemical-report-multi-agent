from src.utils.md_to_pdf import (
    _normalize_table_rows,
    _split_markdown_table_row,
    md_to_pdf,
)


def test_escaped_pipe_stays_inside_one_cell():
    row = r"| E1 | 条件 A \| 条件 B | 工艺说明 |"

    assert _split_markdown_table_row(row) == [
        "E1",
        "条件 A | 条件 B",
        "工艺说明",
    ]


def test_rows_are_padded_to_one_column_count():
    assert _normalize_table_rows([["a", "b"], ["c"]]) == [
        ["a", "b"],
        ["c", ""],
    ]


def test_wide_table_generates_pdf_without_negative_available_width(tmp_path):
    header = "|" + "|".join(f"C{i}" for i in range(20)) + "|"
    divider = "|" + "|".join("---" for _ in range(20)) + "|"
    row = "|" + "|".join("value" for _ in range(20)) + "|"
    output = tmp_path / "wide.pdf"

    md_to_pdf("\n".join([header, divider, row]), str(output))

    assert output.is_file()
    assert output.stat().st_size > 0

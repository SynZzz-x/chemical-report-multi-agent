from src.report_validation import (
    count_report_length,
    extract_markdown_tables,
    parse_length_target,
    remove_mermaid_blocks,
)


def test_report_length_ignores_markdown_urls_and_evidence_markers():
    text = "# 标题\n\n聚乙烯 ABC test [E1] https://example.com/source"

    assert count_report_length(text) == 7


def test_length_target_supports_ranges_approximate_and_one_sided_requirements():
    assert parse_length_target("字数：800-1000字。") == {"min": 800, "max": 1000}
    assert parse_length_target("约1200字。") == {"min": 1080, "max": 1320}
    assert parse_length_target("不少于500字。") == {"min": 500, "max": None}
    assert parse_length_target("不超过300字。") == {"min": None, "max": 300}


def test_markdown_table_is_materialized_without_removing_source_markdown():
    content = """分析如下：

| 参数 | 影响 | 证据 |
| --- | :---: | --- |
| 温度 | 熔融指数升高 | [E1] |
| 压力 | 当前证据不足 |  |
"""

    tables = extract_markdown_tables(content)

    assert len(tables) == 1
    assert tables[0]["type"] == "markdown"
    assert tables[0]["headers"] == ["参数", "影响", "证据"]
    assert tables[0]["rows"][0] == ["温度", "熔融指数升高", "[E1]"]
    assert tables[0]["evidence_refs"] == ["E1"]
    assert "| 温度 |" in content


def test_invalid_pipe_text_is_not_misclassified_as_a_markdown_table():
    assert extract_markdown_tables("A | B\n这只是普通文本") == []


def test_mermaid_blocks_are_removed_but_other_code_blocks_are_preserved():
    content = """正文。

```mermaid
graph TD
A --> B
```

```python
print("keep")
```
"""

    cleaned = remove_mermaid_blocks(content)

    assert "graph TD" not in cleaned
    assert 'print("keep")' in cleaned

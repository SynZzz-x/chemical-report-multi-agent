from src.nodes.intake import extract_initial_request


def test_extract_initial_request_merges_user_text_and_docs():
    messages = [
        {
            "role": "user",
            "content": "分析 ETTh2 数据",
            "resources": [{"name": "a.csv", "path": "/tmp/a.csv", "type": "text/csv"}],
        },
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "输出 Word 报告"},
    ]
    docs = [{"name": "b.csv", "path": "/tmp/b.csv", "type": "text/csv"}]

    result = extract_initial_request(messages, docs)

    assert result["raw_request"] == "分析 ETTh2 数据\n输出 Word 报告"
    assert result["resources"] == [
        {"name": "a.csv", "path": "/tmp/a.csv", "type": "text/csv", "resource_id": None},
        {"name": "b.csv", "path": "/tmp/b.csv", "type": "text/csv", "resource_id": None},
    ]


def test_extract_initial_request_deduplicates_resources_by_name_and_path():
    resource = {"name": "same.csv", "path": "/tmp/same.csv", "type": "text/csv"}

    result = extract_initial_request(
        [{"role": "user", "content": "分析", "resources": [resource]}],
        [resource],
    )

    assert len(result["resources"]) == 1

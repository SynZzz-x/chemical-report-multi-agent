from src.nodes.planner import _ensure_use_resources_paths


def test_ensure_use_resources_paths_maps_resource_names_to_paths():
    tasks = [
        {"task_id": "T1", "task_name": "读取数据", "use_resources": ["metrics.csv"]},
        {"task_id": "T2", "task_name": "保留外部 URL", "use_resources": ["https://example.com/data"]},
    ]
    resources = [{"name": "metrics.csv", "path": "/cache/uploads/metrics.csv", "type": "text/csv"}]

    result = _ensure_use_resources_paths(tasks, resources)

    assert result[0]["use_resources"] == ["/cache/uploads/metrics.csv"]
    assert result[0]["use_rag"] is False
    assert result[0]["task_type"] == "analysis"
    assert result[0]["query"] == ""
    assert result[1]["use_resources"] == ["https://example.com/data"]

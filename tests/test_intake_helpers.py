import json
from types import SimpleNamespace

from src.nodes import intake as intake_module
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


def test_canonical_intake_prompt_is_compact_and_keeps_contract(monkeypatch):
    raw_request = "请分析聚乙烯质量异常，给出排查建议。"
    response = {
        "is_chat": False,
        "user_intent": "排查聚乙烯质量异常并提出建议。",
        "task_type": "工程分析报告",
        "title": "聚乙烯质量异常排查",
        "doc_length": "不限",
        "constraints": ["仅使用离线知识库"],
        "style": "formal",
        "output_format": "Markdown",
        "web_authorized": False,
        "sections": ["聚乙烯质量异常排查"],
        "core_content": ["熔融指数", "灰分", "凝胶含量"],
    }
    calls = []

    class Prompt:
        def __init__(self, templates):
            self.templates = templates

        @classmethod
        def from_messages(cls, templates):
            return cls(templates)

        def format_messages(self, **values):
            return [
                SimpleNamespace(content=template.format(**values))
                for _role, template in self.templates
            ]

    class Recorder:
        def invoke(self, messages, **_kwargs):
            calls.append(messages)
            return SimpleNamespace(content=json.dumps(response, ensure_ascii=False))

    recorder = Recorder()
    monkeypatch.setattr(intake_module, "ChatPromptTemplate", Prompt)
    monkeypatch.setattr(intake_module, "get_llm", lambda *_args, **_kwargs: recorder)
    monkeypatch.setattr(
        intake_module,
        "with_completion_budget",
        lambda model, _purpose: (model, 1),
    )
    monkeypatch.setattr(
        intake_module,
        "invoke_llm",
        lambda model, messages, **kwargs: model.invoke(messages, **kwargs),
    )

    parsed = intake_module.llm_parse_user_need(raw_request, {"configurable": {}})

    assert parsed["title"] == response["title"]
    assert len(calls) == 1
    messages = calls[0]
    assert len(messages) == 2
    assert messages[1].content == raw_request
    serialized = "\n".join(message.content for message in messages)
    assert len(serialized) < 1800
    for field in (
        "is_chat",
        "from",
        "to",
        "type",
        "user_intent",
        "task_type",
        "title",
        "doc_length",
        "constraints",
        "style",
        "output_format",
        "web_authorized",
        "sections",
        "core_content",
        "response",
    ):
        assert field in serialized

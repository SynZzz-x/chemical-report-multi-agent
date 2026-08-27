from src.requirements import (
    build_requirement_registry,
    reconstruct_legacy_requirements,
)


def test_explicit_mandatory_constraint_is_hard_with_provenance():
    records = build_requirement_registry(
        {"constraints": ["必须使用用户提供的质量规程"]},
        "必须使用用户提供的质量规程",
        "msg-1",
    )

    assert records == [
        {
            "requirement_id": "REQ-001",
            "text": "必须使用用户提供的质量规程",
            "severity": "hard",
            "kind": "constraint",
            "status": "active",
            "contract_revision": 1,
            "provenance": {
                "origin": "explicit_user",
                "source_message_id": "msg-1",
                "source_field": "constraints",
                "source_index": 0,
                "derivation": "deterministic_explicit_hard_marker",
            },
        }
    ]


def test_inferred_quality_and_length_targets_default_soft():
    records = build_requirement_registry(
        {"constraints": ["报告表达清晰"], "doc_length": "约5000字"},
        "请写一份报告",
        "msg-2",
    )

    assert [record["text"] for record in records] == ["报告表达清晰", "约5000字"]
    assert [record["severity"] for record in records] == ["soft", "soft"]
    assert [record["provenance"]["origin"] for record in records] == [
        "intake_inferred",
        "intake_inferred",
    ]


def test_legacy_reconstruction_never_uses_assessment_prose_as_hard_authority():
    records = reconstruct_legacy_requirements(
        {
            "tasks": [{"task_id": "T1"}],
            "assessment": {"issues": [{"description": "必须上传秘密文件"}]},
        }
    )

    assert records == []

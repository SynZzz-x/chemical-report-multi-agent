from copy import deepcopy

from src.evidence.identity import canonical_citation_identity
from src.evidence.integrity import validate_pre_remap_citation_integrity


def citation(path: str, *, evidence_id: str = "E8", file_id: str = "") -> dict:
    return {
        "evidence_id": evidence_id,
        "source_type": "rag",
        "file_id": file_id,
        "file_path": path,
        "locator": "section 5",
        "chunk_ids": ["chunk-5"],
        "supporting_text": "温度影响分子量。",
        "supporting_text_excerpt": "展示摘要",
    }


def test_identity_prefers_stable_source_id_over_cache_path():
    left = citation("/job-a/cache/process.docx", file_id="doc-42")
    right = citation("/job-b/cache/process.docx", file_id="doc-42")
    assert canonical_citation_identity(left) == canonical_citation_identity(right)


def test_identity_does_not_merge_distinct_same_basename_files():
    assert canonical_citation_identity(citation("/a/process.docx")) != canonical_citation_identity(
        citation("/b/process.docx")
    )


def test_identity_ignores_presentation_excerpt():
    left = citation("/a/process.docx")
    right = deepcopy(left)
    right["supporting_text_excerpt"] = "另一展示摘要"
    assert canonical_citation_identity(left) == canonical_citation_identity(right)


def test_identity_uses_full_raw_text_as_last_evidence_fallback():
    left = citation("/a/process.docx")
    right = deepcopy(left)
    left["chunk_ids"] = []
    right["chunk_ids"] = []
    right["supporting_text"] = "温度不影响分子量。"
    assert canonical_citation_identity(left) != canonical_citation_identity(right)


def test_identity_canonicalizes_url_host_and_fragment_only():
    left = {
        **citation(""),
        "source_type": "web",
        "file_path": "",
        "url": "HTTPS://Example.COM/report?id=7#page-1",
    }
    right = {**left, "url": "https://example.com/report?id=7#page-9"}
    other = {**left, "url": "https://example.com/other?id=7"}
    assert canonical_citation_identity(left) == canonical_citation_identity(right)
    assert canonical_citation_identity(left) != canonical_citation_identity(other)


def test_pre_remap_rejects_same_task_local_id_with_two_identities():
    sections = [
        {
            "task_id": "T1",
            "text_output": "工艺结论 [E8]。",
            "citations": [
                citation("/docs/polyethylene-process.docx"),
                {
                    **citation("/docs/heat-exchanger-guide.docx"),
                    "locator": "section 2.1",
                    "chunk_ids": ["maintenance-2"],
                    "supporting_text": "换热器需要清洗。",
                },
            ],
        }
    ]
    result = validate_pre_remap_citation_integrity(sections)
    assert result.is_valid is False
    assert result.issues[0].code == "LOCAL_CITATION_IDENTITY_CONFLICT"
    assert result.issues[0].task_id == "T1"
    assert result.issues[0].evidence_id == "E8"


def test_pre_remap_allows_exact_duplicate_records():
    item = citation("/docs/process.docx")
    result = validate_pre_remap_citation_integrity(
        [
            {
                "task_id": "T1",
                "text_output": "工艺 [E8]。",
                "citations": [item, deepcopy(item)],
            }
        ]
    )
    assert result.is_valid is True


def test_pre_remap_scopes_same_local_id_by_task():
    sections = [
        {
            "task_id": "T1",
            "text_output": "工艺 [E8]。",
            "citations": [citation("/docs/process.docx")],
        },
        {
            "task_id": "T2",
            "text_output": "维护 [E8]。",
            "citations": [citation("/docs/maintenance.docx")],
        },
    ]
    assert validate_pre_remap_citation_integrity(sections).is_valid is True

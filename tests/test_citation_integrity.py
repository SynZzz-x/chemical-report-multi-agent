from copy import deepcopy

import pytest

from src.evidence.identity import canonical_citation_identity, normalize_sections_evidence
from src.evidence.integrity import (
    project_lossless_used_citations,
    validate_final_citation_integrity,
    validate_pre_remap_citation_integrity,
)


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


def test_identity_rejects_basename_like_explicit_identity_when_path_is_available():
    left = {**citation("/a/process.docx"), "source_identity": "process.docx"}
    right = {**citation("/b/process.docx"), "source_identity": "process.docx"}
    assert canonical_citation_identity(left) != canonical_citation_identity(right)


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


@pytest.mark.parametrize("body_field", ["text", "content", "text_output"])
def test_pre_remap_rejects_unknown_raw_marker_before_used_id_collision(body_field):
    sections = [{
        "task_id": "T1",
        body_field: "已绑定结论 [E8]。未知结论 [E1]。",
        "citations": [citation("/docs/a.docx", evidence_id="E8")],
    }]

    result = validate_pre_remap_citation_integrity(sections)

    assert not result.is_valid
    assert [(issue.code, issue.task_id, issue.evidence_id) for issue in result.issues] == [
        ("LOCAL_CITATION_BINDING_MISSING", "T1", "E1")
    ]


def test_pre_remap_rejects_unknown_raw_marker_despite_other_section_binding():
    sections = [
        {"task_id": "T0", "text": "前节结论 [E1]。",
         "citations": [citation("/docs/previous.docx", evidence_id="E1")]},
        {"task_id": "T1", "text": "已绑定结论 [E8]。未知结论 [E1]。",
         "citations": [citation("/docs/a.docx", evidence_id="E8")]},
    ]

    result = validate_pre_remap_citation_integrity(sections)

    assert not result.is_valid
    assert [(issue.code, issue.task_id, issue.evidence_id) for issue in result.issues] == [
        ("LOCAL_CITATION_BINDING_MISSING", "T1", "E1")
    ]


def test_pre_remap_preserves_original_scope_of_inherited_bindings():
    raw = [
        {"task_id": task_id, "text": "正文 [E8]。", "citations": [citation(path)]}
        for task_id, path in (("T1", "/docs/a.docx"), ("T2", "/docs/b.docx"))
    ]
    normalized, _ = normalize_sections_evidence(raw)
    synthesis = {
        "task_id": "TS",
        "text": "正文 [E1, E2]。",
        "citations": [item for section in normalized for item in section["citations"]],
    }

    assert validate_pre_remap_citation_integrity([*raw, synthesis]).is_valid


@pytest.mark.parametrize("reverse", [False, True])
def test_pre_remap_rejects_visible_id_conflict_despite_distinct_inherited_keys(reverse):
    citations = [
        {**citation("/docs/a.docx", evidence_id="E1"),
         "evidence_key": "T1:E8", "local_evidence_id": "E8"},
        {**citation("/docs/b.docx", evidence_id="E1"),
         "evidence_key": "T2:E9", "local_evidence_id": "E9"},
    ]
    if reverse:
        citations.reverse()
    result = validate_pre_remap_citation_integrity(
        [{"task_id": "TS", "text": "正文 [E1]。", "citations": citations}]
    )

    assert not result.is_valid
    assert any(issue.task_id == "TS" and issue.evidence_id == "E1" for issue in result.issues)


def test_pre_remap_rejects_inherited_key_conflict_across_current_sections():
    sections = [
        {"task_id": task_id, "text": f"正文 [{visible_id}]。", "citations": [
            {**citation(path, evidence_id=visible_id),
             "evidence_key": "T1:E8", "local_evidence_id": "E8"},
        ]}
        for task_id, visible_id, path in (
            ("T1", "E1", "/docs/a.docx"), ("TS", "E2", "/docs/b.docx")
        )
    ]

    assert not validate_pre_remap_citation_integrity(sections).is_valid


def test_unused_raw_ids_do_not_alias_allocated_display_ids_or_mutate_sources():
    raw = [{
        "task_id": "T1", "text": "正文 [E1, E3]。",
        "citations": [citation(f"/docs/{index}.docx", evidence_id=f"E{index}")
                      for index in (1, 2, 3)],
    }]
    original = deepcopy(raw)
    assert validate_pre_remap_citation_integrity(raw).is_valid

    normalized, display_map = normalize_sections_evidence(raw)
    projected = project_lossless_used_citations(normalized)

    assert display_map == {"T1:E1": "E1", "T1:E3": "E2"}
    assert [item["file_path"] for item in projected] == ["/docs/1.docx", "/docs/3.docx"]
    assert len(normalized[0]["citations"]) == 3
    assert raw == original
    assert normalize_sections_evidence(raw) == (normalized, display_map)


def test_unused_allocation_cannot_bind_an_unknown_body_marker():
    raw = [{
        "task_id": "T1", "text": "正文 [E1, E2]。",
        "citations": [citation("/docs/a.docx", evidence_id="E1"),
                      citation("/docs/unused.docx", evidence_id="E9")],
    }]
    normalized, _ = normalize_sections_evidence(raw)

    assert [item["file_path"] for item in project_lossless_used_citations(normalized)] == ["/docs/a.docx"]


def test_pre_remap_allows_exact_inherited_duplicates():
    item = {**citation("/docs/a.docx", evidence_id="E1"),
            "evidence_key": "T1:E8", "local_evidence_id": "E8"}

    assert validate_pre_remap_citation_integrity([
        {"task_id": "TS", "text": "正文 [E1]。", "citations": [item, deepcopy(item)]}
    ]).is_valid


@pytest.mark.parametrize("with_prior_sections", [False, True])
def test_same_visible_canonical_duplicate_preserves_all_inherited_keys(with_prior_sections):
    items = [
        {**citation("/docs/a.docx", evidence_id="E1"), "evidence_key": key,
         "local_evidence_id": local_id}
        for key, local_id in (("T1:E8", "E8"), ("T2:E9", "E9"))
    ]
    raw = [{"task_id": "TS", "text": "正文 [E1]。", "citations": items}]
    if with_prior_sections:
        raw = [
            {"task_id": task_id, "text": f"正文 [{local_id}]。", "citations": [
                citation("/docs/a.docx", evidence_id=local_id)]}
            for task_id, local_id in (("T1", "E8"), ("T2", "E9"))
        ] + raw

    assert validate_pre_remap_citation_integrity(raw).is_valid
    normalized, display_map = normalize_sections_evidence(raw)

    assert display_map == {"T1:E8": "E1", "T2:E9": "E1"}
    assert normalized[-1]["text"] == "正文 [E1]。"
    assert [item["evidence_key"] for item in normalized[-1]["citations"]] == ["T1:E8", "T2:E9"]
    assert [item["evidence_id"] for item in normalized[-1]["citations"]] == ["E1", "E1"]
    assert len(project_lossless_used_citations(normalized)) == sum(len(section["citations"]) for section in raw)


def test_pre_remap_checks_ambiguous_records_even_when_uncited():
    raw = [{
        "task_id": "T1", "text": "正文 [E1]。",
        "citations": [citation("/docs/a.docx", evidence_id="E1"),
                      citation("/docs/b.docx", evidence_id="E9"),
                      citation("/docs/c.docx", evidence_id="E9")],
    }]

    assert not validate_pre_remap_citation_integrity(raw).is_valid


def test_final_gate_rejects_one_display_id_with_two_identities():
    sections = [
        {
            "task_id": "T1",
            "text_output": "工艺 [E1]。",
            "citations": [citation("/docs/process.docx", evidence_id="E1")],
        },
        {
            "task_id": "T2",
            "text_output": "维护 [E1]。",
            "citations": [citation("/docs/maintenance.docx", evidence_id="E1")],
        },
    ]
    registry = [item for section in sections for item in section["citations"]]

    result = validate_final_citation_integrity(
        sections, "工艺 [E1]。\n维护 [E1]。", registry, body_spans=[(0, 17)]
    )

    assert result.is_valid is False
    assert {issue.code for issue in result.issues} == {
        "FINAL_DISPLAY_IDENTITY_CONFLICT"
    }


def test_final_gate_rejects_unbound_body_marker():
    sections = [{"task_id": "T1", "text_output": "工艺 [E9]。", "citations": []}]

    result = validate_final_citation_integrity(sections, "工艺 [E9]。", [], body_spans=[(0, 8)])

    assert result.is_valid is False
    assert "FINAL_CITATION_BINDING_MISSING" in {
        issue.code for issue in result.issues
    }


def test_final_gate_allows_exact_duplicate_resolution():
    item = citation("/docs/process.docx", evidence_id="E1")
    sections = [
        {
            "task_id": "T1",
            "text_output": "工艺 [E1]。",
            "citations": [item, deepcopy(item)],
        }
    ]

    result = validate_final_citation_integrity(
        sections, "工艺 [E1]。\n\n[E1]", [item, deepcopy(item)], body_spans=[(0, 8)]
    )

    assert result.is_valid is True


def test_appendix_marker_cannot_mask_missing_body_binding():
    sections = [{"task_id": "T1", "text_output": "正文 [E9]。", "citations": []}]

    result = validate_final_citation_integrity(
        sections, "正文 [E9]。\n\n证据附录 [E9]", [], body_spans=[(0, 8)]
    )

    assert "FINAL_CITATION_BINDING_MISSING" in {
        issue.code for issue in result.issues
    }


def test_lossless_projection_retains_conflicting_entries():
    left = citation("/docs/a.docx", evidence_id="E1")
    right = citation("/docs/b.docx", evidence_id="E1")

    projected = project_lossless_used_citations(
        [
            {
                "task_id": "T1",
                "text_output": "正文 [E1]。",
                "citations": [left, right],
            }
        ]
    )

    assert projected == [left, right]


def test_final_gate_rejects_registry_id_unused_by_body():
    sections = [
        {
            "task_id": "T1",
            "text_output": "正文 [E1]。",
            "citations": [citation("/docs/a.docx", evidence_id="E1")],
        }
    ]
    registry = [
        citation("/docs/a.docx", evidence_id="E1"),
        citation("/docs/b.docx", evidence_id="E2"),
    ]

    result = validate_final_citation_integrity(sections, "正文 [E1]。", registry, body_spans=[(0, 8)])

    assert "FINAL_REGISTRY_ID_UNUSED" in {issue.code for issue in result.issues}


def test_final_gate_rejects_task_local_alias_after_remap():
    sections = [
        {
            "task_id": "T1",
            "text_output": "正文 [E8]。",
            "citations": [
                {
                    **citation("/docs/a.docx", evidence_id="E1"),
                    "local_evidence_id": "E8",
                }
            ],
        }
    ]
    registry = sections[0]["citations"]

    result = validate_final_citation_integrity(sections, "正文 [E8]。", registry, body_spans=[(0, 8)])

    assert "FINAL_REMAP_ALIAS" in {issue.code for issue in result.issues}


def test_final_gate_allows_display_id_that_matches_another_local_id():
    raw_sections = [
        {
            "task_id": "T1",
            "text_output": "工艺 [E1]。",
            "citations": [citation("/docs/process.docx", evidence_id="E1")],
        },
        {
            "task_id": "T2",
            "text_output": "维护 [E1, E2]。",
            "citations": [
                citation("/docs/maintenance.docx", evidence_id="E1"),
                citation("/docs/inspection.docx", evidence_id="E2"),
            ],
        },
    ]
    sections, _ = normalize_sections_evidence(raw_sections)
    registry = project_lossless_used_citations(sections)

    result = validate_final_citation_integrity(
        sections, "工艺 [E1]。\n维护 [E2, E3]。", registry, body_spans=[(0, 21)]
    )

    assert sections[0]["text_output"] == "工艺 [E1]。"
    assert sections[1]["text_output"] == "维护 [E2, E3]。"
    assert result.is_valid is True


def test_final_gate_accepts_grouped_markers():
    registry = [
        citation("/docs/a.docx", evidence_id="E1"),
        citation("/docs/b.docx", evidence_id="E2"),
        citation("/docs/c.docx", evidence_id="E3"),
    ]
    sections = [
        {
            "task_id": "T1",
            "text_output": "正文 [E1, 2；E3]。",
            "citations": registry,
        }
    ]

    markdown = "正文 [E1, 2；E3]。"
    result = validate_final_citation_integrity(sections, markdown, registry, body_spans=[(0, len(markdown))])

    assert result.is_valid is True


def test_final_body_slice_excludes_appendix_when_body_marker_was_lost():
    item = citation("/docs/a.docx", evidence_id="E1")
    sections = [{"task_id": "T1", "text": "A[E1]", "citations": [item]}]

    result = validate_final_citation_integrity(
        sections, "A\n\narbitrary appendix [E1]", [item], body_spans=[(0, 1)]
    )

    assert "FINAL_MARKDOWN_CITATION_MISSING" in {issue.code for issue in result.issues}


def test_final_body_slice_detects_introduced_marker_present_elsewhere():
    items = [citation("/docs/a.docx", evidence_id="E1"), citation("/docs/b.docx", evidence_id="E2")]
    sections = [{"task_id": "T1", "text": "A[E1]", "citations": items}]
    markdown = "A[E1, E2]\n\nappendix [E1]"

    result = validate_final_citation_integrity(
        sections, markdown, [items[0]], body_spans=[(0, len("A[E1, E2]"))]
    )

    assert "FINAL_MARKDOWN_CITATION_INTRODUCED" in {issue.code for issue in result.issues}


@pytest.mark.parametrize("body_spans", [[(-1, 3)], [(0, 99)], [(0, 3), (2, 4)]])
def test_final_body_slice_rejects_invalid_or_overlapping_boundaries(body_spans):
    result = validate_final_citation_integrity([], "正文。\n", [], body_spans=body_spans)

    assert "FINAL_BODY_SPANS_INVALID" in {issue.code for issue in result.issues}

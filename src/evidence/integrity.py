"""Lossless validation for raw citations before display-ID remapping."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .citations import extract_inline_evidence_ids
from .identity import canonical_citation_identity


@dataclass(frozen=True)
class CitationIntegrityIssue:
    code: str
    description: str
    task_id: str = ""
    evidence_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CitationIntegrityValidation:
    issues: tuple[CitationIntegrityIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_pre_remap_citation_integrity(
    sections: Sequence[Mapping[str, Any]],
) -> CitationIntegrityValidation:
    """Reject task-local citation IDs that identify more than one raw record."""

    identities_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        task_id = str(section.get("task_id") or "").strip()
        for citation in section.get("citations") or ():
            if not isinstance(citation, Mapping):
                continue
            evidence_id = str(
                citation.get("local_evidence_id") or citation.get("evidence_id") or ""
            ).strip().upper()
            identities_by_key[(task_id, evidence_id)].add(
                canonical_citation_identity(citation)
            )

    issues = tuple(
        CitationIntegrityIssue(
            code="LOCAL_CITATION_IDENTITY_CONFLICT",
            description="同一任务中的本地证据编号对应多个不同证据来源。",
            task_id=task_id,
            evidence_id=evidence_id,
        )
        for (task_id, evidence_id), identities in sorted(identities_by_key.items())
        if len(identities) > 1
    )
    return CitationIntegrityValidation(issues=issues)


def _section_body(section: Mapping[str, Any]) -> str:
    return str(
        section.get("text")
        or section.get("content")
        or section.get("text_output")
        or ""
    )


def _citation_display_id(citation: Mapping[str, Any]) -> str:
    return str(citation.get("evidence_id") or "").strip().upper()


def project_lossless_used_citations(
    sections: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return every normalized citation entry referenced by a section body."""

    projected: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        used_ids = extract_inline_evidence_ids(_section_body(section))
        for citation in section.get("citations") or ():
            if not isinstance(citation, Mapping):
                continue
            if _citation_display_id(citation) in used_ids:
                projected.append(dict(citation))
    return projected


def validate_final_citation_integrity(
    normalized_sections: Sequence[Mapping[str, Any]],
    final_markdown: str,
    lossless_final_citations: Sequence[Mapping[str, Any]],
) -> CitationIntegrityValidation:
    """Validate final display IDs before any report artifact can be created."""

    body_ids: set[str] = set()
    task_by_body_id: dict[str, str] = {}
    issues: list[CitationIntegrityIssue] = []
    for section in normalized_sections:
        if not isinstance(section, Mapping):
            continue
        task_id = str(section.get("task_id") or "").strip()
        section_body_ids = extract_inline_evidence_ids(_section_body(section))
        for evidence_id in section_body_ids:
            body_ids.add(evidence_id)
            task_by_body_id.setdefault(evidence_id, task_id)
        section_citations = [
            citation
            for citation in section.get("citations") or ()
            if isinstance(citation, Mapping)
        ]
        section_display_ids = {
            _citation_display_id(citation) for citation in section_citations
        }
        for citation in section_citations:
            display_id = _citation_display_id(citation)
            local_id = str(citation.get("local_evidence_id") or "").strip().upper()
            if (
                local_id
                and local_id != display_id
                and local_id in section_body_ids
                and local_id not in section_display_ids
            ):
                issues.append(
                    CitationIntegrityIssue(
                        code="FINAL_REMAP_ALIAS",
                        description="重映射后的正文仍使用任务本地证据编号。",
                        task_id=task_id,
                        evidence_id=local_id,
                    )
                )

    identities_by_display_id: dict[str, set[str]] = defaultdict(set)
    registry_ids: set[str] = set()
    for citation in lossless_final_citations:
        if not isinstance(citation, Mapping):
            continue
        display_id = _citation_display_id(citation)
        if not display_id:
            continue
        registry_ids.add(display_id)
        identities_by_display_id[display_id].add(canonical_citation_identity(citation))

    for display_id, identities in sorted(identities_by_display_id.items()):
        if len(identities) > 1:
            issues.append(
                CitationIntegrityIssue(
                    code="FINAL_DISPLAY_IDENTITY_CONFLICT",
                    description="同一最终证据编号对应多个不同证据来源或定位。",
                    evidence_id=display_id,
                )
            )
    for evidence_id in sorted(body_ids - registry_ids):
        issues.append(
            CitationIntegrityIssue(
                code="FINAL_CITATION_BINDING_MISSING",
                description="正文证据编号没有唯一的最终证据绑定。",
                task_id=task_by_body_id.get(evidence_id, ""),
                evidence_id=evidence_id,
            )
        )
    for evidence_id in sorted(registry_ids - body_ids):
        issues.append(
            CitationIntegrityIssue(
                code="FINAL_REGISTRY_ID_UNUSED",
                description="最终证据注册表包含未被正文使用的证据编号。",
                evidence_id=evidence_id,
            )
        )

    final_markdown_ids = extract_inline_evidence_ids(final_markdown)
    for evidence_id in sorted(body_ids - final_markdown_ids):
        issues.append(
            CitationIntegrityIssue(
                code="FINAL_MARKDOWN_CITATION_MISSING",
                description="最终 Markdown 丢失正文中的证据编号。",
                task_id=task_by_body_id.get(evidence_id, ""),
                evidence_id=evidence_id,
            )
        )
    for evidence_id in sorted(final_markdown_ids - body_ids):
        issues.append(
            CitationIntegrityIssue(
                code="FINAL_MARKDOWN_CITATION_INTRODUCED",
                description="最终 Markdown 引入了正文未使用的证据编号。",
                evidence_id=evidence_id,
            )
        )

    return CitationIntegrityValidation(
        issues=tuple(
            sorted(
                issues,
                key=lambda issue: (
                    issue.code,
                    issue.task_id,
                    issue.evidence_id,
                    issue.description,
                ),
            )
        )
    )

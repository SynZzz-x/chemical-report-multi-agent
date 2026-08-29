"""Lossless validation for raw citations before display-ID remapping."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

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

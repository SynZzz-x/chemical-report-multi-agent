from __future__ import annotations

from collections.abc import Iterable

from .models import CoverageReport, EvidenceBundle


CONCEPT_ALIASES: dict[str, frozenset[str]] = {
    "聚合温度": frozenset({"聚合温度", "反应温度", "聚合反应温度"}),
    "聚合压力": frozenset({"聚合压力", "反应压力", "聚合反应压力"}),
}


def _normalise_term(value: str) -> str:
    return "".join(value.casefold().split())


def _aliases_for(concept: str) -> tuple[str, ...]:
    normalized = _normalise_term(concept)
    for canonical, aliases in CONCEPT_ALIASES.items():
        group = frozenset({canonical, *aliases})
        if normalized in {_normalise_term(value) for value in group}:
            return tuple(_normalise_term(value) for value in group)
    return (normalized,)


def assess_coverage(bundle: EvidenceBundle, required_concepts: Iterable[str]) -> CoverageReport:
    concepts = tuple(dict.fromkeys(str(value).strip() for value in required_concepts if str(value).strip()))
    corpus = _normalise_term("\n".join(
        f"{record.title}\n{record.locator}\n{record.supporting_text}" for record in bundle.records
    ))
    covered = tuple(
        concept
        for concept in concepts
        if any(alias in corpus for alias in _aliases_for(concept))
    )
    uncovered = tuple(concept for concept in concepts if concept not in covered)
    if not bundle.records:
        status = "unavailable"
    elif uncovered:
        status = "insufficient"
    else:
        status = "sufficient"
    return CoverageReport(
        status=status,
        covered_concepts=covered,
        uncovered_concepts=uncovered,
        web_fallback_required=bool(uncovered) or not bundle.records,
    )

from __future__ import annotations

from collections.abc import Iterable

from .models import CoverageReport, EvidenceBundle


def assess_coverage(bundle: EvidenceBundle, required_concepts: Iterable[str]) -> CoverageReport:
    concepts = tuple(dict.fromkeys(str(value).strip() for value in required_concepts if str(value).strip()))
    corpus = "\n".join(
        f"{record.title}\n{record.locator}\n{record.supporting_text}" for record in bundle.records
    ).casefold()
    covered = tuple(concept for concept in concepts if concept.casefold() in corpus)
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

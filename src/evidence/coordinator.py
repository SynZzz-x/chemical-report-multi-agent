from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from .coverage import assess_coverage
from .models import CoverageReport, EvidenceBundle, EvidenceRecord
from .normalizer import merge_evidence_bundles


class WebEvidenceProvider(Protocol):
    def search(self, queries: Sequence[str]) -> Iterable[EvidenceRecord]: ...


class EvidenceCoordinator:
    def __init__(self, web_provider: WebEvidenceProvider | None = None, max_web_queries: int = 3):
        self.web_provider = web_provider
        self.max_web_queries = max(0, max_web_queries)

    def complete(
        self,
        bundle: EvidenceBundle,
        *,
        required_concepts: Sequence[str],
        web_queries: Sequence[str],
        allow_web_fallback: bool,
    ) -> tuple[EvidenceBundle, CoverageReport]:
        coverage = assess_coverage(bundle, required_concepts)
        if coverage.status == "sufficient" or not allow_web_fallback or self.web_provider is None:
            return bundle, coverage

        uncovered = tuple(value.casefold() for value in coverage.uncovered_concepts)
        selected: list[str] = []
        for raw_query in web_queries:
            query = str(raw_query).strip()
            if not query or query in selected:
                continue
            if uncovered and not any(concept in query.casefold() for concept in uncovered):
                continue
            selected.append(query)
            if len(selected) >= self.max_web_queries:
                break
        if not selected:
            selected = list(dict.fromkeys(coverage.uncovered_concepts))[: self.max_web_queries]
        if not selected:
            return bundle, coverage

        web_records = tuple(self.web_provider.search(selected))
        completed = merge_evidence_bundles(bundle, EvidenceBundle(records=web_records))
        return completed, assess_coverage(completed, required_concepts)

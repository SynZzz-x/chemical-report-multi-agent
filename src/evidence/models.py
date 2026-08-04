from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceRecord(BaseModel):
    """One attributable statement used by report text or a graph relation."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    source_type: Literal["rag", "web"]
    title: str
    supporting_text: str
    locator: str = ""
    file_path: str = ""
    url: str = ""
    chunk_ids: tuple[str, ...] = ()
    pages: dict[str, Any] = Field(default_factory=dict)
    accessed_at: str = ""
    published_at: str = ""
    source_quality: str = ""
    retrieval_query: str = ""

    @model_validator(mode="after")
    def require_traceable_source(self):
        if self.source_type == "rag" and not self.file_path.strip():
            raise ValueError("RAG evidence requires file_path")
        if self.source_type == "web":
            if not self.url.startswith(("http://", "https://")):
                raise ValueError("web evidence requires an HTTP(S) URL")
            if not self.accessed_at.strip():
                raise ValueError("web evidence requires accessed_at")
        return self


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    records: tuple[EvidenceRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class CoverageReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["sufficient", "insufficient", "unavailable"]
    covered_concepts: tuple[str, ...] = ()
    uncovered_concepts: tuple[str, ...] = ()
    web_fallback_required: bool = False

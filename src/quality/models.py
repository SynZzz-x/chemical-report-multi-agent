"""Structured quality-review contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IssueCategory = Literal[
    "CONTENT_DEFECT",
    "EVIDENCE_GAP",
    "DATA_DEFECT",
    "VISUAL_DEFECT",
    "WORKER_FAILURE",
    "LOCAL_PLAN_DEFECT",
    "SAFETY_BOUNDARY",
    "REQUIREMENT_MISSING",
    "EXTERNAL_BLOCKER",
    "REVIEW_FAILURE",
]


class ReviewIssue(BaseModel):
    code: str = Field(min_length=1)
    category: IssueCategory
    severity: Literal["minor", "major", "critical", "error"]
    description: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    responsible_handler: str = Field(min_length=1)
    revision_instruction: str = Field(min_length=1)
    resource_name: str | None = None


class QualityDimensions(BaseModel):
    completeness: int = Field(ge=0, le=5)
    evidence: int = Field(ge=0, le=5)
    logic: int = Field(ge=0, le=5)
    actionability: int = Field(ge=0, le=5)
    safety: int = Field(ge=0, le=5)


class ReviewAssessment(BaseModel):
    status: Literal["PASS", "REVISE", "BLOCKED", "HUMAN_REVIEW"]
    issues: list[ReviewIssue]
    quality_dimensions: QualityDimensions


class ReviewRecord(ReviewAssessment):
    review_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    created_at: str = Field(min_length=1)

"""Canonical typed contracts for automatic verification outcomes."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AssessmentStatus(str, Enum):
    PASS = "PASS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class AssessmentCategory(str, Enum):
    CONTENT_DEFECT = "CONTENT_DEFECT"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    LOCAL_PLAN_DEFECT = "LOCAL_PLAN_DEFECT"
    EXTERNAL_BLOCKER = "EXTERNAL_BLOCKER"


class AssessmentSeverity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class AssessmentIssue(BaseModel):
    """One semantic issue produced by a valid AutoVerifier assessment."""

    model_config = ConfigDict(extra="forbid")

    code: NonEmptyString
    category: AssessmentCategory
    description: NonEmptyString
    suggestion: NonEmptyString
    severity: AssessmentSeverity
    requirement_ids: list[NonEmptyString] = Field(default_factory=list)
    resource_name: NonEmptyString | None = None
    retrieval_query: NonEmptyString | None = None


class VerifierAssessment(BaseModel):
    """The single canonical schema accepted from the AutoVerifier model."""

    model_config = ConfigDict(extra="forbid")

    status: AssessmentStatus
    current_section: NonEmptyString
    issues: list[AssessmentIssue]
    requirements_met: list[NonEmptyString]
    requirements_missing: list[NonEmptyString]

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "VerifierAssessment":
        if self.status is AssessmentStatus.PASS:
            if self.issues or self.requirements_missing:
                raise ValueError(
                    "PASS assessment cannot contain issues or missing requirements"
                )
        elif not self.issues:
            raise ValueError("FAILED/BLOCKED assessment must contain at least one issue")
        return self


class VerifierExecutionFailure(BaseModel):
    """Infrastructure failure kept separate from semantic assessment issues."""

    model_config = ConfigDict(extra="forbid")

    code: Literal["VERIFIER_UNAVAILABLE", "LLM_ERROR", "LLM_NOT_ENABLED"]
    category: Literal["VERIFIER_FAILURE"] = "VERIFIER_FAILURE"
    message: NonEmptyString
    retryable: bool
    contract_attempts: int = Field(ge=1)


class AssessmentContractError(ValueError):
    """Raised when model output cannot satisfy :class:`VerifierAssessment`."""


def clean_json_fences(content: str) -> str:
    cleaned = str(content).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def parse_verifier_assessment(content: str) -> VerifierAssessment:
    """Parse and validate one model response as a semantic assessment."""

    try:
        payload: Any = json.loads(clean_json_fences(content))
        return VerifierAssessment.model_validate(payload)
    except Exception as exc:
        raise AssessmentContractError(_safe_validation_error(exc)) from exc


def _safe_validation_error(error: Exception) -> str:
    """Describe contract failures without copying invalid input values."""

    if isinstance(error, json.JSONDecodeError):
        return f"malformed_json line={error.lineno} column={error.colno}"
    if isinstance(error, ValidationError):
        details: list[str] = []
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        ):
            location = ".".join(str(part) for part in item.get("loc") or ()) or "root"
            details.append(f"{location}:{item.get('type') or 'validation_error'}")
        return "validation_error " + ", ".join(details[:20])
    return type(error).__name__

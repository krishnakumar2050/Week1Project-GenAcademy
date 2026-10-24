"""Provider-independent review contracts and graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low", "info"]
RANK = {name: i for i, name in enumerate(("critical", "high", "medium", "low", "info"))}


class Finding(BaseModel):
    severity: Severity
    title: str = Field(min_length=1)
    file: str
    line: int = Field(ge=1)
    evidence: str = Field(min_length=1, description="Exact source excerpt supporting the finding")
    explanation: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    suggested_test: str = Field(
        description="Concrete test scenario and expected assertion, or empty"
    )


class Review(BaseModel):
    findings: list[Finding]
    limitations: list[str]


class AgentResult(BaseModel):
    agent: str
    status: Literal["ok", "failed"]
    findings: list[Finding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error: dict | None = None
    execution: dict = Field(default_factory=dict)


class State(TypedDict):
    files: dict[str, str]
    results: Annotated[list[AgentResult], operator.add]
    report: dict

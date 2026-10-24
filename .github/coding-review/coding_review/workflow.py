"""Compatibility facade. New code should use domain, graph, and reporting."""

from coding_review.agents.prompts import POLICY, SPECIALTIES
from coding_review.agents.reviewer import make_reviewer
from coding_review.domain import RANK, AgentResult, Finding, Review, Severity, State
from coding_review.graph import build_graph
from coding_review.reporting.aggregation import orchestrate
from coding_review.reporting.rendering import markdown

__all__ = [
    "AgentResult",
    "Finding",
    "RANK",
    "Review",
    "Severity",
    "State",
    "POLICY",
    "SPECIALTIES",
    "make_reviewer",
    "build_graph",
    "orchestrate",
    "markdown",
]

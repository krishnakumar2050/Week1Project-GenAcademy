"""Specialist execution, response validation, and evidence verification."""

import logging
from time import monotonic

from coding_review.agents.context import source_prompt
from coding_review.agents.evidence import resolve_evidence
from coding_review.agents.knowledge import enrich_quality
from coding_review.agents.prompts import POLICY, SPECIALTIES
from coding_review.domain import AgentResult, Review, State
from coding_review.errors import diagnose, error_summary

logger = logging.getLogger(__name__)


def make_reviewer(name, model):
    def review(state: State):
        started = monotonic()
        logger.info("agent_started agent=%s", name)
        stage = "configure_structured_output"
        reviewer = None
        corrections = []
        knowledge_context, knowledge = (
            enrich_quality(state["files"]) if name == "quality" else ("", None)
        )
        try:
            reviewer = model.with_structured_output(Review)
            stage = "provider_request_or_parsing"
            answer = reviewer.invoke(
                [
                    ("system", POLICY + "\nYour specialty: " + SPECIALTIES[name]),
                    (
                        "human",
                        source_prompt(state["files"]) + knowledge_context,
                    ),
                ]
            )
            stage = "validate_response"
            answer = Review.model_validate(answer)
            stage = "validate_evidence"
            accepted = []
            limitations = list(answer.limitations)
            for finding in answer.findings:
                resolved, method = resolve_evidence(finding, state["files"])
                if resolved is None:
                    limitations.append(f"Discarded unverified finding: {finding.title}")
                else:
                    accepted.append(resolved)
                    if method != "exact":
                        corrections.append(
                            {
                                "title": finding.title,
                                "file": finding.file,
                                "original_line": finding.line,
                                "line": resolved.line,
                                "method": method,
                            }
                        )
            if corrections:
                limitations.append(
                    f"Corrected {len(corrections)} source citation(s) using unambiguous matches in the cited file; exact source excerpts retained."
                )
            result = AgentResult(
                agent=name, status="ok", findings=accepted, limitations=limitations
            )
        except Exception as exc:
            detail = diagnose(exc, stage, backend=getattr(model, "backend_name", "openai"))
            result = AgentResult(
                agent=name,
                status="failed",
                error=detail,
                limitations=["Review failed: " + error_summary(detail)],
            )
        result.execution = dict(getattr(reviewer, "execution_metadata", {}))
        if knowledge is not None:
            result.execution["knowledge_enrichment"] = knowledge
            if knowledge["status"] in {"failed", "empty"}:
                result.limitations.append(
                    "Keenable enrichment unavailable or empty; quality review used source context only."
                )
        result.execution["citation_corrections"] = corrections
        result.execution["duration_seconds"] = round(monotonic() - started, 3)
        if result.execution.get("backend") == "codex" and not result.execution.get("actual_model"):
            result.limitations.append(
                "Codex did not expose its resolved model in the startup header; actual model is unknown."
            )
        logger.info(
            "agent_finished agent=%s status=%s findings=%d duration_seconds=%s",
            name,
            result.status,
            len(result.findings),
            result.execution["duration_seconds"],
        )
        return {"results": [result]}

    return review

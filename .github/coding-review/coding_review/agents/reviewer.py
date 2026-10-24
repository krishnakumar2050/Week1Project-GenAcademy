"""Specialist execution, response validation, and evidence verification."""

import json
import logging
from time import monotonic

from coding_review.agents.evidence import verify_evidence
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
        try:
            reviewer = model.with_structured_output(Review)
            stage = "provider_request_or_parsing"
            answer = reviewer.invoke(
                [
                    ("system", POLICY + "\nYour specialty: " + SPECIALTIES[name]),
                    (
                        "human",
                        "Review this source snapshot (JSON path-to-source mapping):\n"
                        + json.dumps(state["files"]),
                    ),
                ]
            )
            stage = "validate_response"
            answer = Review.model_validate(answer)
            stage = "validate_evidence"
            accepted = []
            limitations = list(answer.limitations)
            for finding in answer.findings:
                if not verify_evidence(finding, state["files"]):
                    limitations.append(f"Discarded unverified finding: {finding.title}")
                else:
                    accepted.append(finding)
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

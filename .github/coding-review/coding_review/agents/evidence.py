"""Exact, line-bound source verification independent of model execution."""

from collections.abc import Mapping

from coding_review.domain import Finding


def verify_evidence(finding: Finding, files: Mapping[str, str]) -> bool:
    source = files.get(finding.file)
    lines = source.splitlines() if source is not None else []
    excerpt = finding.evidence.splitlines()
    return bool(excerpt) and lines[finding.line - 1 : finding.line - 1 + len(excerpt)] == excerpt

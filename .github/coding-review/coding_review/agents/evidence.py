"""Exact, line-bound source verification independent of model execution."""

from collections.abc import Mapping
from textwrap import dedent

from coding_review.domain import Finding


def verify_evidence(finding: Finding, files: Mapping[str, str]) -> bool:
    source = files.get(finding.file)
    lines = source.splitlines() if source is not None else []
    excerpt = finding.evidence.splitlines()
    return bool(excerpt) and lines[finding.line - 1 : finding.line - 1 + len(excerpt)] == excerpt


def resolve_evidence(finding: Finding, files: Mapping[str, str]) -> tuple[Finding | None, str]:
    """Repair coordinates only when the quoted source has an unambiguous match.

    Never search another file or perform fuzzy token/string replacement. The
    canonical excerpt always comes from the supplied snapshot and must pass
    strict verification after repair. Ambiguous or invented quotes are rejected.
    """
    if not finding.evidence.strip() or finding.file not in files:
        return None, "unverified"
    if verify_evidence(finding, files):
        return finding, "exact"
    lines = files[finding.file].splitlines()
    excerpt = finding.evidence.splitlines()
    width = len(excerpt)
    exact = [
        offset
        for offset in range(len(lines) - width + 1)
        if lines[offset : offset + width] == excerpt
    ]
    if len(exact) > 1:
        return None, "ambiguous"
    if exact:
        candidate = finding.model_copy(update={"line": exact[0] + 1})
        return (
            (candidate, "relocated_exact")
            if verify_evidence(candidate, files)
            else (None, "unverified")
        )
    # Common-margin indentation may be lost when a model copies code. Preserve
    # relative indentation and all non-indentation characters, including spaces
    # inside quoted literals, then recover the exact original source window.
    normalized = dedent("\n".join(excerpt))
    matches = [
        offset
        for offset in range(len(lines) - width + 1)
        if dedent("\n".join(lines[offset : offset + width])) == normalized
    ]
    if len(matches) != 1:
        return None, "ambiguous" if matches else "unverified"
    offset = matches[0]
    candidate = finding.model_copy(
        update={"line": offset + 1, "evidence": "\n".join(lines[offset : offset + width])}
    )
    return (
        (candidate, "normalized_indentation")
        if verify_evidence(candidate, files)
        else (None, "unverified")
    )

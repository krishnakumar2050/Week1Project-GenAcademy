"""Deterministic deduplication, prioritization, and completion checks."""

from coding_review.agents.prompts import SPECIALTIES
from coding_review.domain import RANK, State


def orchestrate(state: State):
    merged = {}
    results = sorted(state["results"], key=lambda r: r.agent)
    for result in results:
        for finding in result.findings:
            # Conservatively combine exact issue matches; retain distinct issues at a line.
            key = (finding.file, finding.line, finding.title.strip().casefold())
            if key not in merged:
                merged[key] = {**finding.model_dump(), "agents": [result.agent]}
            else:
                item = merged[key]
                item["agents"] = sorted(set(item["agents"] + [result.agent]))
                if RANK[finding.severity] < RANK[item["severity"]]:
                    item["severity"] = finding.severity
                if finding.suggested_test and finding.suggested_test not in item["suggested_test"]:
                    item["suggested_test"] = "\n".join(
                        filter(None, [item["suggested_test"], finding.suggested_test])
                    )
    findings = sorted(
        merged.values(), key=lambda f: (RANK[f["severity"]], f["file"], f["line"], f["title"])
    )
    report = {
        "status": "complete"
        if len(results) == len(SPECIALTIES)
        and {r.agent for r in results} == set(SPECIALTIES)
        and all(r.status == "ok" for r in results)
        else "incomplete",
        "files_reviewed": sorted(state["files"]),
        "counts": {severity: sum(f["severity"] == severity for f in findings) for severity in RANK},
        "findings": findings,
        "agents": [r.model_dump() for r in results],
        "actual_models": sorted(
            {r.execution["actual_model"] for r in results if r.execution.get("actual_model")}
        ),
        "coverage_note": "Static test-gap analysis only; tests were not executed and coverage was not measured.",
    }
    return {"report": report}

"""Transparent reference matching; unmatched findings require human adjudication."""

import statistics

from coding_review.agents.evidence import verify_evidence
from coding_review.domain import Finding


def score(case, report):
    expected = case["expected"]
    hits = set()
    matched = 0
    severity_ok = 0
    citations = 0
    tests = 0
    unmatched = []
    for i, raw in enumerate(report["findings"]):
        finding = Finding.model_validate(raw)
        citations += verify_evidence(finding, case["files"])
        text = (finding.title + " " + finding.explanation + " " + finding.recommendation).lower()
        end = finding.line + len(finding.evidence.splitlines()) - 1
        candidates = [
            e
            for e in expected
            if finding.file == e["file"]
            and finding.line <= e["end"]
            and end >= e["start"]
            and any(term in text for term in e["terms"])
        ]
        if candidates:
            label = candidates[0]
            hits.add(label["id"])
            matched += 1
            severity_ok += finding.severity in label["severity"]
            tests += bool(finding.suggested_test.strip())
        else:
            unmatched.append(i)
    return {
        "expected": len(expected),
        "detected": len(hits),
        "findings": len(report["findings"]),
        "matched": matched,
        "unmatched_indices": unmatched,
        "valid_citations": citations,
        "severity_acceptable": severity_ok,
        "test_suggestion_present": tests,
        "semantic_duplicate_excess": matched - len(hits),
        "complete": report["status"] == "complete",
        "duration_seconds": report["duration_seconds"],
        "clean_control_pass": not report["findings"] if not expected else None,
    }


def aggregate(rows):
    def total(key):
        return sum(r[key] for r in rows)

    def ratio(a, b):
        return a / b if b else None

    n = total("findings")
    m = total("matched")
    return {
        "runs": len(rows),
        "completion_rate": total("complete") / len(rows),
        "expected": total("expected"),
        "detected": total("detected"),
        "recall_proxy": ratio(total("detected"), total("expected")),
        "reference_precision_proxy": ratio(m, n),
        "retained_findings": n,
        "unmatched_findings": n - m,
        "citation_validity": ratio(total("valid_citations"), n),
        "acceptable_severity_rate": ratio(total("severity_acceptable"), m),
        "test_suggestion_presence": ratio(total("test_suggestion_present"), m),
        "semantic_duplicate_excess": total("semantic_duplicate_excess"),
        "clean_controls_passed": sum(r["clean_control_pass"] is True for r in rows),
        "clean_controls": sum(r["clean_control_pass"] is not None for r in rows),
        "latency_mean_seconds": statistics.mean(r["duration_seconds"] for r in rows),
        "latency_median_seconds": statistics.median(r["duration_seconds"] for r in rows),
        "latency_max_seconds": max(r["duration_seconds"] for r in rows),
    }

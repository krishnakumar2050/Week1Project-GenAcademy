"""Markdown and JSON presentation of review reports."""


def markdown(report):
    lines = [
        "# Code review report",
        "",
        f"Status: **{report['status']}**",
        "",
        report["coverage_note"],
        "",
        " | ".join(f"{key}: {value}" for key, value in report["counts"].items()),
        "",
    ]
    if report.get("run_id"):
        lines += [
            f"Run: `{report['run_id']}` · Schema: {report['schema_version']}",
            "",
            f"Repository: `{report['repository']}`",
            "",
            f"Snapshot SHA-256: `{report['snapshot_sha256']}`",
            "",
            f"Completed: {report['completed_at']} · Duration: {report['duration_seconds']} seconds",
            "",
        ]
    if any(agent.get("execution", {}).get("backend") for agent in report["agents"]):
        lines += [
            "## Model provenance",
            "",
            "| Agent | Requested model | Actual model | Evidence |",
            "| --- | --- | --- | --- |",
        ]
        for agent in report["agents"]:
            metadata = agent.get("execution", {})
            default = "CLI default" if metadata.get("backend") == "codex" else "Unspecified"
            lines += [
                f"| {agent['agent']} | {metadata.get('requested_model') or default} | {metadata.get('actual_model') or 'Unknown'} | {metadata.get('model_source', 'unavailable')} |"
            ]
        lines += [""]
    for index, finding in enumerate(report["findings"], 1):
        lines += [
            f"## {index}. [{finding['severity'].upper()}] {finding['title']}",
            "",
            f"Location: `{finding['file']}:{finding['line']}` · Agents: {', '.join(finding['agents'])}",
            "",
            finding["explanation"],
            "",
            f"Fix: {finding['recommendation']}",
            "",
        ]
        if finding["suggested_test"]:
            lines += [f"Suggested test: {finding['suggested_test']}", ""]
        lines += ["Evidence:", "", "````", finding["evidence"], "````", ""]
    if not report["findings"]:
        lines += [
            "No verified findings returned. This does not establish that the code is defect-free.",
            "",
        ]
    lines += ["## Review limitations", ""]
    for agent in report["agents"]:
        lines += [f"- {agent['agent']}: {agent['status']}"]
        lines += [f"  - {item}" for item in agent["limitations"]]
    return "\n".join(lines) + "\n"

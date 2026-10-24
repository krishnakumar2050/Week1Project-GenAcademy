"""Atomic publication prevents truncated reports if writing fails."""

import json
import os
import tempfile
from pathlib import Path

from coding_review.reporting.rendering import markdown


def write_report(report: dict, destination: Path, output_format: str = "markdown") -> None:
    if destination.is_symlink():
        raise ValueError("Report destination must not be a symlink")
    if output_format not in {"markdown", "json"}:
        raise ValueError("output format must be markdown or json")
    content = (
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if output_format == "json"
        else markdown(report)
    )
    # The temporary file is on the destination filesystem for atomic replace.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".review-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

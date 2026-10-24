"""Give models explicit source coordinates instead of asking them to count lines."""

import json
from collections.abc import Mapping


def source_prompt(files: Mapping[str, str]) -> str:
    snapshot = {
        path: [{"line": number, "text": text} for number, text in enumerate(source.splitlines(), 1)]
        for path, source in sorted(files.items())
    }
    return (
        "Review the source below. Paths are repository-relative. Each line object contains its "
        "actual one-based line number and exact source text. Use that number for the finding's "
        "first evidence line; never estimate or count lines yourself. Copy evidence from text "
        "values, including indentation. For multiline evidence, join consecutive text values "
        "with newline characters. Do not include line labels or JSON wrappers in evidence.\n"
        "SOURCE_LINES_JSON:\n" + json.dumps(snapshot, ensure_ascii=False)
    )

"""Run: python -m evals.run --output reports/evals/run-name [--repeats 1]."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from time import monotonic

from coding_review.agents.prompts import POLICY
from coding_review.agents.reviewer import make_reviewer
from coding_review.backends.factory import create_backend
from coding_review.config import DEFAULT_BACKEND, ReviewConfig
from coding_review.service import review_repository
from evals.dataset import CASES
from evals.scoring import aggregate, score


class CombinedBackend:
    """Same provider and schema/citation pipeline, one combined review request."""

    def __init__(self, backend):
        self.backend = backend
        self.backend_name = getattr(backend, "backend_name", "codex")

    def with_structured_output(self, schema):
        inner = self.backend.with_structured_output(schema)

        class Reviewer:
            @property
            def execution_metadata(self):
                return getattr(inner, "execution_metadata", {})

            def invoke(self, messages):
                return inner.invoke(
                    [
                        (
                            "system",
                            POLICY
                            + "\nReview correctness bugs, anti-patterns, security vulnerabilities, and missing tests. Provide concrete failure scenarios and tests; do not invent measured coverage.",
                        ),
                        *messages[1:],
                    ]
                )

        return Reviewer()


def run_case(case, variant, config):
    os.environ["KEENABLE_ENABLED"] = "true" if variant == "three_keenable" else "false"
    if variant != "single":
        return review_repository(Path("eval-fixture"), config, files=case["files"])
    start = monotonic()
    result = make_reviewer("quality", CombinedBackend(create_backend(config)))(
        {"files": case["files"]}
    )["results"][0]
    return {
        "status": "complete" if result.status == "ok" else "incomplete",
        "findings": [f.model_dump() for f in result.findings],
        "agents": [result.model_dump()],
        "duration_seconds": round(monotonic() - start, 3),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["codex", "fireworks", "openai"],
        help="Default: Fireworks MiniMax M3.1 Flash; other backends require explicit selection",
    )
    parser.add_argument("--model")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--repeats", type=int, default=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    config = ReviewConfig(
        backend=args.backend, model=args.model, timeout=240, codex_executable=args.codex_executable
    )
    fingerprint = hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest()
    tasks = [
        (v, c, repeat)
        for repeat in range(args.repeats)
        for c in CASES
        for v in ("single", "three_source", "three_keenable")
    ]
    random.Random(42).shuffle(tasks)
    rows = []
    for variant, case, repeat in tasks:
        print(f"START {variant} {case['id']} repeat={repeat}", flush=True)
        report = run_case(case, variant, config)
        metrics = score(case, report)
        row = {
            "variant": variant,
            "case": case["id"],
            "repeat": repeat,
            "metrics": metrics,
            "report": report,
        }
        rows.append(row)
        (args.output / f"{variant}-{case['id']}-{repeat}.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
        print(
            f"END complete={metrics['complete']} detected={metrics['detected']}/{metrics['expected']} findings={metrics['findings']}",
            flush=True,
        )
    summary = {
        "dataset_sha256": fingerprint,
        "dataset": CASES,
        "repeats": args.repeats,
        "grading": "line-span plus concept-substring reference matching; proxy only; unmatched findings need adjudication",
        "cost": "Billed cost not measured; provider usage, when available, is retained in per-agent execution metadata",
        "variants": {
            v: aggregate([r["metrics"] for r in rows if r["variant"] == v])
            for v in ("single", "three_source", "three_keenable")
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["variants"], indent=2), flush=True)


if __name__ == "__main__":
    main()

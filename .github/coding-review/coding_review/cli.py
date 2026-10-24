"""CLI composition root; library users can call service.review_repository."""

import argparse
import logging
import sys
from pathlib import Path

from coding_review.config import ReviewConfig
from coding_review.errors import diagnose, error_summary, safe_text
from coding_review.exceptions import ConfigurationError
from coding_review.reporting.writer import write_report
from coding_review.service import review_repository
from coding_review.sources import load_files  # Compatibility for existing users.


def build_parser():
    parser = argparse.ArgumentParser(
        description="Review source with three LangGraph specialists. Source is sent to your configured backend."
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "--backend",
        choices=["fireworks", "codex", "openai"],
        default="fireworks",
        help="fireworks uses MiniMax M3.1 Flash (default); codex uses ChatGPT login; openai uses API billing",
    )
    parser.add_argument(
        "--model", help="Backend model ID; defaults to MiniMax M3.1 Flash for Fireworks"
    )
    parser.add_argument(
        "--codex-executable", default="codex", help="Codex executable name or absolute path"
    )
    parser.add_argument("--timeout", type=float, default=240, help="Per-review timeout in seconds")
    parser.add_argument("--output", type=Path, default=Path("review-report.md"))
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--max-bytes", type=int, default=120_000)
    parser.add_argument(
        "--max-concurrency", type=int, default=3, help="Concurrent specialist calls (1–3)"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude a repository-relative glob; repeat as needed",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Print selected file paths without calling a model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log lifecycle events and timings without source text",
    )
    return parser


def run(args) -> int:
    config = ReviewConfig(
        backend=args.backend,
        model=args.model,
        timeout=args.timeout,
        max_bytes=args.max_bytes,
        max_concurrency=args.max_concurrency,
        exclude=tuple(args.exclude),
        codex_executable=args.codex_executable,
    )
    repository = args.repository.resolve()
    files = load_files(repository, config.max_bytes, config.exclude)
    if args.snapshot_only:
        print("\n".join(files))
        return 0
    destination = args.output.absolute()
    if destination.is_symlink():
        raise ConfigurationError("Report destination must not be a symlink")
    if destination.resolve() in {(repository / name).resolve() for name in files}:
        raise ConfigurationError("Report destination must not overwrite a reviewed source file")
    if not destination.parent.is_dir():
        raise ConfigurationError("Report output directory does not exist")
    report = review_repository(repository, config, files=files)
    write_report(report, destination, args.format)
    print(f"Review {report['status']}: {len(report['findings'])} findings. Report: {destination}")
    if report["status"] == "incomplete":
        for agent in report["agents"]:
            if agent.get("error"):
                print(f"{agent['agent']}: {error_summary(agent['error'])}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        code = run(args)
    except (ConfigurationError, OSError, UnicodeError, ValueError) as exc:
        parser.error(safe_text(exc))
    except KeyboardInterrupt:
        print("Review interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        parser.error(error_summary(diagnose(exc, "review_repository", backend=args.backend)))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

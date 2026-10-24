"""Review immutable PR source blobs; never checkout or execute PR code."""

import argparse
import base64
import html
import json
import os
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from coding_review.config import ReviewConfig
from coding_review.exceptions import ConfigurationError
from coding_review.reporting.rendering import markdown
from coding_review.reporting.writer import write_report
from coding_review.service import review_repository
from coding_review.sources import EXCLUDED, EXTENSIONS

MARKER = "<!-- langgraph-coding-review -->"


class GitHubClient:
    def __init__(self, repository: str, token: str, transport=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ConfigurationError("Invalid GitHub repository name")
        if not token:
            raise ConfigurationError("GITHUB_TOKEN is required")
        self.repository = repository
        self.client = httpx.Client(
            base_url="https://api.github.com",
            timeout=30,
            transport=transport,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def request(self, method, path, **kwargs):
        response = self.client.request(method, f"/repos/{self.repository}/{path}", **kwargs)
        response.raise_for_status()
        return response.json()

    def close(self):
        self.client.close()

    def pull_request(self, number):
        return self.request("GET", f"pulls/{number}")

    def source_snapshot(self, sha: str, max_bytes: int):
        if not re.fullmatch(r"[a-fA-F0-9]{40}", sha):
            raise ValueError("Expected a full PR head commit SHA")
        tree = self.request("GET", f"git/trees/{sha}", params={"recursive": "1"})
        if tree.get("truncated"):
            raise ValueError(
                "GitHub truncated the repository tree; review scope cannot be established"
            )
        files = {}
        total = 0
        for entry in sorted(tree["tree"], key=lambda item: item["path"]):
            path = PurePosixPath(entry["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid repository source path")
            if (
                entry["type"] != "blob"
                or entry["mode"] not in {"100644", "100755"}
                or path.suffix not in EXTENSIONS
                or any(part.startswith(".") or part in EXCLUDED for part in path.parts)
            ):
                continue
            size = entry.get("size")
            if not isinstance(size, int) or size > max_bytes - total:
                raise ValueError("PR source snapshot exceeds byte budget")
            blob = self.request("GET", f"git/blobs/{quote(entry['sha'], safe='')}")
            if blob.get("encoding") != "base64":
                raise ValueError("Unsupported GitHub source encoding")
            # GitHub includes newlines in base64 blob responses.
            content = base64.b64decode(blob["content"], validate=False)
            total += len(content)
            if total > max_bytes:
                raise ValueError("PR source snapshot exceeds byte budget")
            files[path.as_posix()] = content.decode("utf-8")
        if not files:
            raise ValueError("No supported source files found in the PR snapshot")
        return files

    def upsert_comment(self, number: int, body: str):
        # Only replace this Actions bot's marked comment; never a human comment.
        for page in range(1, 101):
            comments = self.request(
                "GET", f"issues/{number}/comments", params={"per_page": 100, "page": page}
            )
            for comment in comments:
                if (
                    comment.get("body", "").startswith(MARKER)
                    and comment.get("user", {}).get("login") == "github-actions[bot]"
                ):
                    return self.request(
                        "PATCH", f"issues/comments/{comment['id']}", json={"body": body}
                    )
            if len(comments) < 100:
                return self.request("POST", f"issues/{number}/comments", json={"body": body})
        raise ValueError("Too many PR comments to safely locate an existing review")


def require_same_repository(pr, repository):
    if pr["head"]["repo"]["full_name"] != repository:
        raise ConfigurationError(
            "Automatic secret-backed reviews are restricted to same-repository PRs"
        )


def comment_body(report: dict) -> str:
    provenance = report["github"]
    intro = f"{MARKER}\n## Automated multi-agent code review\n\nReviewed commit `{provenance['head_sha']}`.\n\n"
    # Escape HTML and mention syntax from untrusted model/source content.
    content = html.escape(markdown(report), quote=False).replace("@", "&#64;")
    maximum = 55_000 - len(intro)
    if len(content) > maximum:
        content = (
            content[:maximum] + "\n\nReport truncated; download the complete Actions artifact."
        )
    return intro + content


def run_review(client, number, expected_sha, output: Path, max_bytes=120_000, backend=None):
    pr = client.pull_request(number)
    require_same_repository(pr, client.repository)
    if pr["head"]["sha"] != expected_sha:
        raise ValueError("PR head changed before review; wait for the latest workflow run")
    snapshot = client.source_snapshot(expected_sha, max_bytes)
    report = review_repository(
        Path("."),
        ReviewConfig(backend="fireworks", max_bytes=max_bytes),
        backend=backend,
        files=snapshot,
    )
    report["repository"] = f"https://github.com/{client.repository}"
    report["github"] = {
        "repository": client.repository,
        "pull_request": number,
        "head_sha": expected_sha,
        "scope": "Full supported source snapshot at the PR head; not limited to changed lines",
    }
    write_report(report, output / "report.json", "json")
    write_report(report, output / "report.md")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["review", "publish"])
    parser.add_argument("--output", type=Path, default=Path("review-output"))
    args = parser.parse_args()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    number = int(event["number"])
    repository = os.environ["GITHUB_REPOSITORY"]
    expected_sha = event["pull_request"]["head"]["sha"]
    client = GitHubClient(repository, os.environ.get("GITHUB_TOKEN", ""))
    try:
        if args.mode == "review":
            args.output.mkdir(parents=True, exist_ok=True)
            report = run_review(client, number, expected_sha, args.output)
            if report["status"] != "complete":
                raise SystemExit(2)
        else:
            report = json.loads((args.output / "report.json").read_text())
            pr = client.pull_request(number)
            require_same_repository(pr, repository)
            if report["github"]["head_sha"] != expected_sha or pr["head"]["sha"] != expected_sha:
                print("Skipping stale review; a newer PR head is available.")
                return
            client.upsert_comment(number, comment_body(report))
    finally:
        client.close()


if __name__ == "__main__":
    main()

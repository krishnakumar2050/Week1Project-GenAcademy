# Install automated multi-agent pull-request review

This integration runs the same three LangGraph specialists with MiniMax M3.1 Flash on Fireworks. It reviews a full supported-source snapshot at each PR head, not just changed lines. Existing code issues can therefore appear in a PR report.

## Activation

1. Merge the integration pull request containing `.github/workflows/ai-code-review.yml` and the review engine under `.github/coding-review/`.
2. In the target repository, open **Settings → Secrets and variables → Actions → New repository secret**. Add `FIREWORKS_API_KEY` with your Fireworks key. Do not commit the key.
3. Open a same-repository pull request, push a new commit to one, reopen it, or mark a draft ready for review.

The installation PR itself does not invoke the engine when the base branch does not yet contain it. The job emits a bootstrap notice instead. A missing Fireworks key produces an actionable Actions error; it does not fall back to another provider or claim a successful review.

## Results

- **Multi-agent AI code review** appears in the PR's checks.
- One marked bot comment contains the prioritized report and is updated on new pushes.
- The Actions run retains full Markdown and JSON report artifacts for 14 days.
- Every report records the reviewed commit, source hash, per-agent results, and runtime model when returned by the provider.
- The job fails on an incomplete review; severity findings are advisory and do not by themselves fail the check.
- A 120,000-byte source budget limits inference cost. Oversized snapshots fail explicitly; adjust the trusted integration entry point before enabling on a larger repository.

## Trust boundaries

The workflow uses `pull_request`, skips fork and Dependabot PRs, and uses a trusted base-commit checkout. The installed engine and dependency manifest come from that trusted checkout. Source at the PR head is retrieved as immutable GitHub blob data through the API; the integration never checks out, imports, installs dependencies from, or runs tests from PR code. The review backend has no tool loop.

The Fireworks key is scoped to the inference step. `GITHUB_TOKEN` receives `contents: read` and `pull-requests: write` for source retrieval and report publication. Only a marked comment belonging to `github-actions[bot]` can be replaced. Stale commit reports are not posted. User/model content is HTML-escaped and mention syntax is escaped before publication.

Only trusted collaborators can create same-repository branches; changes to the workflow by collaborators retain the normal GitHub Actions trust model. Protect the default branch and review changes under `.github/` as part of repository governance. This integration does not configure branch protection, GitHub account-wide hooks, or fork review approvals.

Fireworks API usage is billed to the Fireworks account. GitHub Actions usage and repository workflow policies also apply. The integration does not use ChatGPT Pro authentication in CI.

## Reuse in another repository

Copy the trusted review package, its `pyproject.toml` and README under `.github/coding-review/`, then copy the workflow into `.github/workflows/`. Add a Fireworks secret in each repository. The engine is deliberately vendored so a public repository need not fetch a private coding-agent repository using a long-lived token. Updates to the central local project must be explicitly copied into the installed engine.

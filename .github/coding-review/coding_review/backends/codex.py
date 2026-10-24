"""Structured-output adapter for Codex CLI authenticated with ChatGPT."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from coding_review.errors import safe_text
from coding_review.exceptions import CodexError


def codex_environment():
    env = os.environ.copy()
    # Never silently use API-key billing or inherit a parent agent's thread identity.
    for name in (
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "CODEX_THREAD_ID",
        "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
    ):
        env.pop(name, None)
    return env


def strict_schema(value):
    if isinstance(value, dict):
        result = {key: strict_schema(item) for key, item in value.items()}
        if result.get("type") == "object":
            result["additionalProperties"] = False
            result["required"] = list(result.get("properties", {}))
        return result
    if isinstance(value, list):
        return [strict_schema(item) for item in value]
    return value


class CodexModel:
    def __init__(self, model=None, executable="codex", timeout=240):
        self.executable = shutil.which(executable)
        if not self.executable:
            raise CodexError("Codex CLI not found. Install Codex CLI and run codex login.")
        self.model = model
        self.timeout = timeout

    def check_login(self):
        try:
            result = subprocess.run(
                [self.executable, "login", "status"],
                env=codex_environment(),
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CodexError("Could not check Codex login status. Run codex login status.") from exc
        output = (result.stdout + result.stderr).lower()
        if result.returncode and "not logged in" not in output:
            raise CodexError(
                f"Codex login-status command failed with exit code {result.returncode}. Check executable dependencies (including Node for npm installations), or select --codex-executable."
            )
        if result.returncode or "logged in using chatgpt" not in output:
            raise CodexError(
                "ChatGPT login required. Run codex login and choose Sign in with ChatGPT; API-key login is not accepted."
            )

    def with_structured_output(self, schema):
        return CodexStructuredOutput(self, schema)


class CodexStructuredOutput:
    def __init__(self, model, schema):
        self.model = model
        self.schema = schema
        self.execution_metadata = {
            "backend": "codex",
            "requested_model": model.model,
            "actual_model": None,
            "model_source": "unavailable",
        }

    def invoke(self, messages):
        with tempfile.TemporaryDirectory(prefix="coding-review-") as directory:
            root = Path(directory)
            schema_path = root / "schema.json"
            output_path = root / "result.json"
            schema_path.write_text(
                json.dumps(strict_schema(self.schema.model_json_schema())), encoding="utf-8"
            )
            command = [
                self.model.executable,
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "-c",
                'forced_login_method="chatgpt"',
                "-c",
                'model_provider="openai"',
                "-c",
                'approval_policy="never"',
                "--cd",
                directory,
                "--color",
                "never",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if self.model.model:
                command += ["--model", self.model.model]
            command += ["-"]
            prompt = (
                "Perform a static review of only the snapshot below. Do not call tools, inspect files, "
                "execute code, or delegate. Preserve indentation in exact evidence excerpts. "
                "Return only the requested structured review.\n"
                + "\n\n".join(f"{role.upper()}:\n{content}" for role, content in messages)
            )
            try:
                result = subprocess.run(
                    command,
                    input=prompt,
                    env=codex_environment(),
                    capture_output=True,
                    text=True,
                    timeout=self.model.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexError(
                    f"Codex review exceeded {self.model.timeout} seconds; retry or increase --timeout."
                ) from exc
            except OSError as exc:
                raise CodexError(
                    "Unable to start Codex CLI; check installation and executable permissions."
                ) from exc
            # Non-JSON mode emits a CLI-generated startup header on stderr.
            # Parse only that header, never model-authored review text or a
            # requested --model value, which may differ from the resolved model.
            header = ("\n" + result.stderr).split("\nuser\n", 1)[0]
            match = re.search(r"^model:\s*([A-Za-z0-9][A-Za-z0-9._:/-]*)\s*$", header, re.MULTILINE)
            if match:
                self.execution_metadata.update(
                    actual_model=match.group(1), model_source="codex_cli_startup"
                )
            if result.returncode:
                errors = []
                for line in result.stdout.splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "error":
                        errors.append(event.get("message", "Codex error"))
                    elif event.get("type") == "turn.failed":
                        errors.append(event.get("error", {}).get("message", "Codex turn failed"))
                # Do not dump CLI output, which may contain source or tool outputs.
                if not errors:
                    errors = [
                        line
                        for line in result.stderr.splitlines()
                        if line.startswith("error:") or line.startswith("ERROR:")
                    ]
                message = (
                    safe_text("; ".join(errors))
                    if errors
                    else "No structured error returned; check codex login status and network access."
                )
                raise CodexError(f"Codex exited with status {result.returncode}: {message}")
            if not output_path.exists():
                raise CodexError("Codex completed without a final response; retry the review.")
            return self.schema.model_validate_json(output_path.read_text(encoding="utf-8"))

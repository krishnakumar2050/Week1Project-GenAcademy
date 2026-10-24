"""Actionable diagnostics without dumping request bodies or validation inputs."""

import os
import re

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError

from coding_review.exceptions import CodexError, ProviderResponseError


def safe_text(value):
    text = str(value)
    for name, secret in os.environ.items():
        if secret and any(
            marker in name.upper() for marker in ("KEY", "TOKEN", "PASSWORD", "SECRET")
        ):
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"sk-[\w*-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
    return " ".join(text.split())[:1500]


def diagnose(exc, stage, backend="openai"):
    detail = {"stage": stage, "exception": type(exc).__name__}
    if isinstance(exc, CodexError):
        detail["message"] = safe_text(exc)
        hint = "Check Codex ChatGPT sign-in, subscription usage limits, CLI version, and network access."
    elif isinstance(exc, ProviderResponseError):
        detail["message"] = safe_text(exc)
        hint = "Check provider response limits, model support, and snapshot size."
    elif isinstance(exc, APIStatusError):
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        error = error if isinstance(error, dict) else {}
        detail.update(status_code=exc.status_code)
        for field in ("code", "type", "param"):
            value = error.get(field, getattr(exc, field, None))
            if value is not None:
                detail[field] = safe_text(value)
        if exc.request_id:
            detail["request_id"] = safe_text(exc.request_id)
        # Only extract the API error message, never str(exc), response bodies,
        # request headers, or raw model outputs.
        if isinstance(error.get("message"), str):
            detail["message"] = safe_text(error["message"])
        code = detail.get("code", "")
        if code in {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}:
            hint = "Check API project billing, credits, and spend limits; retrying alone will not restore quota."
        elif exc.status_code == 401:
            variable = "FIREWORKS_API_KEY" if backend == "fireworks" else "OPENAI_API_KEY"
            hint = f"Set a valid {variable} in the shell running coding-review."
        elif exc.status_code in {403, 404}:
            hint = "Check model ID, API project permissions, and model access."
        elif exc.status_code == 429:
            hint = "Check request/token rate limits; wait before retrying."
        elif exc.status_code == 400:
            hint = "Check the named parameter and model support for structured outputs."
        elif exc.status_code >= 500:
            hint = "Provider service error; retry later and retain the request ID."
        else:
            hint = "Inspect the API error code and request ID."
    elif isinstance(exc, APITimeoutError):
        hint = "Provider request timed out; check connectivity or retry later."
    elif isinstance(exc, APIConnectionError):
        hint = "Cannot connect to the provider; check network, proxy, DNS, and TLS certificates."
    elif isinstance(exc, ValidationError):
        detail["validation_errors"] = [
            {"location": list(item["loc"]), "type": item["type"]}
            for item in exc.errors(include_input=False, include_context=False, include_url=False)
        ]
        hint = "Model response does not match the Review schema; check structured-output support."
    else:
        hint = "Local configuration, parsing, or workflow error; use the exception type and failure stage to investigate."
    detail["hint"] = hint
    return detail


def error_summary(detail):
    return "; ".join(f"{key}: {value}" for key, value in detail.items())

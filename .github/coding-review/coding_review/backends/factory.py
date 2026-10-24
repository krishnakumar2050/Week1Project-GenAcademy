"""Backend selection and authentication at the application boundary."""

import os

from coding_review.backends.base import ReviewBackend
from coding_review.config import ReviewConfig
from coding_review.exceptions import ConfigurationError


def create_backend(config: ReviewConfig) -> ReviewBackend:
    if config.backend == "fireworks":
        from coding_review.backends.fireworks import DEFAULT_FIREWORKS_MODEL, FireworksBackend

        key = os.getenv("FIREWORKS_API_KEY", "")
        if not key.strip():
            raise ConfigurationError(
                "Set FIREWORKS_API_KEY (Fireworks API usage is billed separately from ChatGPT Pro)"
            )
        model = config.model or os.getenv("FIREWORKS_MODEL") or DEFAULT_FIREWORKS_MODEL
        return FireworksBackend(api_key=key, model=model, timeout=config.timeout)
    if config.backend == "codex":
        from coding_review.backends.codex import CodexModel

        backend = CodexModel(
            model=config.model, timeout=config.timeout, executable=config.codex_executable
        )
        backend.check_login()
        return backend
    model = config.model or os.getenv("REVIEW_MODEL")
    if not model:
        raise ConfigurationError("Set --model or REVIEW_MODEL to a structured-output-capable model")
    if not os.getenv("OPENAI_API_KEY"):
        raise ConfigurationError("Set OPENAI_API_KEY")
    from coding_review.backends.openai import OpenAIBackend

    return OpenAIBackend(model=model, timeout=config.timeout)

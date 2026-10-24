"""Validated, immutable runtime configuration shared by CLI and library users."""

import math
from dataclasses import dataclass
from typing import Literal

from coding_review.exceptions import ConfigurationError


@dataclass(frozen=True)
class ReviewConfig:
    backend: Literal["codex", "openai", "fireworks"] = "fireworks"
    model: str | None = None
    timeout: float = 240
    max_bytes: int = 120_000
    max_concurrency: int = 3
    exclude: tuple[str, ...] = ()
    codex_executable: str = "codex"

    def __post_init__(self):
        if self.backend not in {"codex", "openai", "fireworks"}:
            raise ConfigurationError("backend must be codex, openai, or fireworks")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ConfigurationError("timeout must be a finite positive number")
        if self.max_bytes <= 0:
            raise ConfigurationError("max_bytes must be positive")
        if not 1 <= self.max_concurrency <= 3:
            raise ConfigurationError("max_concurrency must be between 1 and 3")
        if self.model is not None and not self.model.strip():
            raise ConfigurationError("model cannot be blank")
        if not self.codex_executable.strip():
            raise ConfigurationError("codex_executable cannot be blank")

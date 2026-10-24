"""Structural backend interfaces: each invocation owns its response metadata."""

from typing import Any, Protocol

from pydantic import BaseModel


class StructuredReviewer(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> Any: ...


class ReviewBackend(Protocol):
    def with_structured_output(self, schema: type[BaseModel]) -> StructuredReviewer: ...

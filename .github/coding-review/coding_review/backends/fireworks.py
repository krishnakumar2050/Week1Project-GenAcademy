"""MiniMax reviews through Fireworks' documented OpenAI-compatible API."""

import json

from pydantic import BaseModel

from coding_review.exceptions import ConfigurationError, ProviderResponseError

DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/minimax-m3p1-flash"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksBackend:
    backend_name = "fireworks"

    def __init__(
        self, api_key: str, model: str = DEFAULT_FIREWORKS_MODEL, timeout: float = 240, client=None
    ):
        if not api_key.strip():
            raise ConfigurationError("Set FIREWORKS_API_KEY")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=FIREWORKS_BASE_URL,
                timeout=timeout,
                max_retries=2,
                organization="",
                project="",
            )
        self.client = client
        self.model = model

    def with_structured_output(self, schema: type[BaseModel]):
        return FireworksStructuredReviewer(self.client, self.model, schema)


class FireworksStructuredReviewer:
    def __init__(self, client, model, schema):
        self.client = client
        self.model = model
        self.schema = schema
        self.execution_metadata = {
            "backend": "fireworks",
            "requested_model": model,
            "actual_model": None,
            "model_source": "unavailable",
        }

    def invoke(self, messages):
        schema = self.schema.model_json_schema()
        payload = [
            {
                "role": "system",
                "content": "Return only JSON matching this schema:\n" + json.dumps(schema),
            }
        ]
        roles = {"system": "system", "human": "user", "user": "user", "assistant": "assistant"}
        payload.extend({"role": roles[role], "content": content} for role, content in messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema.__name__,
                    "schema": schema,
                },
            },
            max_tokens=8192,
        )
        actual_model = getattr(response, "model", None)
        if isinstance(actual_model, str) and actual_model:
            self.execution_metadata.update(
                actual_model=actual_model, model_source="provider_response"
            )
        if not response.choices:
            raise ProviderResponseError("Fireworks returned no response choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ProviderResponseError(
                "Fireworks output reached the token limit; review a smaller source snapshot."
            )
        if choice.finish_reason != "stop":
            raise ProviderResponseError(
                "Fireworks did not finish a normal structured response; check provider safety or model limits."
            )
        if getattr(choice.message, "refusal", None):
            raise ProviderResponseError("Fireworks declined the review request.")
        if not choice.message.content:
            raise ProviderResponseError("Fireworks returned an empty structured response.")
        return self.schema.model_validate_json(choice.message.content)

"""API backend with per-invocation provider-reported model provenance."""

from pydantic import BaseModel


class OpenAIBackend:
    def __init__(self, model: str, timeout: float, client=None):
        if client is None:
            from langchain_openai import ChatOpenAI

            client = ChatOpenAI(model=model, timeout=timeout, max_retries=2)
        self.client = client
        self.model = model

    def with_structured_output(self, schema: type[BaseModel]):
        return OpenAIStructuredReviewer(self.client, self.model, schema)


class OpenAIStructuredReviewer:
    def __init__(self, client, model, schema):
        self.runner = client.with_structured_output(schema, include_raw=True)
        self.execution_metadata = {
            "backend": "openai",
            "requested_model": model,
            "actual_model": None,
            "model_source": "unavailable",
        }

    def invoke(self, messages):
        response = self.runner.invoke(messages)
        metadata = getattr(response.get("raw"), "response_metadata", {})
        actual_model = metadata.get("model_name") or metadata.get("model")
        if isinstance(actual_model, str) and actual_model:
            self.execution_metadata.update(
                actual_model=actual_model, model_source="provider_response"
            )
        if response.get("parsing_error") is not None:
            raise response["parsing_error"]
        return response.get("parsed")

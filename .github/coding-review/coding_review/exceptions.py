"""Application exceptions without dependencies on provider adapters."""


class ConfigurationError(ValueError):
    """Invalid review configuration."""


class CodexError(RuntimeError):
    """Contains only a bounded diagnostic suitable for inclusion in reports."""


class ProviderResponseError(RuntimeError):
    """A known response failure with no raw model content in the message."""

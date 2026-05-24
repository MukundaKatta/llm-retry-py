"""Built-in retryable-error code lists for major LLM providers.

Each list is what the provider's docs say is a transient,
caller-should-retry error. Use them on a stringified error or message:

    from llm_retry import predicates
    assert predicates.is_anthropic_retryable("rate_limit_error")
    assert not predicates.is_anthropic_retryable("authentication_error")

The match is `contains` (case-sensitive) so you can pass a full error
message or just the code substring.
"""

from __future__ import annotations

from collections.abc import Iterable

# Anthropic API error codes that are safe to retry.
# Source: https://docs.anthropic.com/en/api/errors
ANTHROPIC_RETRYABLE: tuple[str, ...] = (
    "rate_limit_error",
    "overloaded_error",
    "api_error",
    "timeout",
)

# OpenAI API error codes/conditions that are safe to retry.
# Source: https://platform.openai.com/docs/guides/error-codes
OPENAI_RETRYABLE: tuple[str, ...] = (
    "rate_limit_exceeded",
    "server_error",
    "engine_overloaded",
    "tokens_exhausted",
    "timeout",
)

# AWS Bedrock service-side error codes that are safe to retry.
# Source: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshoot.html
BEDROCK_RETRYABLE: tuple[str, ...] = (
    "ThrottlingException",
    "Throttling",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ProvisionedThroughputExceededException",
    "ModelTimeoutException",
)

# Google Gemini API status codes that are safe to retry.
# Source: https://ai.google.dev/api/rest/v1/HttpStatusCode
GEMINI_RETRYABLE: tuple[str, ...] = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
)

# Generic HTTP status codes that are typically retryable.
HTTP_RETRYABLE_STATUSES: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)


def contains_any(s: str, patterns: Iterable[str]) -> bool:
    """True if `s` contains any of `patterns` (case-sensitive substring)."""
    return any(p in s for p in patterns)


def is_anthropic_retryable(s: str) -> bool:
    """True if `s` looks like an Anthropic retryable error code or message."""
    return contains_any(s, ANTHROPIC_RETRYABLE)


def is_openai_retryable(s: str) -> bool:
    """True if `s` looks like an OpenAI retryable error code or message."""
    return contains_any(s, OPENAI_RETRYABLE)


def is_bedrock_retryable(s: str) -> bool:
    """True if `s` looks like a Bedrock retryable error code or message."""
    return contains_any(s, BEDROCK_RETRYABLE)


def is_gemini_retryable(s: str) -> bool:
    """True if `s` looks like a Gemini retryable error code or message."""
    return contains_any(s, GEMINI_RETRYABLE)


def is_http_status_retryable(code: int) -> bool:
    """True if `code` is a typically-retryable HTTP status."""
    return code in HTTP_RETRYABLE_STATUSES

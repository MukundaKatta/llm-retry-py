"""llm-retry - exponential backoff with full jitter for LLM API calls.

Most retry libraries are framework middleware or async-only. This one is
a small function you wrap around any callable. Sync and async loops
share the same `RetryPolicy`. Built-in retryable-code sets for Anthropic,
OpenAI, AWS Bedrock, and Google Gemini live in `llm_retry.predicates`.

    from llm_retry import retry, RetryPolicy, predicates

    def call_anthropic():
        return client.messages.create(...)

    resp = retry(
        call_anthropic,
        policy=RetryPolicy(),
        should_retry=lambda e: predicates.is_anthropic_retryable(str(e)),
    )

Async:

    resp = await retry_async(
        call_openai_async,
        policy=RetryPolicy(),
        should_retry=lambda e: predicates.is_openai_retryable(str(e)),
    )

Sibling to the Rust crate `llm-retry`.
"""

from llm_retry import predicates
from llm_retry.retry import (
    Jitter,
    RetryError,
    RetryExhausted,
    RetryPolicy,
    retry,
    retry_async,
)

__version__ = "0.1.0"

__all__ = [
    "Jitter",
    "RetryError",
    "RetryExhausted",
    "RetryPolicy",
    "__version__",
    "predicates",
    "retry",
    "retry_async",
]

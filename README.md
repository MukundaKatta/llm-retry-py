# llm-retry-py

[![PyPI](https://img.shields.io/pypi/v/llm-retry-py.svg)](https://pypi.org/project/llm-retry-py/)
[![Python](https://img.shields.io/pypi/pyversions/llm-retry-py.svg)](https://pypi.org/project/llm-retry-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Exponential backoff with full jitter for LLM API calls.**

Most retry libraries are framework middleware or async-only. This one is
a small function you wrap around any callable. Sync and async loops
share the same `RetryPolicy`. Built-in retryable-code sets for Anthropic,
OpenAI, AWS Bedrock, and Google Gemini ship in `llm_retry.predicates`.

Sibling to the Rust crate
[`llm-retry`](https://crates.io/crates/llm-retry).

## Install

```bash
pip install llm-retry-py
```

## Use

Sync:

```python
from llm_retry import retry, RetryPolicy, predicates

def call_anthropic():
    return client.messages.create(model="claude-sonnet-4-5", ...)

resp = retry(
    call_anthropic,
    policy=RetryPolicy(),
    should_retry=lambda e: predicates.is_anthropic_retryable(str(e)),
)
```

Async (works with any asyncio code, no runtime lock-in beyond `asyncio.sleep`):

```python
from llm_retry import retry_async, RetryPolicy, predicates

async def call_openai():
    return await client.chat.completions.create(...)

resp = await retry_async(
    call_openai,
    policy=RetryPolicy(),
    should_retry=lambda e: predicates.is_openai_retryable(str(e)),
)
```

Custom predicate (BYO matcher):

```python
import httpx

def should_retry(e: Exception) -> bool:
    if isinstance(e, httpx.TimeoutException):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        return predicates.is_http_status_retryable(e.response.status_code)
    return False

resp = retry(call, policy=RetryPolicy(), should_retry=should_retry)
```

## RetryPolicy

Defaults: 6 attempts, 500ms base delay, 30s cap, full jitter.

```python
from llm_retry import RetryPolicy, Jitter

policy = RetryPolicy(
    max_attempts=8,
    base_delay_ms=250,
    max_delay_ms=60_000,
    jitter=Jitter.FULL,  # FULL (default, AWS-recommended), EQUAL, or NONE
)
```

Backoff for attempt `i` (0-indexed) is `min(base * 2**i, max)`, then jittered.

| Jitter  | Resulting delay                       |
| ------- | ------------------------------------- |
| `NONE`  | `capped`                              |
| `EQUAL` | `capped/2 + uniform(0, capped/2)`     |
| `FULL`  | `uniform(0, capped)` (recommended)    |

## Presets

```python
from llm_retry import predicates

predicates.is_anthropic_retryable("rate_limit_error")     # True
predicates.is_openai_retryable("server_error")            # True
predicates.is_bedrock_retryable("ThrottlingException")    # True
predicates.is_gemini_retryable("RESOURCE_EXHAUSTED")      # True
predicates.is_http_status_retryable(503)                  # True
```

The underlying lists are public constants you can extend:

```python
from llm_retry.predicates import ANTHROPIC_RETRYABLE, contains_any

my_codes = ANTHROPIC_RETRYABLE + ("my_custom_transient_code",)
should_retry = lambda e: contains_any(str(e), my_codes)
```

## Error type

A retry that does not succeed raises `RetryExhausted`:

```python
from llm_retry import RetryExhausted

try:
    resp = retry(call, policy=RetryPolicy(), should_retry=lambda e: True)
except RetryExhausted as exc:
    exc.attempts        # how many tries ran
    exc.last_error      # the final exception raised by `call`
```

If the predicate returns False for the first error, that error is
re-raised unchanged (no wrapping).

## What it does NOT do

- No HTTP client. Wrap any callable that raises.
- No circuit breaker. Layer one on top if you want.
- No deadline (`stop after N seconds total`). Combine with your own
  `asyncio.wait_for` or `signal.alarm`.
- No structured logging hooks. Add them in your callable.

## License

MIT

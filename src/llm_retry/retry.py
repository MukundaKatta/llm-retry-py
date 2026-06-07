"""Core retry loops: sync `retry()` + async `retry_async()`.

Both share `RetryPolicy` and `Jitter`. The predicate decides what counts
as transient; on a non-retryable error the original exception is
re-raised. On exhaustion, `RetryExhausted` is raised with the last
exception attached as `__cause__`.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class Jitter(Enum):
    """How much randomness to apply on top of the capped exponential delay.

    Values mirror the AWS "Exponential Backoff and Jitter" guidance.
    """

    # delay = capped (no randomness)
    NONE = "none"
    # delay = capped/2 + uniform(0, capped/2)
    EQUAL = "equal"
    # delay = uniform(0, capped). AWS-recommended default.
    FULL = "full"


@dataclass(frozen=True)
class RetryPolicy:
    """Tunables for a retry loop.

    Defaults: 6 attempts, 500ms base delay, 30s cap, full jitter.
    """

    # Total attempts including the first call (so 1 means "no retries").
    max_attempts: int = 6
    # Base delay in ms before the first retry sleep. Doubles each attempt.
    base_delay_ms: int = 500
    # Hard cap on a single sleep duration in ms.
    max_delay_ms: int = 30_000
    # Jitter strategy.
    jitter: Jitter = Jitter.FULL

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        if self.base_delay_ms < 0:
            raise ValueError("base_delay_ms must be >= 0")
        if self.max_delay_ms < 0:
            raise ValueError("max_delay_ms must be >= 0")

    def delay_seconds(
        self,
        attempt_index: int,
        rng: random.Random | None = None,
    ) -> float:
        """Compute the sleep delay (seconds) before retry `attempt_index`.

        `attempt_index` is 0-based: 0 is the first sleep after the first
        failure, 1 is the second sleep, etc.
        """
        if attempt_index < 0:
            raise ValueError("attempt_index must be >= 0")
        base = self.base_delay_ms / 1000.0
        cap = self.max_delay_ms / 1000.0
        # Guard 2**i against absurd attempt counts (clamp the exponent).
        # At i=30, 2**30 already overflows any realistic cap, so cap and break.
        exp = min(attempt_index, 30)
        capped = min(base * (2.0**exp), cap)
        r = rng if rng is not None else random
        if self.jitter is Jitter.NONE:
            return capped
        if self.jitter is Jitter.EQUAL:
            half = capped / 2.0
            return half + r.uniform(0.0, half)
        # FULL
        return r.uniform(0.0, capped)


class RetryExhausted(Exception):
    """Raised when all attempts failed and the predicate kept saying retry.

    Attributes:
        last_error: the exception raised by the final attempt
        attempts: total attempts run before giving up
    """

    def __init__(self, last_error: BaseException, attempts: int) -> None:
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(f"retry exhausted after {attempts} attempts: {last_error!r}")


# Back-compat alias matching the Rust enum name. The Rust crate has a
# single `RetryError` enum with `Exhausted` and `NotRetryable` arms; in
# Python we use exception hierarchy. `NotRetryable` becomes "the original
# exception just propagates", and `Exhausted` becomes `RetryExhausted`.
RetryError = RetryExhausted


def retry(
    op: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    should_retry: Callable[[BaseException], bool] = lambda _e: True,
    rng: random.Random | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `op` with retries on exceptions where `should_retry(e)` is True.

    On a non-retryable exception, the original is re-raised. On exhausting
    `policy.max_attempts`, raises `RetryExhausted` with `last_error` set.

    Blocks the calling thread during backoff. Use `retry_async` from
    within asyncio code.

    `rng` is provided for deterministic tests. `sleep` is injectable so
    tests can record sleep calls instead of actually waiting.
    """
    pol = policy if policy is not None else RetryPolicy()
    # `max(_, 1)` mirrors the Rust crate: even max_attempts=0 calls op once.
    attempts = max(pol.max_attempts, 1)

    last_err: BaseException | None = None
    for i in range(attempts):
        try:
            return op()
        except BaseException as e:  # noqa: BLE001 - intentional: user predicate decides
            if not should_retry(e):
                raise
            last_err = e
            if i + 1 < attempts:
                delay = pol.delay_seconds(i, rng=rng)
                if delay > 0:
                    sleep(delay)
    # We only reach here if every attempt raised and the predicate said retry.
    assert last_err is not None
    raise RetryExhausted(last_err, attempts) from last_err


async def retry_async(
    op: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    should_retry: Callable[[BaseException], bool] = lambda _e: True,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Async version of `retry`. Uses `asyncio.sleep` so it doesn't block the loop.

    `op` is a zero-arg callable that returns an awaitable, so each retry
    builds a fresh coroutine.
    """
    pol = policy if policy is not None else RetryPolicy()
    attempts = max(pol.max_attempts, 1)

    last_err: BaseException | None = None
    for i in range(attempts):
        try:
            return await op()
        except BaseException as e:  # noqa: BLE001 - intentional
            if not should_retry(e):
                raise
            last_err = e
            if i + 1 < attempts:
                delay = pol.delay_seconds(i, rng=rng)
                if delay > 0:
                    await sleep(delay)
    assert last_err is not None
    raise RetryExhausted(last_err, attempts) from last_err

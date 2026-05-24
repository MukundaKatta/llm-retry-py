"""Tests for llm_retry."""

from __future__ import annotations

import asyncio
import random

import pytest

from llm_retry import (
    Jitter,
    RetryError,
    RetryExhausted,
    RetryPolicy,
    predicates,
    retry,
    retry_async,
)

# ---------- helpers ----------


class RecordingSleep:
    """Drop-in for time.sleep that records calls instead of waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class AsyncRecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def fast_policy(max_attempts: int = 5) -> RetryPolicy:
    """Tiny-delay policy for fast unit tests. Uses NONE jitter so the
    recorded sleep amounts are deterministic (1ms per sleep)."""
    return RetryPolicy(
        max_attempts=max_attempts,
        base_delay_ms=1,
        max_delay_ms=1,
        jitter=Jitter.NONE,
    )


# ---------- RetryPolicy validation ----------


def test_policy_defaults_match_rust():
    p = RetryPolicy()
    assert p.max_attempts == 6
    assert p.base_delay_ms == 500
    assert p.max_delay_ms == 30_000
    assert p.jitter is Jitter.FULL


def test_policy_rejects_negative_values():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=-1)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_ms=-1)
    with pytest.raises(ValueError):
        RetryPolicy(max_delay_ms=-1)


def test_retry_error_is_alias_of_retry_exhausted():
    # Keep both names exported for users coming from the Rust crate.
    assert RetryError is RetryExhausted


# ---------- delay computation ----------


def test_delay_no_jitter_is_deterministic_exponential():
    p = RetryPolicy(
        max_attempts=10,
        base_delay_ms=1000,
        max_delay_ms=8000,
        jitter=Jitter.NONE,
    )
    assert p.delay_seconds(0) == pytest.approx(1.0)
    assert p.delay_seconds(1) == pytest.approx(2.0)
    assert p.delay_seconds(2) == pytest.approx(4.0)
    assert p.delay_seconds(3) == pytest.approx(8.0)
    # capped at max
    assert p.delay_seconds(5) == pytest.approx(8.0)


def test_delay_full_jitter_stays_in_bounds():
    p = RetryPolicy(
        max_attempts=10,
        base_delay_ms=1000,
        max_delay_ms=4000,
        jitter=Jitter.FULL,
    )
    rng = random.Random(42)
    for i in range(6):
        d = p.delay_seconds(i, rng=rng)
        upper = min(1.0 * (2**i), 4.0)
        assert 0.0 <= d <= upper + 1e-9


def test_delay_equal_jitter_is_at_least_half():
    p = RetryPolicy(
        max_attempts=10,
        base_delay_ms=2000,
        max_delay_ms=2000,
        jitter=Jitter.EQUAL,
    )
    rng = random.Random(7)
    for _ in range(20):
        d = p.delay_seconds(0, rng=rng)
        # capped at 2.0, half = 1.0 -> d in [1.0, 2.0]
        assert 1.0 - 1e-9 <= d <= 2.0 + 1e-9


def test_delay_attempt_index_negative_raises():
    with pytest.raises(ValueError):
        RetryPolicy().delay_seconds(-1)


def test_delay_huge_attempt_index_does_not_overflow():
    # 2**1000 would overflow a float; we clamp the exponent.
    p = RetryPolicy(base_delay_ms=1000, max_delay_ms=5000, jitter=Jitter.NONE)
    assert p.delay_seconds(1000) == pytest.approx(5.0)


# ---------- sync retry ----------


def test_retry_returns_on_first_success():
    calls = [0]

    def op():
        calls[0] += 1
        return 42

    sleeper = RecordingSleep()
    result = retry(op, policy=fast_policy(), sleep=sleeper)
    assert result == 42
    assert calls[0] == 1
    # success on first try -> no sleeps
    assert sleeper.calls == []


def test_retry_retries_until_success():
    calls = [0]

    def op():
        calls[0] += 1
        if calls[0] < 3:
            raise RuntimeError("transient")
        return 7

    sleeper = RecordingSleep()
    result = retry(op, policy=fast_policy(), sleep=sleeper)
    assert result == 7
    assert calls[0] == 3
    # Two failures -> two sleeps (between the three attempts).
    assert len(sleeper.calls) == 2


def test_retry_exhausts_after_max_attempts():
    calls = [0]

    def op():
        calls[0] += 1
        raise RuntimeError("nope")

    sleeper = RecordingSleep()
    with pytest.raises(RetryExhausted) as exc:
        retry(op, policy=fast_policy(max_attempts=5), sleep=sleeper)
    assert calls[0] == 5
    assert exc.value.attempts == 5
    assert isinstance(exc.value.last_error, RuntimeError)
    assert str(exc.value.last_error) == "nope"
    # We sleep between attempts but not after the final one.
    assert len(sleeper.calls) == 4


def test_retry_non_retryable_propagates_immediately():
    calls = [0]

    class Fatal(Exception):
        pass

    def op():
        calls[0] += 1
        raise Fatal("bad credentials")

    sleeper = RecordingSleep()
    with pytest.raises(Fatal):
        retry(
            op,
            policy=fast_policy(),
            should_retry=lambda e: not isinstance(e, Fatal),
            sleep=sleeper,
        )
    assert calls[0] == 1
    assert sleeper.calls == []


def test_retry_max_attempts_zero_still_calls_op_once():
    # Mirrors Rust crate edge-case: max(_, 1) inside the loop.
    calls = [0]

    def op():
        calls[0] += 1
        raise RuntimeError("x")

    with pytest.raises(RetryExhausted):
        retry(op, policy=fast_policy(max_attempts=0), sleep=RecordingSleep())
    assert calls[0] == 1


def test_retry_predicate_can_inspect_exception_attrs():
    class StatusError(Exception):
        def __init__(self, status: int) -> None:
            self.status = status
            super().__init__(f"http {status}")

    calls = [0]

    def op():
            calls[0] += 1
            if calls[0] < 2:
                raise StatusError(503)
            return "ok"

    result = retry(
        op,
        policy=fast_policy(),
        should_retry=lambda e: isinstance(e, StatusError)
        and predicates.is_http_status_retryable(e.status),
        sleep=RecordingSleep(),
    )
    assert result == "ok"


# ---------- async retry ----------


async def test_async_returns_on_first_success():
    async def op():
        return "ok"

    sleeper = AsyncRecordingSleep()
    result = await retry_async(op, policy=fast_policy(), sleep=sleeper)
    assert result == "ok"
    assert sleeper.calls == []


async def test_async_retries_then_succeeds():
    calls = [0]

    async def op():
        calls[0] += 1
        if calls[0] < 3:
            raise RuntimeError("flaky")
        return 99

    sleeper = AsyncRecordingSleep()
    result = await retry_async(op, policy=fast_policy(), sleep=sleeper)
    assert result == 99
    assert calls[0] == 3
    assert len(sleeper.calls) == 2


async def test_async_exhausts():
    async def op():
        raise RuntimeError("always")

    sleeper = AsyncRecordingSleep()
    with pytest.raises(RetryExhausted) as exc:
        await retry_async(op, policy=fast_policy(max_attempts=4), sleep=sleeper)
    assert exc.value.attempts == 4
    assert len(sleeper.calls) == 3


async def test_async_non_retryable_propagates():
    class Fatal(Exception):
        pass

    async def op():
        raise Fatal("auth")

    with pytest.raises(Fatal):
        await retry_async(
            op,
            policy=fast_policy(),
            should_retry=lambda e: not isinstance(e, Fatal),
            sleep=AsyncRecordingSleep(),
        )


async def test_async_actually_uses_asyncio_sleep_default():
    # Default sleep is asyncio.sleep; a tiny delay should not block forever.
    calls = [0]

    async def op():
        calls[0] += 1
        if calls[0] < 2:
            raise RuntimeError("flaky")
        return "ok"

    policy = RetryPolicy(max_attempts=3, base_delay_ms=1, max_delay_ms=1, jitter=Jitter.NONE)
    result = await asyncio.wait_for(retry_async(op, policy=policy), timeout=1.0)
    assert result == "ok"


# ---------- predicates: Anthropic ----------


def test_anthropic_predicate_matches_codes():
    assert predicates.is_anthropic_retryable("rate_limit_error")
    assert predicates.is_anthropic_retryable("overloaded_error")
    assert predicates.is_anthropic_retryable("api_error")
    assert predicates.is_anthropic_retryable("timeout")
    # substring match
    assert predicates.is_anthropic_retryable(
        "Error: rate_limit_error occurred for org_xxx"
    )


def test_anthropic_predicate_rejects_unsafe():
    assert not predicates.is_anthropic_retryable("authentication_error")
    assert not predicates.is_anthropic_retryable("invalid_request_error")
    assert not predicates.is_anthropic_retryable("permission_error")


# ---------- predicates: OpenAI ----------


def test_openai_predicate_matches_codes():
    assert predicates.is_openai_retryable("rate_limit_exceeded")
    assert predicates.is_openai_retryable("server_error")
    assert predicates.is_openai_retryable("engine_overloaded")
    assert predicates.is_openai_retryable("tokens_exhausted")
    assert predicates.is_openai_retryable("timeout")


def test_openai_predicate_rejects_unsafe():
    assert not predicates.is_openai_retryable("invalid_api_key")
    assert not predicates.is_openai_retryable("insufficient_quota")


# ---------- predicates: Bedrock ----------


def test_bedrock_predicate_matches_codes():
    assert predicates.is_bedrock_retryable("ThrottlingException")
    assert predicates.is_bedrock_retryable("Throttling")
    assert predicates.is_bedrock_retryable("TooManyRequestsException")
    assert predicates.is_bedrock_retryable("ServiceUnavailableException")
    assert predicates.is_bedrock_retryable("ProvisionedThroughputExceededException")
    assert predicates.is_bedrock_retryable("ModelTimeoutException")


def test_bedrock_predicate_rejects_unsafe():
    assert not predicates.is_bedrock_retryable("ValidationException")
    assert not predicates.is_bedrock_retryable("AccessDeniedException")


# ---------- predicates: Gemini ----------


def test_gemini_predicate_matches_codes():
    assert predicates.is_gemini_retryable("RESOURCE_EXHAUSTED")
    assert predicates.is_gemini_retryable("UNAVAILABLE")
    assert predicates.is_gemini_retryable("DEADLINE_EXCEEDED")
    assert predicates.is_gemini_retryable("INTERNAL")


def test_gemini_predicate_rejects_unsafe():
    assert not predicates.is_gemini_retryable("PERMISSION_DENIED")
    assert not predicates.is_gemini_retryable("INVALID_ARGUMENT")


# ---------- predicates: HTTP status ----------


def test_http_status_retryable():
    for s in (408, 425, 429, 500, 502, 503, 504):
        assert predicates.is_http_status_retryable(s)
    for s in (200, 201, 204, 301, 400, 401, 403, 404, 422):
        assert not predicates.is_http_status_retryable(s)


# ---------- predicates: contains_any ----------


def test_contains_any_works_with_lists_and_tuples():
    assert predicates.contains_any("hello world", ["world"])
    assert predicates.contains_any("hello world", ("world",))
    assert not predicates.contains_any("hello", ["WORLD"])  # case-sensitive
    assert not predicates.contains_any("", ["x"])
    # empty pattern list
    assert not predicates.contains_any("hello", [])


# ---------- end-to-end integration with predicates ----------


def test_retry_with_anthropic_preset_recovers_from_rate_limit():
    calls = [0]

    def op():
        calls[0] += 1
        if calls[0] < 3:
            raise RuntimeError("rate_limit_error: backoff please")
        return "claude-response"

    result = retry(
        op,
        policy=fast_policy(),
        should_retry=lambda e: predicates.is_anthropic_retryable(str(e)),
        sleep=RecordingSleep(),
    )
    assert result == "claude-response"
    assert calls[0] == 3


def test_retry_with_anthropic_preset_does_not_retry_auth_error():
    def op():
        raise RuntimeError("authentication_error: bad key")

    with pytest.raises(RuntimeError, match="authentication_error"):
        retry(
            op,
            policy=fast_policy(),
            should_retry=lambda e: predicates.is_anthropic_retryable(str(e)),
            sleep=RecordingSleep(),
        )

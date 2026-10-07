"""Tests for telemetry event types and emit_event helper."""

from __future__ import annotations

from tensorcost import (
    BeforeRequestEvent,
    AfterResponseEvent,
    OnRetryEvent,
    OnErrorEvent,
    OnFallbackEvent,
    LifecycleEvent,
    LifecycleEventCallback,
)
from tensorcost._telemetry import emit_event


class TestEventTypes:
    def test_before_request_event(self):
        e = BeforeRequestEvent(
            kind="before_request",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=1,
            elapsed_ms=0.0,
        )
        assert e.kind == "before_request"
        assert e.provider == "openai"
        assert e.attempt_number == 1

    def test_after_response_event(self):
        e = AfterResponseEvent(
            kind="after_response",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=1,
            elapsed_ms=123.4,
            status=200,
            request_id="req-abc",
        )
        assert e.status == 200
        assert e.request_id == "req-abc"

    def test_on_retry_event(self):
        e = OnRetryEvent(
            kind="on_retry",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            operation="messages",
            attempt_number=2,
            elapsed_ms=50.0,
            reason="5xx",
            status=503,
            delay_ms=500.0,
        )
        assert e.reason == "5xx"
        assert e.status == 503
        assert e.delay_ms == 500.0

    def test_on_error_event(self):
        e = OnErrorEvent(
            kind="on_error",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=5,
            elapsed_ms=3000.0,
            error_class="TensorCostProxyError",
            status=502,
        )
        assert e.error_class == "TensorCostProxyError"

    def test_on_fallback_event(self):
        e = OnFallbackEvent(
            kind="on_fallback",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=1,
            elapsed_ms=0.0,
            consecutive_failures=3,
        )
        assert e.consecutive_failures == 3

    def test_events_are_frozen(self):
        import pytest
        e = BeforeRequestEvent(
            kind="before_request",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=1,
            elapsed_ms=0.0,
        )
        with pytest.raises(Exception):
            e.attempt_number = 2  # type: ignore[misc]


class TestEmitEvent:
    def test_calls_callback_with_event(self):
        received = []
        event = BeforeRequestEvent(
            kind="before_request",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=1,
            elapsed_ms=0.0,
        )

        def cb(e: LifecycleEvent) -> None:
            received.append(e)

        emit_event(cb, event)
        assert received == [event]

    def test_swallows_callback_exceptions(self):
        event = BeforeRequestEvent(
            kind="before_request",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=1,
            elapsed_ms=0.0,
        )

        def bad_cb(e: LifecycleEvent) -> None:
            raise RuntimeError("callback blew up")

        # Should not raise.
        emit_event(bad_cb, event)

    def test_none_callback_is_noop(self):
        event = BeforeRequestEvent(
            kind="before_request",
            provider="openai",
            model="gpt-4o",
            operation="chat.completions",
            attempt_number=1,
            elapsed_ms=0.0,
        )
        # No exception.
        emit_event(None, event)

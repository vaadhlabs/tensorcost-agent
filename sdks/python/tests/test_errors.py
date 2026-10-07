"""Tests for the typed error hierarchy."""

from __future__ import annotations

import pytest

from tensorcost import (
    TensorCostError,
    TensorCostNetworkError,
    TensorCostTimeoutError,
    TensorCostProxyError,
    TensorCostQuotaError,
    TensorCostProviderError,
)


class TestErrorHierarchy:
    def test_all_are_subclasses_of_base(self):
        for cls in (
            TensorCostNetworkError,
            TensorCostTimeoutError,
            TensorCostProxyError,
            TensorCostQuotaError,
            TensorCostProviderError,
        ):
            assert issubclass(cls, TensorCostError)

    def test_all_are_exceptions(self):
        for cls in (
            TensorCostError,
            TensorCostNetworkError,
            TensorCostTimeoutError,
            TensorCostProxyError,
            TensorCostQuotaError,
            TensorCostProviderError,
        ):
            assert issubclass(cls, Exception)


class TestTensorCostError:
    def test_message_is_str_value(self):
        err = TensorCostError("something went wrong")
        assert str(err) == "something went wrong"

    def test_defaults(self):
        err = TensorCostError("msg")
        assert err.root_cause is None
        assert err.status is None
        assert err.request_id is None
        assert err.attempt == 1

    def test_all_fields_stored(self):
        root = ValueError("root")
        err = TensorCostError(
            "msg",
            root_cause=root,
            status=503,
            request_id="req-abc",
            attempt=3,
        )
        assert err.root_cause is root
        assert err.status == 503
        assert err.request_id == "req-abc"
        assert err.attempt == 3

    def test_repr_includes_status_and_attempt(self):
        err = TensorCostError("oops", status=500, attempt=2)
        r = repr(err)
        assert "TensorCostError" in r
        assert "500" in r
        assert "2" in r

    def test_can_be_raised_and_caught(self):
        with pytest.raises(TensorCostError):
            raise TensorCostError("boom")


class TestTensorCostTimeoutError:
    def test_timeout_kind_stored(self):
        err = TensorCostTimeoutError("timed out", "total")
        assert err.timeout_kind == "total"

    def test_idle_kind(self):
        err = TensorCostTimeoutError("idle timed out", "idle")
        assert err.timeout_kind == "idle"

    def test_is_base_subclass(self):
        err = TensorCostTimeoutError("t", "total")
        assert isinstance(err, TensorCostError)

    def test_all_base_fields_work(self):
        rc = RuntimeError("underlying")
        err = TensorCostTimeoutError(
            "msg", "total",
            root_cause=rc,
            status=None,
            request_id="r1",
            attempt=2,
        )
        assert err.root_cause is rc
        assert err.request_id == "r1"
        assert err.attempt == 2


class TestTensorCostQuotaError:
    def test_retry_after_ms_stored(self):
        err = TensorCostQuotaError("rate limited", retry_after_ms=5000)
        assert err.retry_after_ms == 5000

    def test_default_status_is_429(self):
        err = TensorCostQuotaError("rate limited")
        assert err.status == 429

    def test_retry_after_ms_none_when_absent(self):
        err = TensorCostQuotaError("rate limited")
        assert err.retry_after_ms is None

    def test_is_base_subclass(self):
        assert isinstance(TensorCostQuotaError("q"), TensorCostError)


class TestNonRetryableErrors:
    def test_network_error_stores_root_cause(self):
        cause = ConnectionError("no route")
        err = TensorCostNetworkError("network fail", root_cause=cause)
        assert err.root_cause is cause

    def test_proxy_error_stores_status(self):
        err = TensorCostProxyError("proxy 503", status=503)
        assert err.status == 503

    def test_provider_error_stores_status(self):
        err = TensorCostProviderError("provider 401", status=401)
        assert err.status == 401

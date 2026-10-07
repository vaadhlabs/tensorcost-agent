"""Tests for src/tracing.py — OpenTelemetry init/shutdown surface.

These exercise the module's public surface only:
  - OTEL_ENABLED gate (off by default → no-op).
  - happy-path init when OTEL_ENABLED=true with OTel SDK importable.
  - shutdown_tracing() after init flushes cleanly and is safe to call
    twice (or before init at all).
  - import-time failure (simulated ImportError on the SDK packages)
    is caught and logged, NOT propagated.

We deliberately do NOT exercise the actual span exporter — that's an
integration concern. We just verify the module mutates the global
tracer-provider state correctly.
"""

import importlib
import os
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

import src.tracing as tracing_mod


def _inject_fake_otel(monkeypatch, *,
                      tp_factory=None,
                      exporter_factory=None,
                      bsp_factory=None,
                      grpc_instrumentor=None,
                      requests_instrumentor=None):
    """Wire up a minimal fake `opentelemetry.*` module tree so init_tracing()
    can import successfully.

    We inject into sys.modules rather than monkey-patching attributes,
    because the real package isn't installed in the test environment —
    `import opentelemetry...` would otherwise hit ModuleNotFoundError
    long before any patch could take effect.
    """
    tp_factory = tp_factory or MagicMock(return_value=MagicMock())
    exporter_factory = exporter_factory or MagicMock(return_value=MagicMock())
    bsp_factory = bsp_factory or MagicMock(return_value=MagicMock())

    # opentelemetry root + .trace
    otel = types.ModuleType("opentelemetry")
    otel_trace = types.ModuleType("opentelemetry.trace")
    otel_trace.set_tracer_provider = MagicMock()
    otel.trace = otel_trace

    # opentelemetry.sdk.{trace,resources}
    otel_sdk = types.ModuleType("opentelemetry.sdk")
    otel_sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    otel_sdk_trace.TracerProvider = tp_factory
    otel_sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    otel_sdk_trace_export.BatchSpanProcessor = bsp_factory
    otel_sdk.trace = otel_sdk_trace

    otel_sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    otel_sdk_resources.Resource = MagicMock()
    otel_sdk_resources.Resource.create = MagicMock(return_value=MagicMock())
    otel_sdk_resources.SERVICE_NAME = "service.name"
    otel_sdk_resources.SERVICE_VERSION = "service.version"

    # opentelemetry.exporter.otlp.proto.grpc.trace_exporter
    otel_exp = types.ModuleType("opentelemetry.exporter")
    otel_exp_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    otel_exp_otlp_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    otel_exp_otlp_proto_grpc = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc")
    otel_exp_otlp_proto_grpc_te = types.ModuleType(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    )
    otel_exp_otlp_proto_grpc_te.OTLPSpanExporter = exporter_factory

    # opentelemetry.instrumentation.{grpc,requests} — optional
    otel_instr = types.ModuleType("opentelemetry.instrumentation")
    otel_instr_grpc = types.ModuleType("opentelemetry.instrumentation.grpc")
    otel_instr_grpc.GrpcInstrumentorClient = grpc_instrumentor or MagicMock(
        return_value=MagicMock()
    )
    otel_instr_requests = types.ModuleType("opentelemetry.instrumentation.requests")
    otel_instr_requests.RequestsInstrumentor = requests_instrumentor or MagicMock(
        return_value=MagicMock()
    )

    monkeypatch.setitem(sys.modules, "opentelemetry", otel)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", otel_trace)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", otel_sdk)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", otel_sdk_trace)
    monkeypatch.setitem(
        sys.modules, "opentelemetry.sdk.trace.export", otel_sdk_trace_export
    )
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.resources", otel_sdk_resources)
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter", otel_exp)
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp", otel_exp_otlp)
    monkeypatch.setitem(
        sys.modules, "opentelemetry.exporter.otlp.proto", otel_exp_otlp_proto
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.grpc",
        otel_exp_otlp_proto_grpc,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        otel_exp_otlp_proto_grpc_te,
    )
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation", otel_instr)
    monkeypatch.setitem(
        sys.modules, "opentelemetry.instrumentation.grpc", otel_instr_grpc
    )
    monkeypatch.setitem(
        sys.modules, "opentelemetry.instrumentation.requests", otel_instr_requests
    )

    return {
        "set_tracer_provider": otel_trace.set_tracer_provider,
        "TracerProvider": tp_factory,
        "OTLPSpanExporter": exporter_factory,
        "BatchSpanProcessor": bsp_factory,
        "GrpcInstrumentorClient": otel_instr_grpc.GrpcInstrumentorClient,
        "RequestsInstrumentor": otel_instr_requests.RequestsInstrumentor,
    }


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    """Reset module-level state between tests so each one starts fresh.

    The OpenTelemetry global trace provider is also process-global, but
    re-running init_tracing() against an already-set provider is the
    documented "warn and proceed" path on OTel's side. We just make sure
    OUR module-level cache is clean."""
    tracing_mod._tracer_provider = None
    yield
    tracing_mod._tracer_provider = None


class TestInitTracingDisabled:
    """OTEL_ENABLED unset / != 'true' should be a clean no-op."""

    def test_init_noop_when_otel_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_ENABLED", None)
            tracing_mod.init_tracing()
        assert tracing_mod._tracer_provider is None

    def test_init_noop_when_otel_explicitly_false(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            tracing_mod.init_tracing()
        assert tracing_mod._tracer_provider is None

    def test_init_noop_when_otel_unrecognised_value(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "yes"}):
            # Only the literal "true" enables; anything else is off.
            tracing_mod.init_tracing()
        assert tracing_mod._tracer_provider is None

    def test_init_case_insensitive_true(self, monkeypatch):
        # The check lowercases the value; "TRUE" must enable.
        _inject_fake_otel(monkeypatch)
        monkeypatch.setenv("OTEL_ENABLED", "TRUE")
        tracing_mod.init_tracing()
        assert tracing_mod._tracer_provider is not None


class TestInitTracingEnabled:
    """OTEL_ENABLED=true → tracer provider is created and registered."""

    def test_init_sets_tracer_provider_on_global(self, monkeypatch):
        fakes = _inject_fake_otel(monkeypatch)
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "test-svc")
        monkeypatch.setenv("AGENT_VERSION", "9.9.9")
        monkeypatch.setenv("ENVIRONMENT", "test")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")

        tracing_mod.init_tracing()

        assert tracing_mod._tracer_provider is not None
        # The module reaches into the global trace API to register.
        fakes["set_tracer_provider"].assert_called_once_with(tracing_mod._tracer_provider)
        # The provider was constructed with a Resource kwarg.
        fakes["TracerProvider"].assert_called_once()
        kwargs = fakes["TracerProvider"].call_args.kwargs
        assert "resource" in kwargs
        # Span processor was attached.
        tracing_mod._tracer_provider.add_span_processor.assert_called_once()

    def test_init_uses_endpoint_from_env(self, monkeypatch):
        fakes = _inject_fake_otel(monkeypatch)
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
        tracing_mod.init_tracing()
        fakes["OTLPSpanExporter"].assert_called_once()
        ekwargs = fakes["OTLPSpanExporter"].call_args.kwargs
        assert ekwargs.get("endpoint") == "http://otel:4317"
        assert ekwargs.get("insecure") is True

    def test_init_uses_default_endpoint_when_unset(self, monkeypatch):
        fakes = _inject_fake_otel(monkeypatch)
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        tracing_mod.init_tracing()
        ekwargs = fakes["OTLPSpanExporter"].call_args.kwargs
        # Default endpoint points at the local OTel collector.
        assert ekwargs.get("endpoint") == "http://localhost:4317"

    def test_init_invokes_optional_instrumentors(self, monkeypatch):
        fakes = _inject_fake_otel(monkeypatch)
        monkeypatch.setenv("OTEL_ENABLED", "true")
        tracing_mod.init_tracing()
        # Both optional instrumentors get instantiated AND .instrument()-ed.
        fakes["GrpcInstrumentorClient"].assert_called_once()
        fakes["RequestsInstrumentor"].assert_called_once()
        fakes["GrpcInstrumentorClient"].return_value.instrument.assert_called_once()
        fakes["RequestsInstrumentor"].return_value.instrument.assert_called_once()

    def test_init_swallows_unexpected_exception(self, monkeypatch):
        """A non-ImportError raised mid-setup is logged, not propagated."""
        bad_tp = MagicMock(side_effect=RuntimeError("boom"))
        _inject_fake_otel(monkeypatch, tp_factory=bad_tp)
        monkeypatch.setenv("OTEL_ENABLED", "true")
        # Must not raise — failure is caught and logged at error.
        tracing_mod.init_tracing()
        assert tracing_mod._tracer_provider is None

    def test_init_swallows_import_error(self, monkeypatch):
        """If the OTel SDK isn't installed, init must log+continue."""
        # Don't inject — the modules genuinely aren't there. The module's
        # `from opentelemetry...` chain raises ImportError, which the
        # except-block catches and warns.
        for k in list(sys.modules):
            if k == "opentelemetry" or k.startswith("opentelemetry."):
                monkeypatch.delitem(sys.modules, k, raising=False)
        monkeypatch.setenv("OTEL_ENABLED", "true")
        tracing_mod.init_tracing()
        assert tracing_mod._tracer_provider is None


class TestShutdownTracing:
    """shutdown_tracing() must be safe in every state."""

    def test_shutdown_when_never_initialised(self):
        """Calling shutdown without a prior init is a no-op (no AttributeError)."""
        tracing_mod._tracer_provider = None
        tracing_mod.shutdown_tracing()
        # Nothing to assert beyond "did not raise".

    def test_shutdown_calls_provider_shutdown(self):
        mock_provider = MagicMock()
        tracing_mod._tracer_provider = mock_provider
        tracing_mod.shutdown_tracing()
        mock_provider.shutdown.assert_called_once()

    def test_shutdown_swallows_provider_error(self):
        """A misbehaving exporter on shutdown must not propagate to caller."""
        mock_provider = MagicMock()
        mock_provider.shutdown.side_effect = RuntimeError("stuck pipe")
        tracing_mod._tracer_provider = mock_provider
        # Must not raise.
        tracing_mod.shutdown_tracing()
        mock_provider.shutdown.assert_called_once()

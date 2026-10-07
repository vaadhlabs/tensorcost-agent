"""
OpenTelemetry Distributed Tracing Setup

Must be initialized before any instrumented libraries are imported.
Gated on OTEL_ENABLED=true — when disabled, init_tracing() is a no-op.
"""

import os
import logging

logger = logging.getLogger(__name__)

_tracer_provider = None


def init_tracing():
    """Initialize OpenTelemetry tracing if OTEL_ENABLED is set."""
    global _tracer_provider

    if os.getenv('OTEL_ENABLED', '').lower() != 'true':
        logger.debug("OpenTelemetry tracing disabled (OTEL_ENABLED != true)")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

        resource = Resource.create({
            SERVICE_NAME: os.getenv('OTEL_SERVICE_NAME', 'unified-gpu-agent'),
            SERVICE_VERSION: os.getenv('AGENT_VERSION', '1.0.0'),
            'deployment.environment': os.getenv('ENVIRONMENT', 'development'),
        })

        endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)

        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_tracer_provider)

        # Auto-instrument gRPC and requests if available
        try:
            from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
            GrpcInstrumentorClient().instrument()
        except ImportError:
            logger.debug("gRPC instrumentation not available")

        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            RequestsInstrumentor().instrument()
        except ImportError:
            logger.debug("requests instrumentation not available")

        logger.info(f"OpenTelemetry tracing initialized, exporting to {endpoint}")

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not installed, tracing disabled: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")


def shutdown_tracing():
    """Flush and shut down the tracer provider."""
    global _tracer_provider
    if _tracer_provider:
        try:
            _tracer_provider.shutdown()
            logger.info("OpenTelemetry tracing shut down")
        except Exception as e:
            logger.warning(f"Error shutting down tracing: {e}")

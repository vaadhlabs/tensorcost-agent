"""Internal type definitions for observation payloads."""

from __future__ import annotations

from typing import Optional, TypedDict


class Observation(TypedDict, total=False):
    """The wire shape posted to /api/proxy/observation.

    Per TIER-2-SDK-PLAN §6. Fields are flat strings/ints to keep the
    backend ingestion fast.
    """

    sdk_version: str
    provider: str  # "openai" | "anthropic" | "bedrock" | "vertex"
    model: str
    operation: str  # "chat.completions" | "completions" | "messages"
    modality: Optional[str]  # "text" | "image" | "audio" | "video" | "mixed"
    request_at: str  # ISO-8601
    response_at: str  # ISO-8601
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd_cents: Optional[int]  # SDK doesn't compute; backend fills
    status: str  # "success" | "error"
    error_message: Optional[str]
    correlation_id: str  # uuid4
    tenant_id: Optional[str]
    # Phase A4 batch 3 — per-environment data scoping. Optional; the
    # SDK omits the field when ``wrap()`` was not given an
    # ``environment`` kwarg, and the backend then falls back to the
    # column default ('production'). Free-text per
    # project_environment_scoping.md (1..64 chars).
    environment: Optional[str]
    # Phase A4 batch 4 — provider-connection identifier. Optional; the
    # SDK omits the field when ``wrap()`` was not given a
    # ``connection_id`` kwarg. When present, ai-service uses it as the
    # second tier of its env-resolution chain (look up
    # `integration.cloud_account.environment` for the row whose `id`
    # matches, behind a per-process 60s LRU cache).
    connection_id: Optional[str]
    # A-04 chargeback — end-customer cohort. Optional; lands in
    # ``ai.ai_spend_events.customer``.
    customer: Optional[str]
    # A-04 chargeback — product feature. Optional; lands in
    # ``ai.ai_spend_events.feature``.
    feature: Optional[str]
    # Agent-run attribution — optional; lands in ``ai.ai_spend_events.agent_id``.
    agent_id: Optional[str]
    # Agent-run attribution — optional; lands in ``ai.ai_spend_events.workflow_id``.
    workflow_id: Optional[str]

"""1.5.0: the fail-open rule is an allow-list of outages.

1.4.0 decided "may this call go around TensorCost?" with a deny-list of
refusal classes. A money budget's hard cap (BUDGET_HARD_CAP) wasn't on it, so
the proxy refused, the SDK called the provider anyway, and the cap was a log
line. These tests drive the real OpenAI wrapper (proxy path, streaming proxy
path, govern/admit path) against stubbed TensorCost responses (httpx
MockTransport) and pin, for every refusal shape, that TensorCost is asked
exactly once, the provider is never called, and the caller gets the right
typed error with the server's code and reason intact. Then the other side:
genuine outages still fail open when ``fail_open_enabled`` is on, and don't
when it's off.
"""

from __future__ import annotations

import logging
from typing import Any, Callable
from unittest.mock import MagicMock

import httpx
import pytest

from tensorcost import (
    TensorCostBudgetHardCapError,
    TensorCostComplianceDeniedError,
    TensorCostGuardrailHardStopError,
    TensorCostModelGovernanceDeniedError,
    TensorCostPeriodBudgetExceededError,
    TensorCostProviderError,
    TensorCostProxyError,
    TensorCostRefusedError,
    TensorCostRunBudgetExceededError,
    TensorCostRunStoppedError,
)
from tensorcost._circuit import CircuitBreaker
from tensorcost._providers import _openai
from tensorcost._proxy_client import should_fall_back_to_provider
from tensorcost._retry import RetryConfig

# The money-budget bodies the proxy sends, as pinned by the server's contract
# test (inference-proxy-service src/__tests__/money-budget-wire-contract.spec.ts,
# monorepo 7046d775): MoneyBudgetHardCapError("b-1", 1.02, "Prod AI", None,
# {monthly, 10000, 10200}) and ("b-2", 0, "New budget", "spend_not_known_yet",
# {weekly, 5000}). The second goes out with Retry-After: 5.
MONEY_BUDGET_REACHED = {
    "error": {
        "code": "PERIOD_BUDGET_EXCEEDED",
        "budget_code": "BUDGET_HARD_CAP",
        "message": 'Budget hard cap reached for budget "Prod AI" (b-1)',
        "scope": "budget:b-1",
        "period": "monthly",
        "cap_cents": 10_000,
        "spent_cents": 10_200,
        "budget_id": "b-1",
        "budget_name": "Prod AI",
        "burn_pct": 1.02,
        "retryable": False,
    }
}
MONEY_BUDGET_NOT_KNOWN_YET = {
    "error": {
        "code": "PERIOD_BUDGET_EXCEEDED",
        "budget_code": "BUDGET_HARD_CAP",
        "message": (
            'The spend for budget "New budget" (b-2) hasn\'t been counted yet, so its hard cap '
            "holds requests until it is (usually seconds after the budget is created or its scope changes)"
        ),
        "scope": "budget:b-2",
        "period": "weekly",
        "cap_cents": 5_000,
        "budget_id": "b-2",
        "budget_name": "New budget",
        "burn_pct": 0,
        "reason": "spend_not_known_yet",
        "retryable": True,
        "retry_after_seconds": 5,
    }
}

CASES: list[dict[str, Any]] = [
    dict(
        id="RUN_STOPPED",
        body={
            "error": {
                "code": "RUN_STOPPED",
                "message": "Agent run run-abc was stopped by ops@example.test: runaway",
                "workflow_id": "run-abc",
                "stopped_by": "ops@example.test",
                "stopped_at": "2026-10-25T10:00:00.000Z",
                "reason": "runaway",
            }
        },
        cls=TensorCostRunStoppedError,
        code="RUN_STOPPED",
        reason="runaway",
        extra={"workflow_id": "run-abc", "stopped_by": "ops@example.test"},
    ),
    dict(
        id="RUN_BUDGET_EXCEEDED",
        body={
            "error": {
                "code": "RUN_BUDGET_EXCEEDED",
                "message": "Agent run run-abc has reached its spend cap",
                "workflow_id": "run-abc",
                "cap_cents": 100,
                "spent_cents": 150,
            }
        },
        cls=TensorCostRunBudgetExceededError,
        code="RUN_BUDGET_EXCEEDED",
        reason=None,
        extra={"workflow_id": "run-abc", "cap_cents": 100, "spent_cents": 150},
    ),
    dict(
        id="PERIOD_BUDGET_EXCEEDED",
        body={
            "error": {
                "code": "PERIOD_BUDGET_EXCEEDED",
                "message": "Agent period budget exceeded",
                "scope": "agent:a1",
                "period": "day",
                "cap_cents": 500,
                "spent_cents": 600,
            }
        },
        cls=TensorCostPeriodBudgetExceededError,
        code="PERIOD_BUDGET_EXCEEDED",
        reason=None,
        extra={"scope": "agent:a1", "period": "day", "cap_cents": 500},
    ),
    dict(
        id="BUDGET_HARD_CAP-older-server-code-reached",
        body={
            "error": {
                "code": "BUDGET_HARD_CAP",
                "message": "Budget hard cap reached for budget b1",
                "budget_id": "b1",
                "burn_pct": 104,
            }
        },
        cls=TensorCostBudgetHardCapError,
        code="BUDGET_HARD_CAP",
        reason=None,
        extra={"budget_id": "b1", "burn_pct": 104, "retryable": False, "retry_after_seconds": None},
        message="the budget's hard cap is reached (budget b1)",
    ),
    dict(
        id="BUDGET_HARD_CAP-older-server-code-not-known-yet",
        body={
            "error": {
                "code": "BUDGET_HARD_CAP",
                "message": "The spend for budget \"Prod\" (b1) hasn't been counted yet...",
                "budget_id": "b1",
                "budget_name": "Prod",
                "burn_pct": 0,
                "reason": "spend_not_known_yet",
            }
        },
        headers={"retry-after": "5"},
        cls=TensorCostBudgetHardCapError,
        code="BUDGET_HARD_CAP",
        reason="spend_not_known_yet",
        extra={"budget_id": "b1", "budget_name": "Prod", "retryable": True, "retry_after_seconds": 5},
        message="the budget's spend hasn't been counted yet; retry in 5 s",
    ),
    dict(
        id="BUDGET_HARD_CAP-server-shape-reached",
        body=MONEY_BUDGET_REACHED,
        cls=TensorCostBudgetHardCapError,
        code="PERIOD_BUDGET_EXCEEDED",
        reason=None,
        extra={
            "budget_code": "BUDGET_HARD_CAP",
            "budget_id": "b-1",
            "budget_name": "Prod AI",
            "scope": "budget:b-1",
            "period": "monthly",
            "cap_cents": 10_000,
            "spent_cents": 10_200,
            "retryable": False,
            "retry_after_seconds": None,
        },
        message='the budget\'s hard cap is reached (budget "Prod AI", b-1)',
    ),
    dict(
        id="BUDGET_HARD_CAP-server-shape-not-known-yet",
        body=MONEY_BUDGET_NOT_KNOWN_YET,
        headers={"retry-after": "5"},
        cls=TensorCostBudgetHardCapError,
        code="PERIOD_BUDGET_EXCEEDED",
        reason="spend_not_known_yet",
        extra={
            "budget_code": "BUDGET_HARD_CAP",
            "budget_id": "b-2",
            "period": "weekly",
            "cap_cents": 5_000,
            "spent_cents": None,
            "retryable": True,
            "retry_after_seconds": 5,
        },
        message="the budget's spend hasn't been counted yet; retry in 5 s",
    ),
    dict(
        id="GUARDRAIL_HARD_STOP",
        body={
            "error": {
                "code": "GUARDRAIL_HARD_STOP",
                "message": "Request refused by guardrail hard_stop policy",
                "policy_id": "33333333-3333-3333-3333-333333333333",
                "reason": "pii",
            }
        },
        cls=TensorCostGuardrailHardStopError,
        code="GUARDRAIL_HARD_STOP",
        reason="pii",
        extra={"policy_id": "33333333-3333-3333-3333-333333333333"},
    ),
    dict(
        id="MODEL_GOVERNANCE_DENIED",
        body={
            "error": {
                "code": "MODEL_GOVERNANCE_DENIED",
                "message": "gpt-4o is not approved for this tenant",
                "provider": "openai",
                "model": "gpt-4o",
                "governance_status": "denied",
            }
        },
        cls=TensorCostModelGovernanceDeniedError,
        code="MODEL_GOVERNANCE_DENIED",
        reason=None,
        extra={"governance_status": "denied"},
    ),
    dict(
        id="COMPLIANCE_DENIED",
        body={
            "error": {
                "code": "COMPLIANCE_DENIED",
                "message": "Request refused by compliance policy: region",
                "reason": "region",
            }
        },
        cls=TensorCostComplianceDeniedError,
        code="COMPLIANCE_DENIED",
        reason="region",
    ),
    dict(
        id="SOMETHING_NEW",
        body={"error": {"code": "SOMETHING_NEW", "message": "a future refusal", "reason": "future_reason"}},
        cls=TensorCostRefusedError,
        code="SOMETHING_NEW",
        reason="future_reason",
        message="tensorcost: a future refusal",
    ),
    dict(
        id="SOMETHING_NEW-flat",
        body={"code": "SOMETHING_NEW", "message": "flat refusal", "reason": "flat"},
        cls=TensorCostRefusedError,
        code="SOMETHING_NEW",
        reason="flat",
    ),
    dict(
        id="SOMETHING_NEW-on-402",
        status=402,
        body={"error": {"code": "SOMETHING_NEW", "message": "plan says no", "reason": "plan"}},
        cls=TensorCostRefusedError,
        code="SOMETHING_NEW",
        reason="plan",
    ),
    dict(
        id="SOMETHING_NEW-on-429",
        status=429,
        body={"error": {"code": "SOMETHING_NEW", "message": "slow down, for policy reasons"}},
        cls=TensorCostRefusedError,
        code="SOMETHING_NEW",
        reason=None,
    ),
]

PATHS = [
    pytest.param("route", False, "/v1/chat/completions", id="proxy"),
    pytest.param("route", True, "/v1/chat/completions", id="proxy-stream"),
    pytest.param("govern", False, "/v1/admit", id="admit"),
]

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def tensorcost_api(monkeypatch: pytest.MonkeyPatch):
    """Route every httpx.Client the proxy / admit clients build to a handler."""
    state: dict[str, Any] = {"handler": None, "calls": []}
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"].append(str(request.url))
        return state["handler"](request)

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("tensorcost._proxy_client.httpx.Client", factory)
    monkeypatch.setattr("tensorcost._admit_client.httpx.Client", factory)
    return state


def _install(recording_transport: Any, layer: str, fail_open_enabled: bool = True):
    client = MagicMock()
    client.api_key = "sk-test"
    client.base_url = "https://api.openai.com/v1"
    direct = client.chat.completions.create
    direct.return_value = {"id": "direct", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    _openai.install(
        client,
        recording_transport,
        tenant_id="t-1",
        environment=None,
        connection_id=None,
        max_layer=layer,
        proxy_url="https://proxy.test",
        base_url="https://tc.test",
        # Retries on: a refusal must still be asked exactly once.
        retry=RetryConfig(max_attempts=3, base_delay_ms=1.0, max_delay_ms=2.0),
        timeout_s=1.0,
        headers_timeout_s=1.0,
        fail_open_enabled=fail_open_enabled,
        circuit=CircuitBreaker(),
    )
    return client, direct


def _call(client: Any, stream: bool) -> Any:
    kwargs: dict[str, Any] = {"model": "gpt-4o", "messages": []}
    if stream:
        kwargs["stream"] = True
    return client.chat.completions.create(**kwargs)


@pytest.mark.parametrize("layer,stream,path", PATHS)
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_refusal_is_terminal(tensorcost_api, recording_transport, layer, stream, path, case):
    client, direct = _install(recording_transport, layer)
    tensorcost_api["handler"] = lambda _r: httpx.Response(
        case.get("status", 403), json=case["body"], headers=case.get("headers", {})
    )

    with pytest.raises(TensorCostRefusedError) as info:
        _call(client, stream)

    assert len(tensorcost_api["calls"]) == 1
    assert path in tensorcost_api["calls"][0]
    direct.assert_not_called()
    err = info.value
    assert isinstance(err, case["cls"])
    if case["cls"] in (TensorCostRefusedError, TensorCostPeriodBudgetExceededError):
        # Exact class: a plain PERIOD_BUDGET_EXCEEDED (no budget_code) keeps
        # 1.4.0's period-budget error and is not mistaken for a hard cap.
        assert type(err) is case["cls"]
    if case["cls"] is TensorCostBudgetHardCapError:
        # 1.4.0 raised the period-budget error for this wire shape.
        assert isinstance(err, TensorCostPeriodBudgetExceededError)
    assert err.code == case["code"]
    assert err.reason == case["reason"]
    assert err.status == case.get("status", 403)
    for k, v in case.get("extra", {}).items():
        assert getattr(err, k) == v, k
    if "message" in case:
        assert case["message"] in str(err)
    assert should_fall_back_to_provider(err) is False


@pytest.mark.parametrize("layer,stream,path", PATHS)
def test_hard_cap_is_terminal_with_fail_open_disabled(tensorcost_api, recording_transport, layer, stream, path):
    client, direct = _install(recording_transport, layer, fail_open_enabled=False)
    tensorcost_api["handler"] = lambda _r: httpx.Response(403, json=MONEY_BUDGET_REACHED)
    with pytest.raises(TensorCostBudgetHardCapError):
        _call(client, stream)
    assert len(tensorcost_api["calls"]) == 1
    direct.assert_not_called()


def test_admit_refusal_is_no_longer_mislabelled(tensorcost_api, recording_transport):
    client, direct = _install(recording_transport, "govern")
    tensorcost_api["handler"] = lambda _r: httpx.Response(403, json={"statusCode": 403, "message": "Forbidden"})
    with pytest.raises(TensorCostRefusedError) as info:
        _call(client, False)
    assert not isinstance(info.value, (TensorCostModelGovernanceDeniedError, TensorCostProviderError))
    assert info.value.code is None
    direct.assert_not_called()


WAF_HTML = "<html><head><title>403 Forbidden</title></head><body><center><h1>403 Forbidden</h1></center></body></html>"


@pytest.mark.parametrize("layer,stream,path", PATHS)
@pytest.mark.parametrize(
    "respond",
    [
        pytest.param(lambda _r: httpx.Response(403, text=WAF_HTML, headers={"content-type": "text/html"}), id="html-403"),
        pytest.param(lambda _r: httpx.Response(403), id="empty-403"),
        pytest.param(lambda _r: httpx.Response(401, text="<html>401</html>"), id="html-401"),
    ],
)
def test_non_json_4xx_is_a_gateway_rejection(tensorcost_api, recording_transport, layer, stream, path, respond):
    from tensorcost import TensorCostGatewayRejectedError

    client, direct = _install(recording_transport, layer)
    tensorcost_api["handler"] = respond
    with pytest.raises(TensorCostGatewayRejectedError) as info:
        _call(client, stream)
    assert len(tensorcost_api["calls"]) == 1
    direct.assert_not_called()
    assert "before it reached TensorCost or the provider" in str(info.value)
    assert should_fall_back_to_provider(info.value) is False


@pytest.mark.parametrize("layer,stream,path", PATHS)
def test_nest_401_from_tensorcost_is_key_rejected(tensorcost_api, recording_transport, layer, stream, path):
    from tensorcost import TensorCostKeyRejectedError

    client, direct = _install(recording_transport, layer)
    tensorcost_api["handler"] = lambda _r: httpx.Response(401, json={"statusCode": 401, "message": "Unauthorized"})
    with pytest.raises(TensorCostKeyRejectedError):
        _call(client, stream)
    direct.assert_not_called()


def test_nest_400_from_the_proxy_is_a_tensorcost_error(tensorcost_api, recording_transport):
    client, direct = _install(recording_transport, "route")
    tensorcost_api["handler"] = lambda _r: httpx.Response(
        400, json={"statusCode": 400, "message": "x-tc-provider-url header is required", "error": "Bad Request"}
    )
    with pytest.raises(TensorCostRefusedError) as info:
        _call(client, False)
    assert "x-tc-provider-url header is required" in str(info.value)
    direct.assert_not_called()


# ---------------------------------------------------------------------------
# Outages: the allow-list.


def _network(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("timed out", request=request)


OUTAGES = [
    pytest.param(_network, id="network-error"),
    pytest.param(_timeout, id="timeout"),
    pytest.param(lambda _r: httpx.Response(503, json={"error": "unavailable"}), id="proxy-503"),
    pytest.param(
        lambda _r: httpx.Response(
            402,
            json={
                "kind": "scale-cogs-exceeded",
                "error": "Payment Required",
                "plan": "scale",
                "limit": "proxy_requests_per_month",
                "cap": 100000,
                "current": 100001,
                "upgrade_hint": "Scale monthly proxy request ceiling reached. Contact us for Enterprise.",
            },
        ),
        id="proxy-402-plan-ceiling",
    ),
    pytest.param(
        lambda _r: httpx.Response(429, json={"statusCode": 429, "message": "ThrottlerException: Too Many Requests"}),
        id="proxy-429",
    ),
]


@pytest.mark.parametrize("layer,stream,path", PATHS)
@pytest.mark.parametrize("respond", OUTAGES)
def test_outage_falls_back_when_enabled(tensorcost_api, recording_transport, layer, stream, path, respond):
    client, direct = _install(recording_transport, layer, fail_open_enabled=True)
    tensorcost_api["handler"] = respond
    result = _call(client, stream)
    assert result["id"] == "direct"
    direct.assert_called_once()


@pytest.mark.parametrize("layer,stream,path", PATHS)
@pytest.mark.parametrize("respond", OUTAGES)
def test_outage_does_not_fall_back_when_disabled(tensorcost_api, recording_transport, layer, stream, path, respond):
    client, direct = _install(recording_transport, layer, fail_open_enabled=False)
    tensorcost_api["handler"] = respond
    with pytest.raises(Exception) as info:
        _call(client, stream)
    assert should_fall_back_to_provider(info.value) or isinstance(info.value, TensorCostProxyError)
    direct.assert_not_called()


# ---------------------------------------------------------------------------
# A provider's own 4xx, relayed by the proxy.

PROVIDER_400 = {
    "error": {
        "message": "Invalid value for 'messages': empty array",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "invalid_value",
    }
}


@pytest.mark.parametrize("fail_open_enabled", [True, False])
@pytest.mark.parametrize("stream", [False, True])
def test_relayed_provider_4xx_surfaces_without_fallback(tensorcost_api, recording_transport, fail_open_enabled, stream):
    client, direct = _install(recording_transport, "route", fail_open_enabled=fail_open_enabled)
    tensorcost_api["handler"] = lambda _r: httpx.Response(400, json=PROVIDER_400)
    with pytest.raises(TensorCostProviderError) as info:
        _call(client, stream)
    assert len(tensorcost_api["calls"]) == 1
    direct.assert_not_called()
    err = info.value
    assert not isinstance(err, TensorCostRefusedError)
    assert err.status == 400
    assert str(err) == "tensorcost: upstream provider returned 400: Invalid value for 'messages': empty array"
    assert err.body == PROVIDER_400


def test_lowercase_provider_code_in_a_403_is_not_a_refusal(tensorcost_api, recording_transport):
    client, direct = _install(recording_transport, "route")
    tensorcost_api["handler"] = lambda _r: httpx.Response(
        403, json={"error": {"message": "Country not supported", "code": "unsupported_country_region_territory"}}
    )
    with pytest.raises(TensorCostProviderError) as info:
        _call(client, False)
    assert info.value.status == 403
    direct.assert_not_called()


# ---------------------------------------------------------------------------
# Observations: application and tags are held back in 1.5.0 (byoc-v0.1.40
# rejects them), behind _observation_scope.SEND.


@pytest.fixture
def scope_on(monkeypatch):
    monkeypatch.setattr("tensorcost._observation_scope.SEND", True)


def _openai_obs(recording_transport):
    client = MagicMock()
    client.chat.completions.create.return_value = {"id": "c", "usage": {}}
    _openai.install(
        client,
        recording_transport,
        tenant_id="t-1",
        environment=None,
        connection_id=None,
        application="checkout",
        tags=["team=pay", "tier=gold"],
    )
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    return recording_transport.observations


def test_observations_hold_back_application_and_tags_by_default(recording_transport):
    obs = _openai_obs(recording_transport)
    assert len(obs) == 1
    assert "application" not in obs[0]
    assert "tags" not in obs[0]


def test_observations_carry_application_and_tags_with_the_flag(recording_transport, scope_on):
    obs = _openai_obs(recording_transport)
    assert obs[0]["application"] == "checkout"
    assert obs[0]["tags"] == ["team=pay", "tier=gold"]


def test_bedrock_and_vertex_observations_carry_them_with_the_flag(recording_transport, scope_on):
    from tensorcost._providers import _bedrock, _vertex

    br = MagicMock()
    br.meta.service_model.service_name = "bedrock-runtime"
    br.invoke_model.return_value = {"body": MagicMock(read=lambda: b"{}"), "ResponseMetadata": {}}
    _bedrock.install(br, recording_transport, "t-1", None, None, application="app-b", tags=["x=1"])
    try:
        br.invoke_model(modelId="anthropic.claude-3", body=b"{}")
    except Exception:  # noqa: BLE001 - the observation is what's under test
        pass
    vx = MagicMock()
    vx.models.generate_content.return_value = MagicMock(usage_metadata=None)
    _vertex.install(vx, recording_transport, "t-1", None, None, application="app-v", tags=["y=2"])
    vx.models.generate_content(model="gemini-1.5-pro", contents="hi")
    apps = {o.get("application") for o in recording_transport.observations}
    assert {"app-b", "app-v"} <= apps


# ---------------------------------------------------------------------------
# Tenant id, Bedrock/Vertex govern, revoked key.


@pytest.fixture
def fresh_warnings(monkeypatch):
    import tensorcost._wrap as wrap_mod

    wrap_mod._reset_warnings()
    monkeypatch.delenv("TENSORCOST_TENANT_ID", raising=False)
    yield
    wrap_mod._reset_warnings()


@pytest.mark.parametrize("layer", ["govern", "steer", "route"])
def test_wrap_without_tenant_id_raises_at_govern_and_above(fresh_warnings, layer):
    from tensorcost import MissingConfigError, wrap

    client = MagicMock()
    client.messages = None
    with pytest.raises(MissingConfigError) as info:
        wrap(client, api_key="k", max_layer=layer, proxy_url="https://proxy.test")
    msg = str(info.value)
    assert "TENSORCOST_TENANT_ID" in msg and "tenant_id" in msg
    assert "Developers › SDK setup" in msg


def test_wrap_without_tenant_id_raises_even_with_fail_open_off(fresh_warnings):
    from tensorcost import MissingConfigError, wrap

    client = MagicMock()
    client.messages = None
    with pytest.raises(MissingConfigError):
        wrap(client, api_key="k", max_layer="govern", fail_open_enabled=False)


def test_observe_without_tenant_id_works_and_warns_once(fresh_warnings, caplog):
    from tensorcost import wrap

    def make():
        c = MagicMock()
        c.messages = None
        c.chat.completions.create.return_value = {"id": "c"}
        return c

    a = make()
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        wrap(a, api_key="k")
        wrap(make(), api_key="k")
    assert a.chat.completions.create(model="gpt-4o-mini", messages=[]) == {"id": "c"}
    w = [r.getMessage() for r in caplog.records if "no tenant id" in r.getMessage()]
    assert len(w) == 1
    assert "no observations reach TensorCost" in w[0]


@pytest.mark.parametrize("name", ["Bedrock", "Vertex"])
def test_bedrock_and_vertex_govern_warns_once(fresh_warnings, caplog, monkeypatch, name):
    from tensorcost import wrap

    monkeypatch.setattr("tensorcost._wrap.detect_provider", lambda _c: name.lower())
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        wrap(MagicMock(), api_key="k", tenant_id="t-1", max_layer="govern")
        wrap(MagicMock(), api_key="k", tenant_id="t-1", max_layer="govern")
    w = [r.getMessage() for r in caplog.records if "govern isn't available" in r.getMessage()]
    assert w == [
        f"tensorcost: govern isn't available for {name} yet; calls are observed only. "
        "Stops, caps and budgets are not enforced on this client."
    ]


@pytest.mark.parametrize("status", [401, 403])
def test_token_exchange_key_rejection_is_typed(status):
    from tensorcost import TensorCostKeyRejectedError
    from tensorcost._transport import ObservationTransport

    t = ObservationTransport(api_key="k", base_url="https://tc.test", tenant_id="t-1")
    t._client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(status, json={"message": "revoked"})))
    try:
        with pytest.raises(TensorCostKeyRejectedError) as info:
            t.get_token()
    finally:
        t.close()
    assert "the TensorCost key was rejected" in str(info.value)
    assert should_fall_back_to_provider(info.value) is False


def test_token_exchange_5xx_still_falls_back():
    from tensorcost._transport import ObservationTransport

    t = ObservationTransport(api_key="k", base_url="https://tc.test", tenant_id="t-1")
    t._client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(503)))
    try:
        with pytest.raises(TensorCostProxyError) as info:
            t.get_token()
    finally:
        t.close()
    assert should_fall_back_to_provider(info.value) is True

"""An operator stop is terminal in the Python SDK.

The proxy answers a stopped run with 403 RUN_STOPPED, on the chat path and on
the govern-layer /v1/admit. Before these tests the SDK did not know the code:
on the chat path the 403 became a generic TensorCostProviderError, which
should_fall_back_to_provider() allows, so with fail_open_enabled (the default)
the wrapper called the provider directly and the stop did nothing. On /admit
the refusal lost who stopped the run and why.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tensorcost import TensorCostRunStoppedError
from tensorcost._admit_client import admit_request
from tensorcost._errors import (
    TensorCostModelGovernanceDeniedError,
    TensorCostProviderError,
    TensorCostProxyError,
    TensorCostRefusedError,
)
from tensorcost._proxy_client import (
    _terminal_proxy_refusal_error,
    proxy_request,
    should_fall_back_to_provider,
)
from tensorcost._retry import RetryConfig

STOPPED_BODY = {
    "error": {
        "code": "RUN_STOPPED",
        "message": "Agent run run-abc was stopped by ops@example.test at 2026-10-25T10:00:00.000Z: runaway",
        "workflow_id": "run-abc",
        "stopped_by": "ops@example.test",
        "stopped_at": "2026-10-25T10:00:00.000Z",
        "reason": "runaway",
    }
}


def _resp(status: int, body, headers: dict | None = None) -> httpx.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.Response(
        status_code=status,
        headers={"Content-Type": "application/json", **(headers or {})},
        content=content,
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _assert_stop_detail(err: TensorCostRunStoppedError) -> None:
    assert err.workflow_id == "run-abc"
    assert err.stopped_by == "ops@example.test"
    assert err.stopped_at == "2026-10-25T10:00:00.000Z"
    assert err.reason == "runaway"
    assert err.status == 403


def test_terminal_refusal_maps_run_stopped_with_its_detail():
    err = _terminal_proxy_refusal_error(
        json.dumps(STOPPED_BODY).encode(), request_id="req-1", attempt=2
    )
    assert isinstance(err, TensorCostRunStoppedError)
    _assert_stop_detail(err)
    assert err.request_id == "req-1"
    assert err.attempt == 2


def test_run_stopped_does_not_fall_back_to_provider():
    assert should_fall_back_to_provider(TensorCostRunStoppedError("stopped")) is False


def test_proxy_request_raises_run_stopped_and_does_not_retry():
    calls = [0]

    def handler(_req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return _resp(403, STOPPED_BODY)

    with pytest.raises(TensorCostRunStoppedError) as info:
        proxy_request(
            proxy_url="https://proxy.example.com",
            bearer_token="jwt",
            provider_url="https://api.openai.com",
            provider_auth="Bearer sk-test",
            body={"model": "gpt-4o-mini", "messages": []},
            model="gpt-4o-mini",
            provider="openai",
            operation="chat.completions",
            correlation_id="c-1",
            environment=None,
            retry=RetryConfig(max_attempts=3),
            http_client=_client(handler),
        )
    _assert_stop_detail(info.value)
    assert calls[0] == 1


def _admit(handler, **kw):
    return admit_request(
        base_url="https://proxy.example.com",
        bearer_token="jwt",
        max_layer="govern",
        metadata={"provider": "openai", "model": "gpt-4o-mini", "workflow_id": "run-abc"},
        http_client=_client(handler),
        **kw,
    )


def test_admit_raises_run_stopped_even_with_fail_open():
    with pytest.raises(TensorCostRunStoppedError) as info:
        _admit(lambda _r: _resp(403, STOPPED_BODY, {"x-tc-request-id": "req-9"}), fail_open=True)
    _assert_stop_detail(info.value)
    assert info.value.request_id == "req-9"


def test_admit_other_403_is_typed_by_its_code():
    # 1.4.0 returned (False, header) and the wrapper raised a generic error.
    with pytest.raises(TensorCostModelGovernanceDeniedError) as info:
        _admit(
            lambda _r: _resp(
                403,
                {"error": {"code": "MODEL_GOVERNANCE_DENIED"}},
                {"x-tc-decision": "layer=govern; action=refuse"},
            )
        )
    assert info.value.code == "MODEL_GOVERNANCE_DENIED"


def test_admit_403_with_non_json_body_is_a_gateway_rejection():
    # 1.5.0 round 2: HTML means something in front of TensorCost said no.
    from tensorcost import TensorCostGatewayRejectedError

    with pytest.raises(TensorCostGatewayRejectedError) as info:
        _admit(lambda _r: _resp(403, b"<html>403</html>"))
    assert info.value.status == 403
    assert should_fall_back_to_provider(info.value) is False


def test_admit_still_fails_open_on_transport_error_and_5xx():
    def boom(_r):
        raise httpx.ConnectError("refused")

    assert _admit(boom) == (True, None)
    assert _admit(lambda _r: _resp(503, {})) == (True, None)
    assert _admit(lambda _r: _resp(204, b"", {"x-tc-decision": "action=allow"}))[0] is True


def test_admit_5xx_raises_when_fail_open_is_off():
    # Typed since 1.5.0 (was httpx.HTTPStatusError).
    with pytest.raises(TensorCostProxyError):
        _admit(lambda _r: _resp(503, {}), fail_open=False)


def test_uppercase_403_code_on_the_chat_path_is_a_terminal_refusal():
    # 1.4.0: None here, and ProviderError fell back. Now an uppercase code is
    # a TensorCost refusal shape, and unknown means terminal.
    err = _terminal_proxy_refusal_error(
        json.dumps({"error": {"code": "FORBIDDEN"}}).encode(), request_id=None, attempt=1
    )
    assert isinstance(err, TensorCostRefusedError)
    assert err.code == "FORBIDDEN"
    assert should_fall_back_to_provider(err) is False
    # A relayed provider error never falls back either (1.4.0: it did).
    assert should_fall_back_to_provider(TensorCostProviderError("403")) is False
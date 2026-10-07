"""End-to-end smoke for the Python agent ↔ gpu-service gRPC path.

Drives the production code (`src/grpc_client.py`, `src/auth/hmac.py`)
against a real gpu-service. Asserts the full handshake → metric →
instance → command-result loop works against a real database.

This is the test FCR §7.5 asked for. It is intentionally NOT a unit
test — the unit tests in `tests/test_grpc_client_v2.py` and
`tests/test_hmac_auth.py` cover wire-shape and signature byte equality
with mocks. This file exists to catch the things mocks can't catch:

  * gRPC dependency-version drift between the agent's `grpcio` and
    gpu-service's `@grpc/grpc-js`.
  * `agent.proto` schema drift between `apps/unified-gpu-agent/src/proto/`
    and `apps-new/packages/contracts/proto/agent.proto`.
  * HMAC canonicalisation parity between Python `auth.hmac` and TS
    `packages/agent-sdk/src/hmac.ts`.
  * Real Postgres rejecting a row the test thought it was writing.

Skip semantics: every test in this file is skipped when the required
env vars are absent, so a developer running `pytest` from the agent
repo with no infra up doesn't see spurious red.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional

import pytest


# Make the agent's `src/` importable without an editable install — the
# unit tests do the same dance.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "proto"))


_REQUIRED_ENVS = (
    "GRPC_TARGET",
    "TENANT_ID",
    "AGENT_HOSTNAME",
    "AGENT_KEY_ID",
    "AGENT_HMAC_PEPPER",
)


pytestmark = pytest.mark.skipif(
    any(not os.getenv(k) for k in _REQUIRED_ENVS),
    reason=(
        "e2e smoke requires a live gpu-service and the env contract "
        "(" + ", ".join(_REQUIRED_ENVS) + "). "
        "Skipping — see tests/e2e/__init__.py for local-run instructions."
    ),
)


@pytest.fixture(scope="module")
def grpc_target() -> str:
    return os.environ["GRPC_TARGET"]


@pytest.fixture(scope="module")
def tenant_id() -> str:
    return os.environ["TENANT_ID"]


@pytest.fixture(scope="module")
def agent_hostname() -> str:
    return os.environ["AGENT_HOSTNAME"]


@pytest.fixture(scope="module")
def agent_creds() -> Dict[str, str]:
    return {
        "key_id": os.environ["AGENT_KEY_ID"],
        "hmac_pepper": os.environ["AGENT_HMAC_PEPPER"],
    }


@pytest.fixture(scope="module")
def db_conn():
    """Optional direct DB cursor for assertions. Skips DB-flavour assertions
    when E2E_DATABASE_URL isn't set (e.g. running against a remote staging
    env where the runner can't reach Postgres directly)."""
    url = os.getenv("E2E_DATABASE_URL")
    if not url:
        yield None
        return
    try:
        import psycopg2  # type: ignore
    except ImportError:
        pytest.skip("psycopg2 not installed; install requirements-dev.txt")
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def ensure_agent_row(db_conn, tenant_id: str, agent_hostname: str, agent_creds: Dict[str, str]):
    """Insert the gpu.agent row the HMAC handshake needs.

    The CI workflow tries to use the gpu-service `provision-agent` CLI
    first; if that step short-circuited (e.g. the CLI binary wasn't in
    the build) we fall back to a direct upsert here. Either way the
    row must exist before the agent connects, otherwise the server
    rejects with `agent hello: key_id miss`.
    """
    if db_conn is None:
        # Without a DB handle we can't pre-seed; rely on the CI step
        # having done it. The connection attempt below will surface a
        # clear error if the row's missing.
        yield
        return
    import bcrypt  # type: ignore  # in requirements-dev for the e2e suite

    pepper_bytes = bytes.fromhex(agent_creds["hmac_pepper"])
    bcrypt_hash = bcrypt.hashpw(pepper_bytes, bcrypt.gensalt(rounds=4)).decode("utf-8")
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO gpu.agent (
                id, tenant_id, hostname, key_id, hmac_pepper_hash, status, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, 'active', now(), now()
            )
            ON CONFLICT (tenant_id, hostname) DO UPDATE SET
                key_id = EXCLUDED.key_id,
                hmac_pepper_hash = EXCLUDED.hmac_pepper_hash,
                status = 'active',
                updated_at = now()
            """,
            (
                str(uuid.uuid4()),
                tenant_id,
                agent_hostname,
                agent_creds["key_id"],
                bcrypt_hash,
            ),
        )
    db_conn.commit()
    yield
    # No teardown — the per-CI Postgres is ephemeral.


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandshake:
    """Stage-1: open the bidi stream, prove HMAC + key_id + nonce are
    accepted by the real server."""

    def test_connect_and_receive_server_accepted(
        self,
        grpc_target: str,
        tenant_id: str,
        agent_hostname: str,
        agent_creds: Dict[str, str],
    ):
        from grpc_client import GrpcClient  # noqa: WPS433 — production import

        received: Dict[str, Any] = {}

        async def _on_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
            received["cmd"] = cmd
            return {"status": "succeeded"}

        client = GrpcClient(
            grpc_target=grpc_target,
            agent_id=agent_hostname,
            tenant_id=tenant_id,
            key_id=agent_creds["key_id"],
            hmac_pepper=agent_creds["hmac_pepper"],
            on_command=_on_command,
            use_tls=False,
        )

        # Spin the connect loop in the background; assert is_connected
        # flips inside a bounded wait. The client's own retry loop
        # would hide a real handshake failure, so we cap retries here.
        connected = _wait_for_connection(client, timeout_seconds=15.0)
        try:
            assert connected, (
                "handshake did not complete within 15s — check gpu-service "
                "logs for `agent hello: HMAC mismatch` or `key_id miss`"
            )
        finally:
            client.stop()


class TestMetricBatch:
    def test_send_metric_batch_persists(
        self,
        grpc_target: str,
        tenant_id: str,
        agent_hostname: str,
        agent_creds: Dict[str, str],
        db_conn,
    ):
        from grpc_client import GrpcClient  # noqa: WPS433

        client = GrpcClient(
            grpc_target=grpc_target,
            agent_id=agent_hostname,
            tenant_id=tenant_id,
            key_id=agent_creds["key_id"],
            hmac_pepper=agent_creds["hmac_pepper"],
            use_tls=False,
        )
        try:
            assert _wait_for_connection(client, 15.0)

            batch = {
                "device_uuid": "GPU-e2e-0000",
                "samples": [
                    {
                        "timestamp_unix": int(time.time()),
                        "utilisation_pct": 0.42,
                        "memory_used_bytes": 1024 * 1024 * 1024,
                    }
                ],
            }
            client.send_metric_batch(batch)
            # gpu-service buffers in Redis then flushes to Postgres on a
            # 1 s tick. Wait up to 5 s for the row to land.
            if db_conn is not None:
                _poll_db(
                    db_conn,
                    "SELECT count(*) FROM gpu.metric_sample WHERE tenant_id = %s",
                    (tenant_id,),
                    expect_at_least=1,
                    timeout_seconds=5.0,
                )
        finally:
            client.stop()


class TestInstanceUpsert:
    def test_send_instance_update_persists(
        self,
        grpc_target: str,
        tenant_id: str,
        agent_hostname: str,
        agent_creds: Dict[str, str],
        db_conn,
    ):
        from grpc_client import GrpcClient  # noqa: WPS433

        client = GrpcClient(
            grpc_target=grpc_target,
            agent_id=agent_hostname,
            tenant_id=tenant_id,
            key_id=agent_creds["key_id"],
            hmac_pepper=agent_creds["hmac_pepper"],
            use_tls=False,
        )
        try:
            assert _wait_for_connection(client, 15.0)
            client.send_instance_update(
                {
                    "instance_id": "i-e2e-0001",
                    "instance_type": "g5.xlarge",
                    "region": "us-east-1",
                    "state": "running",
                }
            )
            if db_conn is not None:
                _poll_db(
                    db_conn,
                    "SELECT count(*) FROM gpu.instance WHERE tenant_id = %s "
                    "AND instance_id = 'i-e2e-0001'",
                    (tenant_id,),
                    expect_at_least=1,
                    timeout_seconds=5.0,
                )
        finally:
            client.stop()


class TestCommandRoundtrip:
    """The hardest path — server pushes a command, agent applies, server
    receives the result, DB reflects the new state."""

    def test_pending_command_is_applied(
        self,
        grpc_target: str,
        tenant_id: str,
        agent_hostname: str,
        agent_creds: Dict[str, str],
        db_conn,
    ):
        from grpc_client import GrpcClient  # noqa: WPS433

        if db_conn is None:
            pytest.skip("command roundtrip requires E2E_DATABASE_URL to seed the pending row")

        # 1. Insert a pending command server-side.
        cmd_id = str(uuid.uuid4())
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gpu.instance_command (
                    id, tenant_id, agent_hostname, command_type, payload, status, created_at
                ) VALUES (
                    %s, %s, %s, 'noop', '{}'::jsonb, 'pending', now()
                )
                """,
                (cmd_id, tenant_id, agent_hostname),
            )
        db_conn.commit()

        # 2. Wire a callback that records what the server pushes.
        applied: Dict[str, Any] = {}

        async def _on_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
            applied["cmd"] = cmd
            return {"status": "succeeded"}

        client = GrpcClient(
            grpc_target=grpc_target,
            agent_id=agent_hostname,
            tenant_id=tenant_id,
            key_id=agent_creds["key_id"],
            hmac_pepper=agent_creds["hmac_pepper"],
            on_command=_on_command,
            use_tls=False,
        )
        try:
            assert _wait_for_connection(client, 15.0)

            # 3. Wait for the command to flow agent-side.
            deadline = time.time() + 10.0
            while time.time() < deadline and "cmd" not in applied:
                time.sleep(0.2)
            assert "cmd" in applied, (
                "agent never received the pending command — check gpu-service "
                "command-dispatch loop and server logs"
            )

            # 4. Wait for the server to record `succeeded`.
            _poll_db(
                db_conn,
                "SELECT count(*) FROM gpu.instance_command "
                "WHERE id = %s AND status = 'succeeded'",
                (cmd_id,),
                expect_at_least=1,
                timeout_seconds=5.0,
            )
        finally:
            client.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_connection(client, timeout_seconds: float) -> bool:
    # The production client starts its own thread on `start()`; we
    # poll its is_connected flag until either the timeout or the
    # client surfaces a terminal failure.
    if hasattr(client, "start"):
        client.start()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if getattr(client, "is_connected", False):
            return True
        time.sleep(0.2)
    return False


def _poll_db(conn, sql: str, params: tuple, *, expect_at_least: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_count: Optional[int] = None
    while time.time() < deadline:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            (last_count,) = cur.fetchone()
        if last_count is not None and last_count >= expect_at_least:
            return
        time.sleep(0.2)
    raise AssertionError(
        f"DB assertion failed: expected at least {expect_at_least} rows from "
        f"`{sql}`, last observed {last_count} after {timeout_seconds}s"
    )

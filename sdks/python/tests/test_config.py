import pytest

from tensorcost._config import (
    DEFAULT_BASE_URL,
    MissingConfigError,
    resolve_config,
)


def test_explicit_args_win(monkeypatch):
    monkeypatch.setenv("TENSORCOST_API_KEY", "from-env")
    monkeypatch.setenv("TENSORCOST_BASE_URL", "https://env.example")
    monkeypatch.setenv("TENSORCOST_TENANT_ID", "tenant-env")

    cfg = resolve_config(
        api_key="explicit",
        base_url="https://explicit.example/",
        tenant_id="tenant-x",
    )
    assert cfg.api_key == "explicit"
    assert cfg.base_url == "https://explicit.example"  # trailing / stripped
    assert cfg.tenant_id == "tenant-x"
    assert cfg.fail_open is True


def test_env_var_resolution(monkeypatch):
    monkeypatch.setenv("TENSORCOST_API_KEY", "from-env")
    monkeypatch.setenv("TENSORCOST_BASE_URL", "https://env.example")
    cfg = resolve_config()
    assert cfg.api_key == "from-env"
    assert cfg.base_url == "https://env.example"
    assert cfg.tenant_id is None


def test_default_base_url_when_unset(monkeypatch):
    monkeypatch.setenv("TENSORCOST_API_KEY", "k")
    cfg = resolve_config()
    assert cfg.base_url == DEFAULT_BASE_URL


def test_missing_api_key_raises():
    with pytest.raises(MissingConfigError) as exc_info:
        resolve_config()
    assert "TENSORCOST_API_KEY" in str(exc_info.value)


def test_fail_open_propagates():
    cfg = resolve_config(api_key="k", fail_open=False)
    assert cfg.fail_open is False

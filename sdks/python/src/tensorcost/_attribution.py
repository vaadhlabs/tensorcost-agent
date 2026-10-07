"""Per-call attribution scope for wrapped clients (with_meta support)."""

from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import Any, Optional

# Active per-call overrides while a with_meta()-scoped method runs.
_scope_override: ContextVar[dict[str, str] | None] = ContextVar(
    "tensorcost_scope_override", default=None
)


class MetaOverrideClient:
    """Shallow wrapper so ``with_meta()`` can override tags per task."""

    __slots__ = ("_tensorcost_inner", "_tensorcost_meta_override")

    def __init__(self, inner: Any, override: dict[str, str]) -> None:
        self._tensorcost_inner = inner
        self._tensorcost_meta_override = override

    def __getattr__(self, name: str) -> Any:
        val = getattr(self._tensorcost_inner, name)
        return _scope_namespace(val, self._tensorcost_meta_override)

    def with_meta(self, **kwargs: str | None) -> "MetaOverrideClient":
        merged = {
            **self._tensorcost_meta_override,
            **{k: v for k, v in kwargs.items() if v is not None and str(v).strip()},
        }
        return MetaOverrideClient(self._tensorcost_inner, merged)


class _ScopeNamespace:
    """Proxies nested provider namespaces (``chat.completions.create``)."""

    __slots__ = ("_inner", "_override")

    def __init__(self, inner: Any, override: dict[str, str]) -> None:
        self._inner = inner
        self._override = override

    def __getattr__(self, name: str) -> Any:
        val = getattr(self._inner, name)
        return _scope_namespace(val, self._override)


def _is_namespace_like(val: Any) -> bool:
    if _looks_like_namespace(val):
        return True
    mod = type(val).__module__
    if mod.startswith("unittest.mock"):
        return True
    return False


# MagicMock leaf methods (list, create, …) are callable but must not be
# treated as namespace containers.
_MOCK_METHOD_NAMES = frozenset(
    {
        "create",
        "list",
        "generate_content",
        "generate_content_stream",
        "invoke_model",
        "converse",
    }
)


def _scope_namespace(val: Any, override: dict[str, str]) -> Any:
    if getattr(val, "__tensorcost_wrapped__", False):
        return _scope_callable(val, override)
    if _is_namespace_like(val):
        mock_name = getattr(val, "_mock_name", None)
        if callable(val) and mock_name in _MOCK_METHOD_NAMES:
            return val
        return _ScopeNamespace(val, override)
    return val


def _looks_like_namespace(val: Any) -> bool:
    mod = type(val).__module__
    return mod.startswith("openai.") or mod.startswith("anthropic.")


def _scope_callable(fn: Any, override: dict[str, str]) -> Any:
    @functools.wraps(fn)
    def scoped(*args: Any, **kwargs: Any) -> Any:
        token = _scope_override.set(override)
        try:
            return fn(*args, **kwargs)
        finally:
            _scope_override.reset(token)

    setattr(scoped, "__tensorcost_wrapped__", True)
    return scoped


def attach_with_meta(client: Any) -> None:
    """Attach ``client.with_meta(**kwargs)`` after provider install."""

    def with_meta(**kwargs: str | None) -> MetaOverrideClient:
        cleaned = {
            k: str(v).strip()
            for k, v in kwargs.items()
            if v is not None and str(v).strip()
        }
        return MetaOverrideClient(client, cleaned)

    setattr(client, "with_meta", with_meta)


def resolve_scope_field(
    client_ref: Any | None,
    field: str,
    default: Optional[str],
) -> Optional[str]:
    """Walk MetaOverrideClient chain; per-call override beats wrap default."""
    ctx = _scope_override.get()
    if ctx and field in ctx and ctx[field]:
        return ctx[field]
    if client_ref is None:
        return default
    if isinstance(client_ref, MetaOverrideClient):
        override = client_ref._tensorcost_meta_override
        if field in override and override[field]:
            return override[field]
        return resolve_scope_field(client_ref._tensorcost_inner, field, default)
    return default

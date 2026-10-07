# Contributing

Thanks for helping improve TensorCost’s open-source collection layer.

## Scope

This repository contains the GPU telemetry agent and LLM SDK wrappers. The control plane (attribution, routing, savings proof, governance) is not developed here.

## Development

```bash
cd agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

SDKs:

```bash
cd sdks/node && npm ci && npm test
cd sdks/python && pip install -e ".[dev]" && pytest
```

## Pull requests

Anyone can **fork and open a PR**. Only **Vaadh Labs maintainers** merge to `main`.

- `main` requires **green CI** (all four jobs) and **at least one approving review** from a maintainer.
- Direct pushes to `main` are blocked; use a PR branch.
- One logical change per PR.
- Include tests for agent behavior changes.
- Do not commit secrets or tenant-specific URLs.

**Releases** (`v*` tags → Docker / npm / PyPI) are **not** cut by contributors. Maintainers create annotated tags after merge; tag creation is restricted to Admin/Maintain roles, and the Release workflow uses the protected **`release`** environment (see [RELEASING.md](./RELEASING.md)).

## Security

See [SECURITY.md](./agent/SECURITY.md). Report vulnerabilities to security@tensorcost.com.

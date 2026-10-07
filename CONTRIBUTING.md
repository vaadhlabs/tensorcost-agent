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

- One logical change per PR.
- Include tests for agent behavior changes.
- Do not commit secrets or tenant-specific URLs.

## Security

See [SECURITY.md](./agent/SECURITY.md). Report vulnerabilities to security@tensorcost.com.

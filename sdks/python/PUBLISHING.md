# Publishing tensorcost to PyPI

Manual publish runbook for the `tensorcost` Python SDK (`packages/sdk-python`).

---

## Prerequisites

Install build and upload tools into an isolated environment so they don't
pollute your project venv:

```bash
pipx install build
pipx install twine
```

Or, if you prefer a plain venv:

```bash
python3 -m venv .venv-publish
source .venv-publish/bin/activate
pip install build twine
```

You'll also need a PyPI account with Maintainer/Owner rights on the
`tensorcost` project, and an API token (scoped to the `tensorcost` project)
from https://pypi.org/manage/account/token/. For TestPyPI dry-runs, a
separate token from https://test.pypi.org/manage/account/token/ is needed.

### Credentials file (recommended)

Same idea as `~/.aws/credentials` — keep tokens out of the repo and shell
history:

```bash
cp packages/sdk-python/.py-credentials.example ~/.py-credentials
chmod 600 ~/.py-credentials
# edit ~/.py-credentials — paste pypi-... tokens under [pypi] / [testpypi]
```

Format:

```ini
[pypi]
token = pypi-AgEIcHlwaS...

[testpypi]
token = pypi-AgENdGVzdC...
```

Load into the current shell (sets `TWINE_USERNAME=__token__` and
`TWINE_PASSWORD`):

```bash
source packages/sdk-python/scripts/load-pypi-creds.sh
```

Override path or profile:

```bash
export PYPI_CREDENTIALS_FILE=~/.py-credentials
export PYPI_PROFILE=testpypi   # default: pypi
source packages/sdk-python/scripts/load-pypi-creds.sh
```

Optional `~/.zshrc` shortcut:

```bash
alias pypi-creds='source /path/to/tensorcost/packages/sdk-python/scripts/load-pypi-creds.sh'
```

Alternatives: `~/.pypirc` (twine native) or `TWINE_PASSWORD` env var.

---

## Before you build — version consistency check

Both of these must match and equal the intended release version:

1. `[project] version` in `pyproject.toml`
2. `__version__` in `src/tensorcost/__init__.py`

Check:

```bash
grep '^version' packages/sdk-python/pyproject.toml
grep '__version__' packages/sdk-python/src/tensorcost/__init__.py
```

If they differ, update both to the same string before proceeding. Commit and
tag after the upload (see below), not before — don't tag a version that hasn't
landed on PyPI yet.

---

## Step 1 — clean previous artifacts

```bash
cd packages/sdk-python
rm -rf dist/ build/
```

---

## Step 2 — build

From `packages/sdk-python/`:

```bash
python -m build
```

This produces two artifacts in `dist/`:

- `tensorcost-<version>-py3-none-any.whl` (wheel)
- `tensorcost-<version>.tar.gz` (source distribution)

---

## Step 3 — pre-flight check

```bash
twine check dist/*
```

Both artifacts must pass. Fix any warnings about the long description or
metadata before continuing.

---

## Step 4 — TestPyPI dry-run

Upload to TestPyPI first and do a clean install to confirm the package is
installable:

```bash
./scripts/publish.sh --test
```

Or manually after `source scripts/load-pypi-creds.sh` with
`PYPI_PROFILE=testpypi`:

```bash
twine upload --repository testpypi dist/*
```

Install from TestPyPI in a fresh venv to verify:

```bash
python3 -m venv /tmp/tc-test-install
source /tmp/tc-test-install/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tensorcost
python -c "import tensorcost; print(tensorcost.__version__)"
deactivate
rm -rf /tmp/tc-test-install
```

The version printed must match what you built. If the install or import fails,
fix the issue and loop back to Step 1.

---

## Step 5 — publish to PyPI

One command (build, check, upload):

```bash
./scripts/publish.sh
```

Or manually:

```bash
source scripts/load-pypi-creds.sh
twine upload dist/*
```

Confirm the release landed: https://pypi.org/project/tensorcost/

---

## Step 6 — git tag

After a successful upload:

```bash
git tag sdk-python/v<version>
git push origin sdk-python/v<version>
```

Example for v0.4.0:

```bash
git tag sdk-python/v0.4.0
git push origin sdk-python/v0.4.0
```

---

## Future: migrate to PyPI Trusted Publishing (OIDC)

Trusted Publishing removes the need for long-lived API tokens. Instead, PyPI
issues a short-lived token directly to the GitHub Actions runner via OIDC — no
secret to rotate or accidentally leak.

To set it up:

1. On PyPI, go to the `tensorcost` project → **Publishing** → **Add a new
   publisher** and fill in the GitHub org (`vaadhlabs`), repo (`tensorcost`),
   and the workflow file name you'll use for releases.
2. In the GitHub Actions workflow, set `permissions: id-token: write` and use
   the `pypa/gh-action-pypi-publish` action — no `TWINE_PASSWORD` secret
   needed.

See https://docs.pypi.org/trusted-publishers/ for the full guide. This is the
recommended approach once automated publishing is in place.

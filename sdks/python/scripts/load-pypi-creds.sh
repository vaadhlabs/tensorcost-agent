#!/usr/bin/env bash
# Load PyPI twine credentials from an INI file (AWS-credentials style).
#
# Default file: ~/.py-credentials
# Override:     PYPI_CREDENTIALS_FILE=/path/to/file
# Profile:       PYPI_PROFILE=pypi | testpypi  (default: pypi)
#
# Usage:
#   source packages/sdk-python/scripts/load-pypi-creds.sh
#   twine upload dist/*
#
# Or rely on publish.sh which sources this automatically.

load_pypi_creds() {
  local creds_file="${PYPI_CREDENTIALS_FILE:-${HOME}/.py-credentials}"
  local profile="${PYPI_PROFILE:-pypi}"

  if [[ ! -f "${creds_file}" ]]; then
    echo "load-pypi-creds: missing ${creds_file}" >&2
    echo "  cp packages/sdk-python/.py-credentials.example ~/.py-credentials" >&2
    echo "  chmod 600 ~/.py-credentials" >&2
    return 1
  fi

  local token
  token="$(
    CREDS_FILE="${creds_file}" PROFILE="${profile}" python3 - <<'PY'
import configparser
import os
import sys

path = os.environ["CREDS_FILE"]
profile = os.environ.get("PROFILE", "pypi")

cfg = configparser.ConfigParser()
if not cfg.read(path):
    sys.exit("could not read credentials file")

if not cfg.has_section(profile):
    sys.exit(f"missing [{profile}] section in {path}")

token = cfg.get(profile, "token", fallback="").strip()
if not token or "REPLACE_ME" in token:
    sys.exit(f"missing or placeholder token in [{profile}] of {path}")

print(token)
PY
  )" || return 1

  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="${token}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  load_pypi_creds
  echo "TWINE_USERNAME=${TWINE_USERNAME}"
  echo "TWINE_PASSWORD=<set (${#TWINE_PASSWORD} chars)>"
else
  load_pypi_creds || return 1
fi

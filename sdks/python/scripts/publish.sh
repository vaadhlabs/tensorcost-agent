#!/usr/bin/env bash
# Build and upload tensorcost to PyPI (or TestPyPI).
#
# Prerequisites: pipx install build twine  (or pip install build twine)
# Credentials:   ~/.py-credentials — see .py-credentials.example
#
# Usage:
#   ./scripts/publish.sh              # PyPI (profile pypi)
#   ./scripts/publish.sh --test       # TestPyPI (profile testpypi)
#   ./scripts/publish.sh --check-only # build + twine check, no upload

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

TEST=0
CHECK_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --test) TEST=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${TEST}" -eq 1 ]]; then
  export PYPI_PROFILE=testpypi
  REPO_FLAG=(--repository testpypi)
  REPO_NAME="TestPyPI"
else
  export PYPI_PROFILE=pypi
  REPO_FLAG=()
  REPO_NAME="PyPI"
fi

echo "==> Clean dist/"
rm -rf dist/ build/

echo "==> Build"
python3 -m build

echo "==> twine check"
twine check dist/*

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "==> --check-only: skipping upload"
  exit 0
fi

# shellcheck source=load-pypi-creds.sh
source "${ROOT}/scripts/load-pypi-creds.sh"

echo "==> Upload to ${REPO_NAME} (profile ${PYPI_PROFILE})"
if [[ ${#REPO_FLAG[@]} -gt 0 ]]; then
  twine upload "${REPO_FLAG[@]}" dist/*
else
  twine upload dist/*
fi

echo "==> Done. Verify: https://pypi.org/project/tensorcost/"

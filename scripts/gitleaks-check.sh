#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if ! command -v gitleaks >/dev/null 2>&1; then
  echo "Install gitleaks: https://github.com/gitleaks/gitleaks#installing"
  exit 1
fi
gitleaks detect --source "$ROOT" --no-git --redact -v

#!/usr/bin/env bash
# Regenerate Python stubs for the vendored agent.proto.
#
# Re-vendor the proto from apps-new/packages/contracts/proto/agent.proto
# whenever it changes upstream, then run this script. The generated
# *_pb2.py and *_pb2_grpc.py files are committed so the agent installs
# without protoc on the target host.
#
# Usage:
#   ./bin/regen-proto.sh
#
# grpcio-tools is intentionally NOT in requirements-*.txt (the 1.80+
# line requires protobuf>=6.31, which conflicts irreconcilably with
# sagemaker / google-cloud-* in the agent's runtime variants). We
# install it on demand in a throwaway venv here so devs don't need to
# poison their main environment with the conflicting protobuf pin.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use python3 — older docker images may not symlink `python` to py3.
PY=${PYTHON:-python3}

VENV="$ROOT/.regen-venv"
if [ ! -d "$VENV" ]; then
  echo "creating throwaway venv at $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Pin grpcio-tools to a version whose bundled protoc generates code
# that targets protobuf 4.x at RUNTIME. This matters because:
#   - The cloud SDKs we depend on (sagemaker, google-cloud-*) pin
#     `protobuf<5.0` — pip resolves to protobuf 4.25.x in our image.
#   - grpcio-tools 1.66+ bundle a protoc that emits a hard
#     `_runtime_version.ValidateProtobufRuntimeVersion(6, 31, 1)`
#     call at the top of every generated *_pb2.py. Importing those
#     stubs against a protobuf 4.x runtime raises at import — the
#     symptom we hit in the smoke test (commit 1587207).
#   - grpcio-tools 1.62.x emits NO runtime-version check; the stubs
#     work against any protobuf 4.x runtime. That's the highest
#     toolchain compatible with our cloud-SDK pin set.
#
# Build-isolation: 1.62.x doesn't have wheels for python 3.13, so
# devs on 3.13 either downgrade to 3.11/3.12 OR use a python:3.9-slim
# Docker container to regen. The throwaway venv this script creates
# uses whatever `python3` is on PATH; pin the regen-machine's Python
# to <=3.12 if the install errors with `ModuleNotFoundError:
# pkg_resources` (the source-build symptom on 3.13).
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet "grpcio-tools>=1.62,<1.66" "protobuf>=4.25,<5"

"$PY" -m grpc_tools.protoc \
  -I src/proto \
  --python_out src/proto \
  --grpc_python_out src/proto \
  src/proto/agent.proto

deactivate

echo "regenerated src/proto/agent_pb2.py + src/proto/agent_pb2_grpc.py"
echo "(throwaway venv kept at $VENV — delete with: rm -rf $VENV)"

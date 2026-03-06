#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP_FILE="$ROOT_DIR/spinn_core.cpp"
SO_FILE="$ROOT_DIR/spinn_core.so"

if [[ ! -f "$CPP_FILE" ]]; then
  echo "Missing source file: $CPP_FILE" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but was not found on PATH" >&2
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ is required but was not found on PATH" >&2
  exit 1
fi

PY_INCLUDES="$(python3-config --includes)"
PY_LDFLAGS="$(python3-config --ldflags)"

if [[ ! -d /usr/include/eigen3 ]]; then
  echo "Missing /usr/include/eigen3. Install libeigen3-dev first." >&2
  exit 1
fi

echo "Building $SO_FILE from $CPP_FILE"
g++ -O3 -march=native -fPIC -shared \
  "$CPP_FILE" \
  -o "$SO_FILE" \
  $PY_INCLUDES \
  -I/usr/include/eigen3 \
  $PY_LDFLAGS

echo "Built: $SO_FILE"

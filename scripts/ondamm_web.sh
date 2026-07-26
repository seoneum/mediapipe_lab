#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON"
  echo "Run: bash scripts/setup_env.sh"
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" app/ondamm_web.py "$@"

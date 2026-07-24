#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python app/ondamm_learning_cli.py "$@"

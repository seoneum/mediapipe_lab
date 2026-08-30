#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec .venv/bin/python app/ondamm_purge_cli.py "$@"

#!/usr/bin/env bash
# 사용 가능한 macOS 카메라 인덱스를 찾는 스크립트.
# 예: bash scripts/camera_probe.sh --max-index 8
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

.venv/bin/python app/camera_probe.py "$@"

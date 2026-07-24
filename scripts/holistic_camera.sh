#!/usr/bin/env bash
# Holistic 실시간 카메라 데모 실행 스크립트.
# 뒤에 붙인 인자는 그대로 app/holistic_camera.py로 전달된다.
# 예: bash scripts/holistic_camera.sh --camera 1 --no-lines --point-radius 3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# MPLCONFIGDIR을 프로젝트 내부로 고정해 macOS/샌드박스 설정 파일 문제를 줄인다.
MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python app/holistic_camera.py "$@"

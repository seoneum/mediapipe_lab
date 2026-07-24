#!/usr/bin/env bash
# 설치된 MediaPipe 패키지와 모델 파일이 실제로 열리는지 확인한다.
# 카메라를 켜지는 않고 Task 초기화만 테스트한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python app/check_install.py

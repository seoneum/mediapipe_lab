#!/usr/bin/env bash
# 핵심 Python 패키지 버전 확인용 스크립트.
# 문제가 생겼을 때 이 출력으로 환경이 바뀌었는지 먼저 확인한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIP_CACHE_DIR=.pip-cache .venv/bin/python -m pip show \
  mediapipe \
  opencv-contrib-python \
  numpy \
  matplotlib \
  soundfile \
  sounddevice \
  | awk '/^(Name|Version):/ { print }'

#!/usr/bin/env bash
# MediaPipe Lab 전체 환경을 처음부터 준비하는 스크립트.
# - Python 3.12 가상환경 생성
# - requirements.txt 설치
# - 모델 파일 다운로드
# - 설치 검증 실행
set -euo pipefail

# scripts/ 안에서 실행해도 프로젝트 루트로 이동하도록 경로를 계산한다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 이 Mac에서 uv가 설치한 Python 3.12 경로.
# 경로가 바뀌었거나 파일이 없으면 아래 if문에서 uv로 다시 설치한다.
PY312="/Users/seoneum/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12"

cd "$ROOT_DIR"

if [[ ! -x "$PY312" ]]; then
  # Python 3.12가 없으면 uv로 설치하고 바로 .venv까지 만든다.
  UV_CACHE_DIR=.uv-cache uv python install 3.12
  UV_CACHE_DIR=.uv-cache uv venv --python 3.12 .venv
elif [[ ! -d ".venv" ]]; then
  # Python 3.12는 있는데 .venv만 없으면 표준 venv로 만든다.
  "$PY312" -m venv .venv
fi

# 기존 가상환경이 uv 등의 방식으로 만들어져 pip 모듈이 빠진 경우 복구한다.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade
fi

# pip 캐시를 프로젝트 내부에 둬서 재설치 속도를 높이고 파일 위치를 예측 가능하게 한다.
PIP_CACHE_DIR=.pip-cache .venv/bin/python -m pip install -U pip
PIP_CACHE_DIR=.pip-cache .venv/bin/python -m pip install -r requirements.txt

# 모델 파일을 받고, 마지막에 실제 MediaPipe Task 초기화까지 확인한다.
bash scripts/download_models.sh
MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python app/check_install.py

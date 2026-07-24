#!/usr/bin/env bash
# 정적 이미지 Object Detector 실행 스크립트.
# 예: bash scripts/object_image.sh --image data/images/my_photo.jpg --output outputs/my_photo_detected.jpg --score 0.35
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python app/object_image.py "$@"

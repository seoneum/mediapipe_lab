#!/usr/bin/env bash
# MediaPipe 모델 파일만 다시 다운로드하는 스크립트.
# 모델이 깨졌거나 최신 파일로 다시 받고 싶을 때 실행한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"

mkdir -p "$MODEL_DIR"

# 얼굴 + 포즈 + 양손 landmark용 Holistic 모델.
curl -L \
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" \
  -o "$MODEL_DIR/holistic_landmarker.task"

# COCO 객체 탐지용 EfficientDet Lite2 모델.
# 이미지 안의 person detection을 세면 여러 사람 수 추정의 앞단으로 쓸 수 있다.
curl -L \
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/latest/efficientdet_lite2.tflite" \
  -o "$MODEL_DIR/efficientdet_lite2.tflite"

# 다운로드 결과 파일 크기를 확인한다.
ls -lh "$MODEL_DIR"

#!/usr/bin/env bash
# 미세표정 캡처/분석에 필요한 MediaPipe 모델을 준비한다.
# DINOv3는 gated 모델이므로 Hugging Face에서 라이선스 승인 후 별도로 내려받아야 한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
LANDMARKER="$MODEL_DIR/face_landmarker.task"

mkdir -p "$MODEL_DIR"

if [[ -s "$LANDMARKER" ]]; then
  echo "[skip] $LANDMARKER already exists"
else
  echo "[download] MediaPipe Face Landmarker"
  curl -L --fail \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    -o "$LANDMARKER"
fi

echo "[ok] Face Landmarker: $LANDMARKER"

if [[ -f "$MODEL_DIR/dinov3/vits16/model.safetensors" ]]; then
  echo "[ok] DINOv3 ViT-S/16: $MODEL_DIR/dinov3/vits16"
else
  echo "[optional] live DINO heatmap requires a local DINOv3 ViT-S/16 checkout at:"
  echo "           $MODEL_DIR/dinov3/vits16"
  echo "           Accept the model license on Hugging Face, then download"
  echo "           facebook/dinov3-vits16-pretrain-lvd1689m into that directory."
fi

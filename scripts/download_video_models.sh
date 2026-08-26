#!/usr/bin/env bash
# ON DAMM 영상 분석기(video analyzer)용 모델/가중치 사전 다운로드 스크립트.
# scripts/download_models.sh 와 같은 스타일. 네트워크 사용은 이 설정 시점뿐이다.
# 멱등(idempotent): 이미 받은 파일은 건너뛰고, 재실행해도 항상 exit 0을 목표로 한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
PY="$ROOT_DIR/.venv/bin/python"

mkdir -p "$MODEL_DIR"

file_bytes() {
  # 포터블 파일 크기(byte). 없으면 0.
  if [[ -f "$1" ]]; then wc -c < "$1" | tr -d ' '; else echo 0; fi
}

# 1) MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes + head pose).
#    이미 1MB 넘게 있으면 재다운로드하지 않는다.
LANDMARKER="$MODEL_DIR/face_landmarker.task"
if (( $(file_bytes "$LANDMARKER") > 1048576 )); then
  echo "[skip] $LANDMARKER 이미 존재 ($(du -h "$LANDMARKER" | cut -f1))"
else
  echo "[download] MediaPipe face_landmarker.task ..."
  curl -L --fail \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    -o "$LANDMARKER"
fi

# 2) ultralytics YOLO26 가중치 미리 받기 (person detection/tracking 용).
#    ultralytics는 현재 디렉토리에 받으므로 models/ 안에서 실행한다.
YOLO_WEIGHTS="$MODEL_DIR/yolo26s.pt"
if [[ -s "$YOLO_WEIGHTS" ]]; then
  echo "[skip] $YOLO_WEIGHTS 이미 존재 ($(du -h "$YOLO_WEIGHTS" | cut -f1))"
else
  echo "[download] ultralytics yolo26s.pt ..."
  (cd "$MODEL_DIR" && "$PY" -c "from ultralytics import YOLO; YOLO('yolo26s.pt')")
fi

# 3) insightface buffalo_l 패키지 미리 받기 (ArcFace 임베딩 용).
#    FaceAnalysis와 동일한 저장소 규칙(~/.insightface/models/buffalo_l)을 따른다.
#    force=False 이므로 이미 있으면 다시 받지 않는다.
echo "[prewarm] insightface buffalo_l (~/.insightface/models/buffalo_l) ..."
"$PY" - <<'PYEOF'
from insightface.utils.storage import ensure_available
path = ensure_available("models", "buffalo_l")
print("[prewarm] buffalo_l ready:", path)
PYEOF

# 4) EmotiEffLib enet_b0_8_va_mtl 체크포인트 미리 받기 (표정+valence/arousal 용).
#    HuggingFace 캐시에 저장되며, 이후 로드 시 네트워크가 필요 없다.
echo "[prewarm] EmotiEffLib enet_b0_8_va_mtl checkpoint ..."
"$PY" - <<'PYEOF'
from emotiefflib.facial_analysis import get_model_path_torch
path = get_model_path_torch("enet_b0_8_va_mtl")
print("[prewarm] enet_b0_8_va_mtl ready:", path)
PYEOF

# 5) ffmpeg 확인. 렌더러(todo 5)가 필요로 하므로 없으면 설치 안내를 출력한다.
#    모델 다운로드 자체는 ffmpeg 없이도 성공해야 하므로 실패로 처리하지 않는다.
if command -v ffmpeg >/dev/null 2>&1; then
  echo "[ok] ffmpeg found: $(command -v ffmpeg)"
else
  echo "[warn] ffmpeg를 PATH에서 찾지 못했다. MP4 인코딩 단계에서 필요하다."
  echo "       macOS (Homebrew): brew install ffmpeg"
  echo "       Ubuntu/Debian   : sudo apt install ffmpeg"
fi

# 다운로드 결과 파일 크기를 확인한다.
ls -lh "$MODEL_DIR"

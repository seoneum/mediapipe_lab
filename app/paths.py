from pathlib import Path

import mediapipe as mp


# 이 파일은 프로젝트 안에서 반복해서 쓰는 경로를 한 곳에 모아 둔 '경로 설정 파일'이다.
# 모델 파일명을 바꾸거나 폴더 구조를 바꾸고 싶으면 다른 코드보다 먼저 여기를 확인한다.

# /Users/seoneum/ai/mediapipe_lab
ROOT = Path(__file__).resolve().parents[1]

# 큰 분류별 폴더. 새 모델/입력/출력 파일도 가능하면 이 구조 안에 둔다.
MODELS = ROOT / "models"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# Holistic: 얼굴 + 포즈 + 양손 landmark를 한 번에 보는 모델.
HOLISTIC_MODEL = MODELS / "holistic_landmarker.task"

# Object Detector: 이미지 안의 사람/물체 bounding box를 찾는 모델.
# 여러 사람 수를 세고 싶다면 Holistic이 아니라 이 계열 detector를 앞단에 붙이는 방식이 필요하다.
OBJECT_MODEL = MODELS / "efficientdet_lite2.tflite"


def require_file(path: Path) -> str:
    """모델 파일이 실제로 있는지 먼저 확인하고 문자열 경로로 돌려준다."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return str(path)


def base_options(model_path: Path) -> mp.tasks.BaseOptions:
    """MediaPipe Tasks가 공통으로 쓰는 BaseOptions를 만든다."""
    return mp.tasks.BaseOptions(
        model_asset_path=require_file(model_path),
        # CPU delegate를 명시해 Apple Silicon/macOS에서 우선 안정성을 확보한다.
        # Web Tasks나 일부 네이티브 경로에서는 GPU delegate 실험을 따로 할 수 있다.
        delegate=mp.tasks.BaseOptions.Delegate.CPU,
    )

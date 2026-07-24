import importlib.metadata
import os
from pathlib import Path

# check_install도 MediaPipe 내부에서 matplotlib/GL 관련 초기화를 건드릴 수 있으므로
# 프로젝트 내부 outputs/.matplotlib를 설정 디렉터리로 사용한다.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
)

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import holistic_landmarker

from paths import HOLISTIC_MODEL, OBJECT_MODEL, base_options


def main() -> None:
    # 1) 설치된 mediapipe 버전과 Python Tasks API 접근 가능 여부를 확인한다.
    print(f"mediapipe={importlib.metadata.version('mediapipe')}")
    print(f"python task BaseOptions={mp.tasks.BaseOptions.__name__}")

    # 2) Object Detector 모델 파일을 실제 Task로 열 수 있는지 확인한다.
    # max_results는 정적 이미지에서 최대 몇 개의 detection을 받을지 정하는 값이다.
    object_options = vision.ObjectDetectorOptions(
        base_options=base_options(OBJECT_MODEL),
        max_results=3,
        running_mode=vision.RunningMode.IMAGE,
    )
    with vision.ObjectDetector.create_from_options(object_options):
        print(f"object detector ok: {OBJECT_MODEL}")

    # 3) Holistic 모델 파일을 실제 Task로 열 수 있는지 확인한다.
    # output_face_blendshapes=True 여야 holistic_camera.py에서 표정 추정용 점수를 받을 수 있다.
    holistic_options = holistic_landmarker.HolisticLandmarkerOptions(
        base_options=base_options(HOLISTIC_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
    )
    with holistic_landmarker.HolisticLandmarker.create_from_options(holistic_options):
        print(f"holistic landmarker ok: {HOLISTIC_MODEL}")


if __name__ == "__main__":
    main()

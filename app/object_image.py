import argparse
import os
from pathlib import Path

# matplotlib/MediaPipe가 만드는 설정 파일을 프로젝트 내부로 고정한다.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
)

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from paths import OBJECT_MODEL, OUTPUTS, base_options


# Object Detector 기본 설정.
# max_results를 키우면 더 많은 물체/사람 후보를 표시한다.
DEFAULT_MAX_RESULTS = 8

# score_threshold를 낮추면 약한 후보도 나오고, 높이면 확실한 후보만 나온다.
DEFAULT_SCORE_THRESHOLD = 0.25


def create_demo_image(path: Path) -> None:
    """테스트 이미지가 없을 때 최소 데모 이미지를 자동 생성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = 255 * __import__("numpy").ones((480, 640, 3), dtype="uint8")
    cv2.putText(
        image,
        "MediaPipe object detector demo",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (40, 40, 40),
        2,
    )
    cv2.rectangle(image, (220, 160), (420, 360), (50, 120, 220), -1)
    cv2.imwrite(str(path), image)


def resolve_path(path_text: str) -> Path:
    """상대경로를 프로젝트 실행 위치 기준의 절대경로로 바꾼다."""
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/images/demo.jpg")
    parser.add_argument("--output", default="outputs/object_demo.jpg")
    parser.add_argument("--score", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    if not image_path.exists():
        create_demo_image(image_path)

    # IMAGE 모드는 단일 이미지 한 장을 동기적으로 처리할 때 쓴다.
    options = vision.ObjectDetectorOptions(
        base_options=base_options(OBJECT_MODEL),
        max_results=args.max_results,
        score_threshold=args.score,
        running_mode=vision.RunningMode.IMAGE,
    )

    # MediaPipe 입력은 mp.Image, OpenCV로 그림을 그릴 출력 이미지는 bgr 배열을 사용한다.
    mp_image = mp.Image.create_from_file(str(image_path))
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    with vision.ObjectDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    # 사람 수를 세고 싶으면 category_name == "person"인 detection만 세는 식으로 확장하면 된다.
    person_count = 0
    for detection in result.detections:
        box = detection.bounding_box
        category = detection.categories[0]
        label = category.category_name or category.display_name or "object"
        if label == "person":
            person_count += 1
        text = f"{label} {category.score:.2f}"
        cv2.rectangle(
            bgr,
            (box.origin_x, box.origin_y),
            (box.origin_x + box.width, box.origin_y + box.height),
            (0, 180, 0),
            2,
        )
        cv2.putText(
            bgr,
            text,
            (box.origin_x, max(20, box.origin_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
        )
        print(text, box)

    print(f"person_count={person_count}")

    output_path = resolve_path(args.output)
    OUTPUTS.mkdir(exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), bgr)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

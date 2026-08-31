import argparse
import os
import time
from pathlib import Path

# MediaPipe / matplotlib가 macOS에서 설정 파일을 홈 디렉터리에 만들려 할 수 있다.
# 프로젝트 안 outputs/.matplotlib 로 고정하면 권한 문제와 설정 파일 흩어짐을 줄일 수 있다.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import holistic_landmarker

from ondamm_facial_movement import analyze_facial_movements
from paths import HOLISTIC_MODEL, base_options


# -----------------------------------------------------------------------------
# 직접 수정하기 좋은 기본값 모음
# -----------------------------------------------------------------------------
# 카메라가 안 잡히면 실행할 때 --camera 0, --camera 1 식으로 바꾸거나 아래 기본값을 바꾼다.
DEFAULT_CAMERA_INDEX = 1

# 영상 해상도. 높일수록 화면은 선명하지만 FPS가 떨어질 수 있다.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# 화면에 그리는 점/선 기본 크기. 실행 옵션 --point-radius, --line-thickness 로도 조절 가능하다.
DEFAULT_POINT_RADIUS = 2
DEFAULT_LINE_THICKNESS = 2

# 표정 blendshape 중 상위 몇 개를 글자로 보여줄지 정한다.
DEFAULT_TOP_EXPRESSIONS = 3

# Holistic Landmarker 신뢰도 기본값.
# 낮추면 더 민감하게 잡지만 오탐이 늘 수 있고, 높이면 더 확실한 결과만 남는다.
DEFAULT_FACE_DETECTION_CONFIDENCE = 0.5
DEFAULT_FACE_LANDMARKS_CONFIDENCE = 0.5
DEFAULT_POSE_DETECTION_CONFIDENCE = 0.5
DEFAULT_POSE_LANDMARKS_CONFIDENCE = 0.5
DEFAULT_HAND_LANDMARKS_CONFIDENCE = 0.5

# 중요:
# 현재 사용하는 HolisticLandmarkerOptions에는 max_num_people, max_num_hands 같은 옵션이 없다.
# 이 Task는 기본적으로 '주 피사체 1명'의 pose/face와 '왼손 1개 + 오른손 1개' landmark를 반환하는 구조로 이해하면 된다.
# 여러 사람 수를 제대로 세려면 이 파일 앞단에 Object Detector 또는 Pose Landmarker를 별도로 붙여야 한다.


# -----------------------------------------------------------------------------
# Landmark 연결선 정의
# -----------------------------------------------------------------------------
# MediaPipe Pose landmark 번호 기준으로 몸통/팔/다리의 어떤 점끼리 선을 이을지 정한다.
# 선을 추가/삭제하고 싶으면 (시작점번호, 끝점번호) 쌍을 수정한다.
POSE_CONNECTIONS = [
    (11, 12),  # 양쪽 어깨
    (11, 13),  # 왼쪽 어깨 -> 왼쪽 팔꿈치
    (13, 15),  # 왼쪽 팔꿈치 -> 왼쪽 손목
    (12, 14),  # 오른쪽 어깨 -> 오른쪽 팔꿈치
    (14, 16),  # 오른쪽 팔꿈치 -> 오른쪽 손목
    (11, 23),  # 왼쪽 어깨 -> 왼쪽 골반
    (12, 24),  # 오른쪽 어깨 -> 오른쪽 골반
    (23, 24),  # 양쪽 골반
    (23, 25),  # 왼쪽 골반 -> 왼쪽 무릎
    (25, 27),  # 왼쪽 무릎 -> 왼쪽 발목
    (24, 26),  # 오른쪽 골반 -> 오른쪽 무릎
    (26, 28),  # 오른쪽 무릎 -> 오른쪽 발목
    (27, 31),  # 왼쪽 발목 -> 왼쪽 발끝 계열
    (28, 32),  # 오른쪽 발목 -> 오른쪽 발끝 계열
]

# MediaPipe Hand landmark 번호 기준.
# 0은 손목, 1~4는 엄지, 5~8은 검지, 9~12는 중지, 13~16은 약지, 17~20은 새끼손가락이다.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Face Mesh에서 iris(동공/홍채) 중심을 대략 잡기 위한 landmark 번호 묶음.
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)

# iris 중심은 정면을 보고 있어도 카메라 높이, 눈꺼풀 모양, 얼굴 각도에
# 따라 약간 위/아래로 치우칠 수 있다. 작은 수직 편향은 center deadband로
# 흡수하고, 명확한 이동만 up/down으로 표시한다.
GAZE_LEFT_THRESHOLD = 0.42
GAZE_RIGHT_THRESHOLD = 0.58
GAZE_UP_THRESHOLD = -0.075
GAZE_DOWN_THRESHOLD = 0.075

# face_blendshapes 점수를 사람이 읽기 쉬운 대략적 표정 라벨로 묶는 규칙.
# 임상적 감정 판정이 아니라 화면 표시용 휴리스틱이다.
EXPRESSION_RULES = [
    ("smile", ["mouthSmileLeft", "mouthSmileRight"]),
    ("surprise", ["jawOpen", "eyeWideLeft", "eyeWideRight", "browInnerUp"]),
    ("blink", ["eyeBlinkLeft", "eyeBlinkRight"]),
    ("frown", ["mouthFrownLeft", "mouthFrownRight", "browDownLeft", "browDownRight"]),
    ("squint", ["eyeSquintLeft", "eyeSquintRight"]),
]
# Micro-expression visualization ROIs.
# MediaPipe Face Mesh 478 landmark indexing.
LEFT_EYE_ROI = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]

RIGHT_EYE_ROI = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

LEFT_BROW_ROI = [
    70, 63, 105, 66, 107,
    55, 65, 52, 53, 46,
]

RIGHT_BROW_ROI = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276,
]

MOUTH_ROI = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 78,
]


def landmark_to_pixel(landmark, width, height):
    """MediaPipe의 정규화 좌표(0~1)를 OpenCV 화면 픽셀 좌표로 변환한다."""
    return int(landmark.x * width), int(landmark.y * height)


def draw_points(frame, landmarks, color, radius=2):
    """landmark 점들을 화면에 원으로 그린다."""
    if not landmarks:
        return
    h, w = frame.shape[:2]
    for landmark in landmarks:
        x, y = landmark_to_pixel(landmark, w, h)
        # 카메라 밖으로 튄 좌표는 그리지 않는다.
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), radius, color, -1)


def face_bbox_info(landmarks, frame):
    """얼굴 landmark로 bbox와 영상 세로 대비 얼굴 높이 비율을 계산한다."""
    if not landmarks:
        return None

    h, w = frame.shape[:2]

    points = np.array(
        [
            landmark_to_pixel(lm, w, h)
            for lm in landmarks
        ],
        dtype=np.int32,
    )

    x1 = int(points[:, 0].min())
    y1 = int(points[:, 1].min())
    x2 = int(points[:, 0].max())
    y2 = int(points[:, 1].max())

    face_height = max(1, y2 - y1)
    face_ratio = face_height / h

    return x1, y1, x2, y2, face_ratio


def draw_roi_mask(
    frame,
    landmarks,
    indices,
    color,
    alpha=0.20,
):
    """landmark 집합으로 반투명 얼굴 ROI mask를 그린다."""
    if not landmarks:
        return

    h, w = frame.shape[:2]

    pts = []

    for index in indices:
        if index >= len(landmarks):
            continue

        pts.append(
            landmark_to_pixel(
                landmarks[index],
                w,
                h,
            )
        )

    if len(pts) < 3:
        return

    pts = np.asarray(
        pts,
        dtype=np.int32,
    )

    # 순서가 완벽한 polygon이 아니어도 안정적으로 영역을 만들기 위해 convex hull.
    hull = cv2.convexHull(pts)

    overlay = frame.copy()

    cv2.fillConvexPoly(
        overlay,
        hull,
        color,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )

    cv2.polylines(
        frame,
        [hull],
        True,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_connections(frame, landmarks, connections, color, thickness=2):
    """landmark 번호 쌍(connections)을 따라 선을 그린다."""
    if not landmarks:
        return
    h, w = frame.shape[:2]
    for start_idx, end_idx in connections:
        # 모델/태스크에 따라 landmark 개수가 다를 수 있으므로 범위를 방어한다.
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmark_to_pixel(landmarks[start_idx], w, h)
        end = landmark_to_pixel(landmarks[end_idx], w, h)
        cv2.line(frame, start, end, color, thickness)


def average_landmark(landmarks, indices):
    """여러 landmark의 평균 위치를 구한다. iris 중심점 계산에 사용한다."""
    valid = [landmarks[index] for index in indices if index < len(landmarks)]
    if not valid:
        return None
    x = sum(landmark.x for landmark in valid) / len(valid)
    y = sum(landmark.y for landmark in valid) / len(valid)
    return x, y


def eye_gaze_ratio(landmarks, iris_indices, corner_indices):
    """눈 양끝 대비 iris가 어느 위치에 있는지 비율로 계산한다."""
    if len(landmarks) <= max(max(iris_indices), *corner_indices):
        return None
    iris = average_landmark(landmarks, iris_indices)
    if iris is None:
        return None
    corner_a = landmarks[corner_indices[0]]
    corner_b = landmarks[corner_indices[1]]
    left_x = min(corner_a.x, corner_b.x)
    right_x = max(corner_a.x, corner_b.x)
    center_y = (corner_a.y + corner_b.y) / 2
    eye_width = max(0.001, right_x - left_x)
    horizontal = (iris[0] - left_x) / eye_width
    vertical = (iris[1] - center_y) / eye_width
    return horizontal, vertical, iris


def estimate_gaze(face_landmarks):
    """iris 위치를 left/right/up/down/center 중 하나로 거칠게 분류한다."""
    if not face_landmarks or len(face_landmarks) < 478:
        return None
    right = eye_gaze_ratio(face_landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS)
    left = eye_gaze_ratio(face_landmarks, LEFT_IRIS, LEFT_EYE_CORNERS)
    if right is None or left is None:
        return None
    horizontal = (right[0] + left[0]) / 2
    vertical = (right[1] + left[1]) / 2

    direction = classify_gaze_direction(horizontal, vertical)

    return {
        "direction": direction,
        "horizontal": horizontal,
        "vertical": vertical,
        "right_iris": right[2],
        "left_iris": left[2],
    }


def classify_gaze_direction(horizontal, vertical):
    """iris 비율을 넓은 center deadband를 둔 카메라 상대 구역으로 분류한다."""
    if horizontal < GAZE_LEFT_THRESHOLD:
        direction = "left"
    elif horizontal > GAZE_RIGHT_THRESHOLD:
        direction = "right"
    elif vertical < GAZE_UP_THRESHOLD:
        direction = "up"
    elif vertical > GAZE_DOWN_THRESHOLD:
        direction = "down"
    else:
        direction = "center"
    return direction


def draw_iris(frame, gaze, radius=4):
    """추정된 양쪽 iris 중심을 빨간 원/십자로 표시한다."""
    if not gaze:
        return
    h, w = frame.shape[:2]
    for key in ("right_iris", "left_iris"):
        x = int(gaze[key][0] * w)
        y = int(gaze[key][1] * h)
        cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
        cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, radius * 3, 1)


def blendshape_scores(face_blendshapes):
    """MediaPipe blendshape category 리스트를 {이름: 점수} dict로 바꾼다."""
    if not face_blendshapes:
        return {}
    return {category.category_name: category.score for category in face_blendshapes}


def top_blendshapes(face_blendshapes, limit):
    """점수가 높은 blendshape만 골라 화면에 표시하기 위해 정렬한다."""
    scores = blendshape_scores(face_blendshapes)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def estimate_expression(face_blendshapes, top_limit=DEFAULT_TOP_EXPRESSIONS):
    """감정이 아닌 관찰 가능한 얼굴 움직임 힌트를 반환한다."""
    analysis = analyze_facial_movements(face_blendshapes, top_limit=top_limit)
    return analysis.primary_label, list(analysis.top_blendshapes)


def put_lines(frame, lines, origin=(16, 32), line_height=28):
    """왼쪽 위에 디버그/상태 텍스트를 여러 줄로 출력한다."""
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        y += line_height


def main() -> None:
    # 실행 예:
    # bash scripts/holistic_camera.sh --camera 1 --point-radius 3 --line-thickness 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--no-lines", action="store_true", help="draw only points, without skeleton lines")
    parser.add_argument("--no-face", action="store_true", help="hide face landmark points")
    parser.add_argument("--no-expression", action="store_true", help="hide face expression estimate")
    parser.add_argument("--no-iris", action="store_true", help="hide iris centers and gaze estimate")
    parser.add_argument("--point-radius", type=int, default=DEFAULT_POINT_RADIUS)
    parser.add_argument("--line-thickness", type=int, default=DEFAULT_LINE_THICKNESS)
    parser.add_argument("--top-expressions", type=int, default=DEFAULT_TOP_EXPRESSIONS)
    parser.add_argument("--face-detection-confidence", type=float, default=DEFAULT_FACE_DETECTION_CONFIDENCE)
    parser.add_argument("--face-landmarks-confidence", type=float, default=DEFAULT_FACE_LANDMARKS_CONFIDENCE)
    parser.add_argument("--pose-detection-confidence", type=float, default=DEFAULT_POSE_DETECTION_CONFIDENCE)
    parser.add_argument("--pose-landmarks-confidence", type=float, default=DEFAULT_POSE_LANDMARKS_CONFIDENCE)
    parser.add_argument("--hand-landmarks-confidence", type=float, default=DEFAULT_HAND_LANDMARKS_CONFIDENCE)
    args = parser.parse_args()

    # MediaPipe Task 설정.
    # running_mode=VIDEO 이므로 매 프레임마다 증가하는 timestamp_ms를 넣어 detect_for_video()를 호출한다.
    options = holistic_landmarker.HolisticLandmarkerOptions(
        base_options=base_options(HOLISTIC_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=args.face_detection_confidence,
        min_face_landmarks_confidence=args.face_landmarks_confidence,
        min_pose_detection_confidence=args.pose_detection_confidence,
        min_pose_landmarks_confidence=args.pose_landmarks_confidence,
        min_hand_landmarks_confidence=args.hand_landmarks_confidence,
        output_face_blendshapes=True,
    )

    # macOS에서는 AVFoundation 백엔드를 명시하면 카메라 접근이 비교적 안정적이다.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    start = time.monotonic()
    with holistic_landmarker.HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("camera read failed")
                break

            # OpenCV는 BGR, MediaPipe는 SRGB/RGB 기준이므로 색상 순서를 바꾼다.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            bbox_info = None

            # 선 먼저 그리고 점을 나중에 그리면 점이 선 위에 보여서 읽기 쉽다.
            if not args.no_lines:
                draw_connections(frame, result.pose_landmarks, POSE_CONNECTIONS, (0, 180, 255), args.line_thickness)
                draw_connections(frame, result.left_hand_landmarks, HAND_CONNECTIONS, (0, 180, 0), args.line_thickness)
                draw_connections(frame, result.right_hand_landmarks, HAND_CONNECTIONS, (180, 0, 180), args.line_thickness)

            draw_points(frame, result.pose_landmarks, (0, 220, 255), args.point_radius + 1)
            if not args.no_face:
                draw_points(frame, result.face_landmarks, (255, 180, 0), max(1, args.point_radius - 1))
            draw_points(frame, result.left_hand_landmarks, (0, 255, 0), args.point_radius + 1)
            draw_points(frame, result.right_hand_landmarks, (255, 0, 255), args.point_radius + 1)
            if result.face_landmarks:
                # ------------------------------------
                # Micro-expression ROI visualization
                # ------------------------------------
                draw_roi_mask(
                    frame,
                    result.face_landmarks,
                    LEFT_EYE_ROI,
                    (255, 120, 0),
                )

                draw_roi_mask(
                    frame,
                    result.face_landmarks,
                    RIGHT_EYE_ROI,
                    (255, 120, 0),
                )

                draw_roi_mask(
                    frame,
                    result.face_landmarks,
                    LEFT_BROW_ROI,
                    (0, 200, 255),
                )

                draw_roi_mask(
                    frame,
                    result.face_landmarks,
                    RIGHT_BROW_ROI,
                    (0, 200, 255),
                )

                draw_roi_mask(
                    frame,
                    result.face_landmarks,
                    MOUTH_ROI,
                    (100, 0, 255),
                )

                bbox_info = face_bbox_info(
                    result.face_landmarks,
                    frame,
                )

                if bbox_info is not None:
                    x1, y1, x2, y2, face_ratio = bbox_info

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        1,
                    )

                    if 0.50 <= face_ratio <= 0.75:
                        face_status = "GOOD"
                    elif face_ratio < 0.50:
                        face_status = "TOO SMALL"
                    else:
                        face_status = "TOO LARGE"

            gaze = None if args.no_iris else estimate_gaze(result.face_landmarks)
            if gaze:
                draw_iris(frame, gaze, args.point_radius + 2)

            # landmark 개수: pose=33, face=478 근처, hand=21이 정상적으로 잡혔을 때의 대표 값이다.
            counts = (
                len(result.pose_landmarks or []),
                len(result.face_landmarks or []),
                len(result.left_hand_landmarks or []),
                len(result.right_hand_landmarks or []),
            )
            detected_people = 1 if result.pose_landmarks else 0
            detected_hands = int(bool(result.left_hand_landmarks)) + int(bool(result.right_hand_landmarks))

            # people은 'Holistic이 잡은 주 피사체 수'라서 0 또는 1이다.
            # hands는 왼손/오른손 landmark가 있는지 기준이라 0~2 범위다.
            overlay_lines = [
                (
                    f"people={detected_people} hands={detected_hands} "
                    f"pose_pts={counts[0]} face_pts={counts[1]} "
                    f"left_pts={counts[2]} right_pts={counts[3]}  q=quit"
                ),
            ]
            if result.face_landmarks and bbox_info is not None:
                overlay_lines.append(
                    f"face_size={face_ratio * 100:.1f}%  "
                    f"capture={face_status}"
                )
            if not args.no_expression:
                expression, blendshapes = estimate_expression(result.face_blendshapes, args.top_expressions)
                shown = ", ".join(f"{name}:{score:.2f}" for name, score in blendshapes[: args.top_expressions])
                overlay_lines.append(f"expression={expression}  {shown}")
            if gaze:
                overlay_lines.append(
                    f"gaze={gaze['direction']} h={gaze['horizontal']:.2f} v={gaze['vertical']:.2f}"
                )
            elif not args.no_iris:
                overlay_lines.append("gaze=unavailable iris landmarks not detected")

            put_lines(frame, overlay_lines)
            cv2.imshow("MediaPipe Holistic", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
